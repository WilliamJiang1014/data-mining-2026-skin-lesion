# KGAE-inspired 异构图/GNN 模型落地开发方案

> 模型名称建议：**LGKE-GNN**，Lesion Graph Knowledge Encoder GNN。  
> 任务场景：基于 ISIC 2024 SLICE-3D Permissive 的皮肤病变恶性风险二分类，并在 PAD-UFES-20 上做外部验证和公平性补充分析。  
> 设计目标：给出一个可以直接进入开发的模型方案，覆盖数据构建、图构建、训练、评估、消融和工程交付。

---

## 1. 设计思想概述

### 1.1 从 KGAE 借鉴什么

KGAE 原始论文的关键思想不是“照搬报告生成器”，而是三点：

1. **知识图谱作为共享潜空间**：图结构不是简单辅助特征，而是视觉信息和语义/临床知识之间的中间空间。
2. **知识驱动编码器**：输入样本通过注意力或图卷积从知识图谱中读取相关知识表示。
3. **可在弱配对或非完全配对场景下利用外部结构知识**：不完全依赖逐样本标注对，而是把领域结构作为归纳偏置。

本项目不是医学报告生成，而是恶性风险二分类，因此将 KGAE 改造成：

- 不使用报告解码器；
- 不做文本重构；
- 将“医学知识图谱”替换为更贴合 ISIC 数据结构的**病灶异构图**；
- 将“知识表示读取”替换为**图消息传递 + 知识原型注意力**；
- 最终输出每个 lesion node 的恶性概率。

### 1.2 模型一句话定义

**LGKE-GNN = 图像编码器 + 元数据编码器 + 病灶异构图 GNN + KGAE-inspired 知识原型注意力 + 二分类风险头。**

模型接收单个病灶的图像和元数据，同时利用同患者、多病灶、解剖部位、年龄段、尺寸、视觉相似性、元数据相似性等图结构，输出恶性概率：

```text
(image, metadata, graph context) -> malignant risk score in [0, 1]
```

---

## 2. 数据构建总流程

### 2.1 原始数据目录约定

建议统一成以下目录，后续所有脚本都从配置文件读取路径。

```text
data/
  raw/
    isic2024_permissive/
      images/
        ISIC_0000000.jpg
        ...
      metadata.csv
      supplemental_metadata.csv
      labels.csv
    pad_ufes_20/
      images/
        img_000001.png
        ...
      metadata.csv
  processed/
    manifest_isic.csv
    manifest_pad.csv
    folds_isic.csv
    tabular_schema_fold0.json
    graph/
      fold0_nodes.parquet
      fold0_edges_train.parquet
      fold0_edges_val.parquet
      fold0_edges_test.parquet
    embeddings/
      fold0_image_train.npy
      fold0_image_val.npy
      fold0_image_test.npy
      fold0_meta_train.npy
      fold0_meta_val.npy
      fold0_meta_test.npy
```

### 2.2 Manifest 构建

核心原则：**所有模型都只读取 manifest，不允许每个模型脚本自己合并原始表。**

#### 2.2.1 ISIC manifest 字段

`manifest_isic.csv` 建议至少包含：

| 字段 | 类型 | 是否进模型 | 用途 |
|---|---:|---:|---|
| `sample_id` / `isic_id` | string | 否 | 主键，连接图像、元数据、标签 |
| `image_path` | string | 是 | 图像读取路径 |
| `target` | int | 是 | 恶性标签，0/1 |
| `patient_id` | string | 不作为普通特征 | 患者级划分、same-patient 图边 |
| `sex` | category | 是 | 元数据特征、属性节点 |
| `age_approx` | float | 是 | 元数据特征、年龄段属性节点 |
| `anatom_site_general` | category | 是 | 元数据特征、部位属性节点 |
| `clin_size_long_diam_mm` | float | 是 | 元数据特征、尺寸属性节点 |
| `tbp_lv_*` | float/category | 是 | TBP Lesion Visualizer 特征 |
| `source` | string | 否 | 数据源标识 |
| `fold` | int | 否 | 患者级交叉验证折号 |

#### 2.2.2 泄漏字段黑名单

以下字段不得进入普通特征或图节点特征：

```text
target, malignant, diagnosis, diagnosis_confirm_type,
iddx_full, iddx_1, iddx_2, iddx_3, iddx_4, iddx_5,
mel_mitotic_index, mel_thick_mm,
image_id, isic_id, lesion_id, patient_id
```

说明：

- `patient_id` 可以用于患者级分组和构建同患者边，但不能作为 one-hot 类别特征。
- `isic_id`、`image_id`、`lesion_id` 只能做索引，不能进模型。
- 任何诊断后字段、病理字段或标签同义字段都删除。

### 2.3 数据校验脚本

开发脚本：

```text
src/data/build_manifest.py
```

脚本需要完成：

1. 检查每条样本是否有图像文件；
2. 检查图像是否可解码；
3. 检查 `sample_id` 是否唯一；
4. 检查标签是否为 0/1；
5. 统计缺失率；
6. 统计患者数、样本数、阳性数、阳性率；
7. 输出数据质量报告。

命令示例：

```bash
python src/data/build_manifest.py \
  --config configs/data_isic.yaml \
  --out data/processed/manifest_isic.csv \
  --report reports/data_quality_isic.md
```

输出报告至少包含：

```text
n_samples
n_patients
n_positive
positive_rate
missing_rate_by_column
invalid_images
columns_removed_as_leakage
```

---

## 3. 患者级数据划分

### 3.1 划分原则

使用患者级分组，避免同一患者的多个病灶同时出现在训练和测试中。

推荐协议：

- 外层：`StratifiedGroupKFold(n_splits=5)`；
- 分组键：`patient_id`；
- 分层标签：`target`；
- 每个 fold 内，从训练患者中再划出 10% 到 15% 作为 validation；
- 所有阈值、早停、超参数和校准都只用 validation。

### 3.2 输出文件

`folds_isic.csv`：

| sample_id | patient_id | target | fold | split |
|---|---|---:|---:|---|
| ISIC_xxx | P001 | 0 | 0 | train |
| ISIC_yyy | P002 | 1 | 0 | val |
| ISIC_zzz | P003 | 0 | 0 | test |

开发脚本：

```text
src/data/make_folds.py
```

命令示例：

```bash
python src/data/make_folds.py \
  --manifest data/processed/manifest_isic.csv \
  --group_col patient_id \
  --label_col target \
  --n_splits 5 \
  --val_ratio 0.1 \
  --seed 42 \
  --out data/processed/folds_isic.csv
```

---

## 4. 按折预处理

### 4.1 基本原则

**必须先划分，再拟合预处理器。**

每个 fold 都有独立的预处理器：

```text
data/processed/preprocessors/fold0_tabular.pkl
data/processed/preprocessors/fold0_image_config.json
data/processed/preprocessors/fold0_knn_index.faiss
data/processed/preprocessors/fold0_kmeans.pkl
```

不得在全量数据上拟合：

- 数值标准化均值/方差；
- 缺失值填充值；
- 类别词表；
- 年龄和尺寸分箱边界；
- KNN 索引；
- KMeans 原型；
- 校准器。

### 4.2 元数据预处理

#### 数值字段

处理顺序：

```text
raw value
-> missing indicator
-> median imputation fitted on train split
-> robust scaling or z-score fitted on train split
```

推荐输入：

```text
age_approx
clin_size_long_diam_mm
tbp_lv_A
tbp_lv_Aext
tbp_lv_B
tbp_lv_Bext
tbp_lv_C
tbp_lv_Cext
tbp_lv_H
tbp_lv_Hext
tbp_lv_L
tbp_lv_Lext
tbp_lv_areaMM2
tbp_lv_area_perim_ratio
tbp_lv_color_std_mean
tbp_lv_deltaA
tbp_lv_deltaB
tbp_lv_deltaL
tbp_lv_deltaLBnorm
tbp_lv_eccentricity
tbp_lv_minorAxisMM
tbp_lv_nevi_confidence
tbp_lv_norm_border
tbp_lv_norm_color
tbp_lv_perimeterMM
tbp_lv_radial_color_std_max
tbp_lv_stdL
tbp_lv_stdLExt
tbp_lv_symm_2axis
tbp_lv_symm_2axis_angle
tbp_lv_x
tbp_lv_y
tbp_lv_z
```

字段列表以真实 `metadata.csv` 为准，脚本要自动跳过不存在字段并写入报告。

#### 类别字段

处理顺序：

```text
raw category
-> fillna("UNK")
-> rare category merge fitted on train split
-> one-hot for LightGBM/XGBoost
-> embedding id for neural model
```

推荐字段：

```text
sex
anatom_site_general
```

可选字段：

```text
tbp_lv_location
tbp_lv_location_simple
```

### 4.3 图像预处理

训练集增强：

```text
Resize to 384 or 512
RandomResizedCrop
HorizontalFlip
VerticalFlip
ColorJitter with small range
RandomErasing with small probability
Normalize by backbone requirement
```

验证、测试、PAD 外部验证：

```text
Resize
CenterCrop
Normalize
```

建议第一版开发直接使用离线图像 embedding，降低 GNN 开发难度：

```bash
python src/features/extract_image_embeddings.py \
  --manifest data/processed/manifest_isic.csv \
  --fold 0 \
  --backbone convnext_tiny \
  --resolution 384 \
  --checkpoint checkpoints/image_fold0_best.pt \
  --out_dir data/processed/embeddings/
```

---

## 5. 异构图数据构建

### 5.1 图的总体定义

对每个 fold 构造一个异构图：

```text
G = (V, E, R)
```

其中：

- `V` 是多类型节点集合；
- `E` 是边集合；
- `R` 是边类型集合；
- 分类目标只定义在 `lesion` 节点上。

### 5.2 节点类型设计

#### 5.2.1 lesion 节点

每个样本一个 lesion 节点。

节点特征：

```text
x_lesion = concat(
  image_embedding,
  metadata_embedding,
  missing_indicators,
  optional_numeric_features
)
```

推荐维度：

```text
image_embedding: 768 or 1024
metadata_embedding: 128
projected hidden dim: 256
```

标签：

```text
y_lesion = target in {0, 1}
```

mask：

```text
train_mask
val_mask
test_mask
```

#### 5.2.2 patient 节点

每个患者一个 patient 节点。

用途：

- 让同一患者多个病灶共享上下文；
- 避免把 `patient_id` 作为普通类别特征；
- 支持“一个患者多病灶”的筛查场景。

节点特征：

```text
patient_feature = aggregate mean of lesion metadata features within current graph
```

注意：

- 训练图中 patient 节点只聚合训练 split 的 lesion；
- 验证/测试查询图中 patient 节点只聚合对应 validation/test 患者自己的 lesion，不读取标签；
- 不允许跨 fold 复用 patient 节点。

#### 5.2.3 attribute 节点

属性节点包括：

```text
sex=male
sex=female
sex=UNK
age_bin=0-29
age_bin=30-39
...
site=head_neck
site=upper_extremity
site=lower_extremity
site=torso
site=UNK
size_bin=Q1
size_bin=Q2
size_bin=Q3
size_bin=Q4
```

用途：

- 给低频属性提供共享参数；
- 将样本和临床上下文连接到同一空间；
- 方便解释：某个样本从哪些属性节点获得消息。

#### 5.2.4 prototype 节点

prototype 节点是 KGAE-inspired 的关键设计。

来源包括三类：

1. `visual_proto_k`：训练折图像 embedding KMeans 聚类中心；
2. `meta_proto_k`：训练折元数据 embedding KMeans 聚类中心；
3. `clinical_proto_k`：人工定义或统计得到的属性组合，如 `site=head_neck + age>60`。

推荐数量：

```text
visual prototypes: 64
metadata prototypes: 32
clinical prototypes: 16-64
total P: 112-160
```

第一版可以只做 visual + metadata prototypes，clinical prototypes 作为第二阶段增强。

### 5.3 边类型设计

| 边类型 | 方向 | 权重 | 作用 |
|---|---|---:|---|
| `lesion -> patient` | 双向 | 1 | 同患者病灶上下文 |
| `lesion -> attribute` | 双向 | 1 | 性别、年龄段、部位、尺寸上下文 |
| `lesion -> visual_proto` | 双向 | cosine similarity | 视觉知识原型读取 |
| `lesion -> meta_proto` | 双向 | RBF / cosine | 元数据知识原型读取 |
| `lesion -> lesion_visual_knn` | 双向 | cosine similarity | 视觉相似病灶上下文 |
| `lesion -> lesion_meta_knn` | 双向 | RBF similarity | 元数据相似病灶上下文 |

### 5.4 KNN 边构建规则

#### 训练图

只在训练 split 内建立 KNN：

```text
train lesion -> train lesion
```

#### 验证/测试查询图

为了避免 validation/test 样本之间相互泄漏评估信息，推荐第一版采用**查询到训练锚点**的方式：

```text
val lesion -> nearest train lesions
test lesion -> nearest train lesions
```

validation/test 节点之间不连 KNN 边，除非在报告中明确说明是 transductive setting。

更严格的版本：

```text
val/test lesion -> prototype nodes only
```

可以作为消融实验的一项。

### 5.5 图构建伪代码

```python
def build_graph_for_fold(manifest, fold, split, preprocessors, image_emb, meta_emb):
    # 1. select train / val / test according to fold file
    df = manifest[manifest["split"] == split].copy()
    train_df = manifest[manifest["split"] == "train"].copy()

    # 2. create lesion nodes
    lesion_nodes = create_lesion_nodes(df, image_emb, meta_emb)

    # 3. create patient nodes for current split only
    patient_nodes = create_patient_nodes(df)
    edges_lp = connect_lesion_patient(df)

    # 4. create attribute nodes using train-fitted vocab/bin edges
    attribute_nodes = load_attribute_nodes(preprocessors)
    edges_la = connect_lesion_attribute(df, preprocessors)

    # 5. create prototype nodes fitted on train only
    prototypes = load_train_prototypes(fold)
    edges_lproto = connect_to_topk_prototypes(df, prototypes, k=3)

    # 6. create knn edges
    if split == "train":
        edges_knn = build_knn_edges_within_train(train_df, k_img=10, k_meta=10)
    else:
        edges_knn = build_query_to_train_knn_edges(df, train_df, k_img=10, k_meta=10)

    # 7. save nodes and edges
    return HeteroGraph(nodes=[lesion_nodes, patient_nodes, attribute_nodes, prototypes],
                      edges=[edges_lp, edges_la, edges_lproto, edges_knn])
```

开发脚本：

```text
src/graph/build_hetero_graph.py
```

命令示例：

```bash
python src/graph/build_hetero_graph.py \
  --manifest data/processed/manifest_isic.csv \
  --fold 0 \
  --image_emb data/processed/embeddings/fold0_image.npy \
  --meta_emb data/processed/embeddings/fold0_meta.npy \
  --k_img 10 \
  --k_meta 10 \
  --n_visual_proto 64 \
  --n_meta_proto 32 \
  --out_dir data/processed/graph/
```

---

## 6. 模型架构

### 6.1 输入与输出

输入：

```text
image: RGB lesion crop
metadata: numeric + categorical metadata
hetero graph: lesion / patient / attribute / prototype nodes and edges
```

输出：

```text
logit_i: real number
p_i = sigmoid(logit_i): malignant probability
```

### 6.2 图像编码器

第一版推荐离线特征：

```text
ConvNeXt-Tiny / EfficientNetV2-S / MONET image encoder
```

输出：

```text
e_img_i in R^768 or R^1024
```

再投影：

```text
z_img_i = Linear(e_img_i) -> LayerNorm -> GELU -> Dropout
z_img_i in R^256
```

### 6.3 元数据编码器

数值特征：

```text
numeric vector -> MLP -> R^128
```

类别特征：

```text
category ids -> Embedding -> concat -> MLP -> R^128
```

融合：

```text
z_meta_i = MLP(concat(num_emb, cat_emb))
z_meta_i in R^256
```

### 6.4 lesion 初始节点表示

```text
h_lesion_i^0 = MLP(concat(z_img_i, z_meta_i))
h_lesion_i^0 in R^256
```

patient、attribute、prototype 节点也投影到同一维度：

```text
h_patient^0 in R^256
h_attribute^0 in R^256
h_prototype^0 in R^256
```

### 6.5 异构图消息传递层

推荐优先实现 PyTorch Geometric 的 `HeteroConv + SAGEConv`，稳定后再尝试 `GATv2Conv` 或 `HGTConv`。

第 `l` 层：

```text
h_i^{l+1} = Norm(
  W_self h_i^l +
  sum over relation r sum over j in N_r(i) alpha_ij^r W_r h_j^l
)
```

其中：

- `r` 是边类型；
- `alpha_ij^r` 来自边权或 attention；
- 每层后接 `GELU + Dropout + Residual`。

推荐配置：

```yaml
hidden_dim: 256
num_layers: 2
conv_type: sage   # sage / gatv2 / hgt
heads: 4
edge_dropout: 0.1
node_dropout: 0.2
```

### 6.6 知识原型注意力模块

这是对 KGAE 的核心迁移。

设 prototype bank：

```text
K = [k_1, k_2, ..., k_P] in R^{P x d}
```

样本查询：

```text
q_i = W_q concat(z_img_i, z_meta_i, h_lesion_i^L)
```

注意力读取：

```text
a_i = softmax(q_i K^T / sqrt(d))
g_i = a_i K
```

输出：

```text
g_i in R^256
```

解释含义：

- `a_i` 可以展示样本最关注的 top-k 原型；
- 原型可以追溯到对应训练聚类中心或属性组合；
- 这比普通 MLP 更容易解释模型借用了哪些“群体知识”。

### 6.7 分类头

最终表示：

```text
u_i = concat(z_img_i, z_meta_i, h_lesion_i^L, g_i)
```

分类：

```text
logit_i = MLP(u_i)
p_i = sigmoid(logit_i)
```

推荐 MLP：

```text
Linear(1024, 512)
LayerNorm
GELU
Dropout(0.3)
Linear(512, 128)
GELU
Linear(128, 1)
```

### 6.8 第一版模型类结构

```text
src/models/lgke_gnn.py

class ImageProjector(nn.Module)
class MetadataEncoder(nn.Module)
class PrototypeAttention(nn.Module)
class HeteroGraphEncoder(nn.Module)
class LGKEGNN(nn.Module)
```

`forward` 输入：

```python
out = model(
    x_dict=data.x_dict,
    edge_index_dict=data.edge_index_dict,
    edge_attr_dict=data.edge_attr_dict,
    lesion_node_ids=batch.n_id_dict["lesion"],
)
```

输出：

```python
{
  "logits": logits,
  "probs": torch.sigmoid(logits),
  "proto_attention": attention_weights,
  "node_embeddings": lesion_embeddings,
}
```

---

## 7. 训练方案

### 7.1 训练分阶段

#### Stage A：训练或准备基础编码器

先实现 M0 到 M4 的基础模型，得到稳定的图像 embedding 和元数据 embedding。

输出：

```text
fold0_image_embeddings.npy
fold0_metadata_embeddings.npy
fold0_image_model.ckpt
fold0_metadata_encoder.pkl
```

#### Stage B：构建异构图

使用 Stage A 的 embedding 构图。

输出：

```text
fold0_train_heterodata.pt
fold0_val_query_heterodata.pt
fold0_test_query_heterodata.pt
```

#### Stage C：训练 LGKE-GNN

训练目标为 lesion 节点二分类。

#### Stage D：验证集校准和阈值选择

只在 validation 上完成：

- temperature scaling 或 Platt scaling；
- Youden 阈值；
- sensitivity ≥ 0.90 阈值；
- pAUC 最优模型选择。

#### Stage E：测试集与外部验证

- ISIC 内部测试：按 fold 输出 metrics；
- PAD 外部验证：只应用 ISIC 训练好的预处理器和模型，不重新拟合。

### 7.2 损失函数

主损失：加权 BCE。

```text
pos_weight = n_negative_train / n_positive_train
```

```python
loss_cls = BCEWithLogitsLoss(pos_weight=pos_weight)(logits, labels)
```

备选：Focal Loss。

```text
gamma = 2.0
alpha = positive class weight
```

图平滑正则，仅用于视觉 KNN 和元数据 KNN 边，不用于 patient 边：

```text
loss_smooth = sum_{(i,j) in E_sim} w_ij ||h_i - h_j||_2^2
```

原型多样性正则，防止所有 prototype 塌缩到相似方向：

```text
loss_proto_div = || normalize(K) normalize(K)^T - I ||_F^2
```

总损失：

```text
loss = loss_cls + lambda_smooth * loss_smooth + lambda_proto * loss_proto_div
```

推荐初值：

```yaml
lambda_smooth: 0.001
lambda_proto: 0.0001
```

如果训练不稳定，第一版先关闭正则，只保留加权 BCE。

### 7.3 采样策略

由于类别极度不均衡，GNN 训练不要直接随机 lesion 节点。

推荐：

```text
Balanced lesion seed sampler
-> positive oversampling
-> NeighborLoader 扩展邻居
```

每个 batch：

```text
seed lesion nodes: 256 or 512
positive ratio target: 0.25 to 0.50
neighbor sizes: [15, 10]
```

注意：

- 过高阳性比例可能造成概率校准偏移，因此校准必须在原始分布 validation 上做；
- 评估时不能重采样，必须对完整 validation/test 预测。

### 7.4 优化器与默认超参数

```yaml
optimizer: AdamW
learning_rate: 0.0003
weight_decay: 0.0001
batch_size_seed_nodes: 512
epochs: 50
warmup_epochs: 3
scheduler: cosine
mixed_precision: true
early_stopping_metric: pAUC@TPR>=0.80
patience: 8
hidden_dim: 256
gnn_layers: 2
prototype_count: 128
dropout: 0.2
seed: [42, 3407, 2025]
```

### 7.5 训练命令

```bash
python src/training/train_lgke_gnn.py \
  --config configs/lgke_gnn.yaml \
  --fold 0 \
  --train_graph data/processed/graph/fold0_train_heterodata.pt \
  --val_graph data/processed/graph/fold0_val_query_heterodata.pt \
  --out_dir experiments/lgke_gnn/fold0/
```

### 7.6 训练日志要求

每个 epoch 保存：

```text
train_loss
val_loss
val_pauc_tpr80
val_auprc
val_auroc
val_sensitivity_youden
val_specificity_youden
val_sensitivity_highsens
val_specificity_highsens
val_brier
val_ece
best_epoch
learning_rate
```

文件：

```text
experiments/lgke_gnn/fold0/train_log.csv
experiments/lgke_gnn/fold0/best.ckpt
experiments/lgke_gnn/fold0/val_predictions.csv
experiments/lgke_gnn/fold0/config_resolved.yaml
```

---

## 8. 测评指标设计

本节指标需要与实验报告最终结果表一致。

### 8.1 主指标

#### 8.1.1 pAUC@TPR≥0.80

这是报告中的主指标，用来强调高召回区间的排序能力。

输出字段：

```text
pauc_tpr80
pauc_tpr80_ci_low
pauc_tpr80_ci_high
```

实现注意：

- 代码中不要硬编码 0.80，写成 `min_tpr` 参数；
- 若后续需要对齐 ISIC 官方指标仓库，可把 `min_tpr` 改成 0.88；
- 报告当前版本默认使用 `min_tpr=0.80`。

#### 8.1.2 AUPRC

由于阳性样本很少，AUPRC 比 AUROC 更能反映少数类检出能力。

输出字段：

```text
auprc
auprc_ci_low
auprc_ci_high
```

#### 8.1.3 AUROC

用于与传统二分类论文对比，但不能单独作为结论。

输出字段：

```text
auroc
auroc_ci_low
auroc_ci_high
```

### 8.2 固定阈值指标

阈值从 validation 得到，然后固定到 test。

#### 阈值 A：Youden 阈值

```text
threshold_youden = argmax_t (sensitivity(t) + specificity(t) - 1)
```

报告：

```text
sensitivity_youden
specificity_youden
fnr_youden
balanced_accuracy_youden
```

#### 阈值 B：高灵敏度阈值

```text
threshold_sens90 = max threshold such that validation sensitivity >= 0.90
```

报告：

```text
sensitivity_sens90
specificity_sens90
fnr_sens90
balanced_accuracy_sens90
```

如果 validation 阳性数太少导致阈值不稳定，报告需要说明并给出 bootstrap 区间。

### 8.3 校准指标

报告：

```text
brier_score
ece_10bins
ece_15bins
calibration_slope
calibration_intercept
```

推荐同时输出校准曲线：

```text
reports/figures/calibration_lgke_gnn_fold0.pdf
```

### 8.4 公平性指标

子群：

```text
sex
age_bin
anatom_site_general
Fitzpatrick skin type  # PAD 外部验证为主
```

每个子群必须报告：

```text
n_samples
n_positive
auroc
auprc
pauc_tpr80
sensitivity_sens90
specificity_sens90
fnr_sens90
brier_score
ece
```

组间差异：

```text
delta_auc = max(AUC_g) - min(AUC_g)
delta_fnr = max(FNR_g) - min(FNR_g)
delta_ece = max(ECE_g) - min(ECE_g)
```

规则：

- 子群阳性数 `< 10` 时，不做强结论，只展示统计和置信区间；
- 公平性结论优先看 FNR 差异，因为筛查任务中漏诊更关键；
- Fitzpatrick 类型主要在 PAD-UFES-20 上报告。

### 8.5 外部验证指标

PAD 外部验证输出：

```text
pad_auprc
pad_auroc
pad_pauc_tpr80
pad_sensitivity_sens90
pad_specificity_sens90
pad_brier
pad_ece
pad_bacc
```

重要：

- 不重新拟合 scaler、encoder、KMeans、KNN、calibrator；
- 不重新选阈值；
- 如果额外做 PAD 阈值重调，需要作为补充实验单独报告。

### 8.6 模型效率指标

报告：

```text
params_million
inference_ms_per_image
peak_gpu_memory_mb
embedding_extraction_time
knn_graph_build_time
```

对于 GNN 模型，还要报告：

```text
number_of_nodes_by_type
number_of_edges_by_type
average_degree_by_relation
neighbor_sampling_fanout
```

### 8.7 模型特有解释指标

LGKE-GNN 需要额外输出：

```text
top_relation_attention_by_sample
top_prototypes_by_sample
edge_type_ablation_delta_pauc
prototype_attention_entropy
```

用于回答：模型到底是不是利用了图结构，而不是只靠图像 embedding。

---

## 9. 实验表格设计

### 9.1 主结果表

| 模型 | 输入 | pAUC@TPR≥0.80 | AUPRC | AUROC | Sens@90 | Spec@90 | Brier | ECE | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| M0 LightGBM | Metadata | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 表格基线 |
| M1 ConvNeXt | Image | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 图像基线 |
| M2 MONET Embedding | Image+TextMeta | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 现代强基线 |
| M3 ViT/Swin | Image | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Transformer 图像模型 |
| M4 Gated Fusion | Image+Metadata | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 常规多模态融合 |
| M5 LGKE-GNN | Image+Metadata+Graph | TBD | TBD | TBD | TBD | TBD | TBD | TBD | KGAE-inspired 图模型 |
| M6 Ensemble | All | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 可选最终集成 |

### 9.2 图结构消融表

| 设置 | pAUC@TPR≥0.80 | AUPRC | AUROC | FNR@Sens90 | 目的 |
|---|---:|---:|---:|---:|---|
| M4，无图 | TBD | TBD | TBD | TBD | 判断图结构总收益 |
| LGKE，仅属性边 | TBD | TBD | TBD | TBD | 属性节点收益 |
| LGKE，属性边 + patient 边 | TBD | TBD | TBD | TBD | 同患者上下文收益 |
| LGKE，属性边 + visual KNN | TBD | TBD | TBD | TBD | 视觉相似病灶收益 |
| LGKE，属性边 + metadata KNN | TBD | TBD | TBD | TBD | 元数据相似病灶收益 |
| LGKE，无 prototype attention | TBD | TBD | TBD | TBD | KGAE-inspired 原型模块收益 |
| LGKE full | TBD | TBD | TBD | TBD | 完整模型 |

### 9.3 外部验证表

| 训练数据 | 测试数据 | 模型 | AUPRC | AUROC | pAUC@TPR≥0.80 | BACC | ECE | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---|
| ISIC | PAD | M4 Fusion | TBD | TBD | TBD | TBD | TBD | 常规融合 |
| ISIC | PAD | M5 LGKE-GNN | TBD | TBD | TBD | TBD | TBD | 外部泛化 |

### 9.4 公平性表

| 数据 | 子群变量 | 子群 | n | positive | AUROC | AUPRC | FNR@Sens90 | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ISIC | sex | male | TBD | TBD | TBD | TBD | TBD | TBD |
| ISIC | sex | female | TBD | TBD | TBD | TBD | TBD | TBD |
| ISIC | site | head/neck | TBD | TBD | TBD | TBD | TBD | TBD |
| PAD | Fitzpatrick | I-II | TBD | TBD | TBD | TBD | TBD | TBD |
| PAD | Fitzpatrick | III-IV | TBD | TBD | TBD | TBD | TBD | TBD |
| PAD | Fitzpatrick | V-VI | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 10. 评估脚本设计

开发脚本：

```text
src/evaluation/metrics.py
src/evaluation/evaluate_predictions.py
src/evaluation/bootstrap.py
src/evaluation/fairness.py
src/evaluation/calibration.py
```

### 10.1 prediction 文件格式

每个模型每个 fold 输出：

```text
experiments/lgke_gnn/fold0/test_predictions.csv
```

字段：

| 字段 | 说明 |
|---|---|
| `sample_id` | 样本 ID |
| `patient_id` | 患者 ID，仅用于 bootstrap 和患者级分析 |
| `target` | 真值 |
| `prob_raw` | 原始概率 |
| `prob_calibrated` | 校准后概率 |
| `logit` | 模型 logit |
| `split` | val/test/external |
| `fold` | fold id |
| `sex` | 公平性变量 |
| `age_bin` | 公平性变量 |
| `anatom_site_general` | 公平性变量 |
| `fitzpatrick` | PAD 公平性变量，可缺失 |

### 10.2 评估命令

```bash
python src/evaluation/evaluate_predictions.py \
  --pred experiments/lgke_gnn/fold0/test_predictions.csv \
  --thresholds experiments/lgke_gnn/fold0/thresholds.json \
  --min_tpr 0.80 \
  --group_col patient_id \
  --out_json experiments/lgke_gnn/fold0/metrics.json \
  --out_tables reports/tables/lgke_gnn_fold0_metrics.csv \
  --out_figures reports/figures/lgke_gnn_fold0/
```

### 10.3 metrics.json 示例

```json
{
  "model": "LGKE-GNN",
  "fold": 0,
  "n_samples": 43500,
  "n_patients": 1200,
  "n_positive": 80,
  "pauc_tpr80": null,
  "auprc": null,
  "auroc": null,
  "threshold_youden": null,
  "threshold_sens90": null,
  "sensitivity_sens90": null,
  "specificity_sens90": null,
  "fnr_sens90": null,
  "brier": null,
  "ece_10bins": null,
  "params_million": null,
  "inference_ms_per_image": null
}
```

---

## 11. 消融实验与开发优先级

### 11.1 必做消融

必须完成以下 5 项，否则无法证明 GNN 模型的贡献：

1. **M4 Fusion vs M5 LGKE-GNN full**：验证图结构是否有总体提升。
2. **去掉 patient 边**：验证同患者上下文是否有效。
3. **去掉 visual KNN 边**：验证视觉相似图是否有效。
4. **去掉 prototype attention**：验证 KGAE-inspired 模块是否有效。
5. **GraphSAGE vs GATv2**：验证边注意力是否带来收益。

### 11.2 可选消融

1. prototype 数量：`P=32/64/128/256`；
2. KNN 邻居数：`K=5/10/20`；
3. 是否使用 PAD 做 domain adaptation；
4. 图像 embedding 来源：ConvNeXt vs MONET vs DINOv2；
5. 是否对图像编码器做 end-to-end 微调。

### 11.3 开发优先级

#### 第 1 优先级：能跑通

```text
manifest -> folds -> metadata/image embedding -> graph -> LGKE-GNN train -> metrics
```

只实现：

```text
lesion nodes
attribute nodes
prototype nodes
visual KNN edges
weighted BCE
pAUC/AUPRC/AUROC
```

#### 第 2 优先级：补全图结构

加入：

```text
patient nodes
metadata KNN edges
prototype attention explanation
calibration
```

#### 第 3 优先级：论文级完整性

加入：

```text
external PAD validation
fairness
bootstrap CI
paired bootstrap significance
edge-type ablation
Streamlit demo explanation
```

---

## 12. 防止数据泄漏的实现清单

开发时逐项打勾。

- [ ] `patient_id` 没有作为普通类别特征输入模型。
- [ ] 同一 `patient_id` 没有跨 train/val/test。
- [ ] scaler、imputer、category vocab 只在 train split 拟合。
- [ ] age/size 分箱边界只在 train split 拟合。
- [ ] KMeans prototype 只在 train split 拟合。
- [ ] FAISS/KNN index 只用 train split 建立。
- [ ] validation/test 样本没有互相连 KNN 边，除非明确声明 transductive。
- [ ] threshold 只由 validation 决定。
- [ ] calibrator 只由 validation 决定。
- [ ] PAD 外部验证没有重新拟合任何 ISIC 预处理器。
- [ ] 结果表来自脚本输出，不手工改数。

---

## 13. 代码结构建议

```text
src/
  data/
    build_manifest.py
    make_folds.py
    preprocess_tabular.py
    preprocess_pad.py
  features/
    train_image_encoder.py
    extract_image_embeddings.py
    extract_metadata_embeddings.py
  graph/
    build_hetero_graph.py
    graph_schema.py
    graph_diagnostics.py
  models/
    tabular.py
    image_cnn.py
    fusion.py
    lgke_gnn.py
  training/
    train_tabular.py
    train_image.py
    train_fusion.py
    train_lgke_gnn.py
  evaluation/
    metrics.py
    calibration.py
    bootstrap.py
    fairness.py
    evaluate_predictions.py
  visualization/
    graph_explain.py
    plot_curves.py
configs/
  data_isic.yaml
  data_pad.yaml
  image_convnext.yaml
  fusion.yaml
  lgke_gnn.yaml
scripts/
  run_fold0_lgke.sh
  run_all_experiments.sh
reports/
  tables/
  figures/
experiments/
  lgke_gnn/
```

---

## 14. 配置文件示例

`configs/lgke_gnn.yaml`：

```yaml
seed: 42
fold: 0

paths:
  manifest: data/processed/manifest_isic.csv
  folds: data/processed/folds_isic.csv
  graph_dir: data/processed/graph/
  image_embeddings: data/processed/embeddings/fold0_image.npy
  metadata_embeddings: data/processed/embeddings/fold0_meta.npy
  out_dir: experiments/lgke_gnn/fold0/

data:
  label_col: target
  group_col: patient_id
  min_tpr: 0.80
  high_sensitivity_target: 0.90

image:
  embedding_dim: 768
  projected_dim: 256

metadata:
  embedding_dim: 128
  projected_dim: 256

prototype:
  n_visual_proto: 64
  n_meta_proto: 32
  n_clinical_proto: 32
  attention_dim: 256

hetero_gnn:
  conv_type: sage
  hidden_dim: 256
  num_layers: 2
  heads: 4
  dropout: 0.2
  edge_dropout: 0.1
  neighbor_sizes: [15, 10]

training:
  epochs: 50
  batch_size_seed_nodes: 512
  positive_seed_ratio: 0.33
  optimizer: adamw
  lr: 0.0003
  weight_decay: 0.0001
  scheduler: cosine
  warmup_epochs: 3
  early_stopping_metric: pauc_tpr80
  patience: 8
  mixed_precision: true
  loss: weighted_bce
  lambda_smooth: 0.001
  lambda_proto: 0.0001

evaluation:
  bootstrap_unit: patient_id
  n_bootstrap: 1000
  calibration: platt
  fairness_cols:
    - sex
    - age_bin
    - anatom_site_general
    - fitzpatrick
```

---

## 15. 一键运行示例

单折：

```bash
bash scripts/run_fold0_lgke.sh
```

脚本内容建议：

```bash
set -e

python src/data/build_manifest.py --config configs/data_isic.yaml
python src/data/make_folds.py --manifest data/processed/manifest_isic.csv --n_splits 5 --seed 42
python src/features/extract_image_embeddings.py --config configs/image_convnext.yaml --fold 0
python src/features/extract_metadata_embeddings.py --config configs/lgke_gnn.yaml --fold 0
python src/graph/build_hetero_graph.py --config configs/lgke_gnn.yaml --fold 0
python src/training/train_lgke_gnn.py --config configs/lgke_gnn.yaml --fold 0
python src/evaluation/evaluate_predictions.py --config configs/lgke_gnn.yaml --fold 0
python src/evaluation/fairness.py --config configs/lgke_gnn.yaml --fold 0
python src/visualization/graph_explain.py --config configs/lgke_gnn.yaml --fold 0
```

---

## 16. 最终报告中应如何描述该模型

可以直接写成：

> 本文提出 LGKE-GNN 作为 KGAE-inspired 的复杂图结构基线。该模型首先利用图像编码器和元数据编码器获得病灶节点表示，然后构建由 lesion、patient、attribute 和 prototype 节点组成的异构图。图中边类型覆盖同患者关系、临床属性关系、视觉近邻关系、元数据近邻关系和知识原型关系。模型通过异构 GraphSAGE/GATv2 进行关系特定的消息传递，并通过知识原型注意力模块读取与当前病灶最相关的群体知识表示。最终，模型将图像表示、元数据表示、图上下文表示和原型知识表示拼接后输出恶性概率。与普通 late fusion 相比，该模型显式利用了 ISIC 2024 数据中患者级、多病灶、解剖部位和视觉相似性结构；与原始 KGAE 相比，本模型保留“图作为共享知识空间”和“知识驱动编码”的思想，但将解码目标替换为监督二分类风险预测。

---

## 17. 风险与应对

### 17.1 图太大导致显存不足

应对：

- 使用离线 embedding；
- 使用 NeighborLoader；
- 限制 KNN 边 `K<=10`；
- 原型数量先用 96 或 128；
- GNN 层数控制在 2 层。

### 17.2 图模型提升不明显

应对：

- 先确认 M4 fusion 是否已经很强；
- 用消融表确认是否某类边引入噪声；
- 减少 KNN K 值；
- 只保留 attribute + prototype；
- 尝试 GATv2 或 HGT；
- 检查 validation pAUC 与 AUPRC 是否同时变化，避免只看 AUROC。

### 17.3 类别不均衡导致模型全部预测阴性

应对：

- 使用 weighted BCE；
- 使用 positive seed oversampling；
- 每个 batch 保证阳性 seed；
- 早停看 pAUC/AUPRC，不看 accuracy；
- 报告 Sens@90 与 FNR。

### 17.4 外部验证掉点严重

应对：

- 不把 PAD 混入主训练，先作为真实域外测试；
- 单独报告 PAD 指标和校准；
- 检查 PAD 图像模态与 ISIC 3D-TBP 的差异；
- 可在补充实验中做 domain adaptation，但不要混入主结果。

---

## 18. 最小可交付版本

如果时间有限，最低限度需要交付：

1. `build_manifest.py`、`make_folds.py`；
2. `extract_image_embeddings.py`；
3. `build_hetero_graph.py`，至少支持 lesion、attribute、prototype、visual KNN；
4. `lgke_gnn.py`，至少支持 HeteroConv + SAGEConv + prototype attention；
5. `train_lgke_gnn.py`，支持 weighted BCE、early stopping、保存预测；
6. `evaluate_predictions.py`，输出 pAUC@TPR≥0.80、AUPRC、AUROC、Sens/Spec/FNR、Brier、ECE；
7. 一张主结果表、一张图消融表、一张公平性表。

完成以上内容后，该模型就能在最终报告中作为“复杂且较新的 KGAE-inspired 图结构多模态模型”成立。
