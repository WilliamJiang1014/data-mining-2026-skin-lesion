# M0–M4 模型实现与训练管理

本文档详细记录本项目中负责的全部工作，包括开发环境配置、数据管线、M0–M4 模型实现、统一训练/评估管线和代码质量修复。

---

## 1. 开发环境

### 1.1 虚拟环境

使用 conda 管理独立 Python 环境，环境名称为 `skin`：

```bash
conda create -n skin python=3.11 -y
conda activate skin
```

当前开发环境信息：

| 项目 | 值 |
|---|---|
| 操作系统 | Windows 11 Pro |
| Python 版本 | 3.11.15 |
| GPU | NVIDIA GeForce RTX 4060 Laptop (8GB VRAM) |
| NVIDIA 驱动 | 591.44 |
| CUDA 驱动版本 | 13.1 |

### 1.2 依赖安装

项目依赖定义在 `requirements.txt`，安装方式：

```bash
pip install -r requirements.txt
```

依赖清单及版本要求：

| 包 | 版本要求 | 用途 |
|---|---|---|
| numpy | >=1.23 | 数值计算 |
| pandas | >=1.5 | 数据处理 |
| pyyaml | >=6.0 | YAML 配置文件读写 |
| scikit-learn | >=1.2 | 预处理、StratifiedGroupKFold、评估指标 |
| torch | >=2.0 | 深度学习框架 |
| torchvision | >=0.15 | 图像变换（Resize、Normalize、DataAugmentation） |
| timm | >=0.9 | 预训练骨干网络（EfficientNet-B0、Swin-Tiny、ConvNeXt-Tiny） |
| lightgbm | >=4.0 | M0 表格模型梯度提升树 |
| pillow | >=10.0 | 图像读取 |
| pyarrow | >=12.0 | Parquet 格式读写 |
| joblib | >=1.2 | 模型序列化 |
| pytest | >=7.0 | 单元测试 |

### 1.3 CUDA 版 PyTorch 安装

默认的 `pip install torch torchvision` 安装的是 **CPU-only** 版本，`torch.cuda.is_available()` 返回 `False`。图像模型（M1/M2/M3/M4）和图模型（M5）需要 GPU 加速，必须安装 CUDA 版本。

验证当前 PyTorch 是否支持 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available())"
# 期望输出: True
```

安装 CUDA 版本 PyTorch：

```bash
# 方式一：pip（推荐，CUDA 12.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 方式二：conda
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
```

CUDA 版本兼容性说明：

- NVIDIA 驱动版本 >= 525.60.13 支持 CUDA 12.x
- 运行 `nvidia-smi` 查看驱动支持的最高 CUDA 版本
- PyTorch 的 CUDA 版本只需 <= 驱动支持的 CUDA 版本即可

---

## 2. 数据管线

### 2.1 数据集

| 数据集 | 用途 | 样本数 |
|---|---|---|
| ISIC 2024 SLICE-3D Permissive | 主数据集，内部训练与评估 | 217,477 |
| PAD-UFES-20 | 外部验证，不参与训练 | — |

### 2.2 数据划分

本文采用**患者级分层 5 折划分**。每一折中，约 20% 患者作为测试集；其余约 80% 患者中再划分约 10% 作为验证集，因此每折样本比例约为 72% 训练集、8% 验证集、20% 测试集。该策略避免同一患者同时出现在训练、验证和测试集中，并尽量保持恶性/良性标签比例一致。

具体实现（`src/skin_lesion_risk/data/splits.py`）：

1. **外层划分**：使用 `StratifiedGroupKFold(n_splits=5)` 按 `patient_id` 分组，保证同一患者的所有病灶样本不跨折泄漏，同时通过分层采样保持各折恶性/良性标签比例一致。每折约 20% 患者进入测试集。
2. **内层划分**：从训练+验证池中按 `val_ratio=0.1` 使用 `StratifiedShuffleSplit` 划分验证集，同样按患者分组，确保同一患者不同时出现在训练集和验证集中。
3. **泄漏断言**：每折均通过 `assert_no_group_overlap()` 断言检查，确保 train/val/test 三个集合之间无任何患者重叠。

实际各折样本分布：

| Fold | 训练集 | 验证集 | 测试集 |
|---|---|---|---|
| 0 | 154,204 (70.9%) | 19,778 (9.1%) | 43,495 (20.0%) |
| 1 | 157,975 (72.6%) | 16,007 (7.4%) | 43,495 (20.0%) |
| 2 | 159,530 (73.4%) | 14,453 (6.6%) | 43,494 (20.0%) |
| 3 | 154,527 (71.1%) | 19,451 (8.9%) | 43,499 (20.0%) |
| 4 | 152,966 (70.3%) | 21,017 (9.7%) | 43,494 (20.0%) |

划分产物：`data/processed/folds_isic.csv`，包含 `sample_id`、`patient_id`、`target`、`fold`、`split` 五列。

### 2.3 数据预处理

实现 `FoldTabularPreprocessor`（`src/skin_lesion_risk/data/preprocessing.py`），每折独立拟合，仅在训练子集上学习预处理参数：

**数值特征处理**（35 个 TBP 数值字段，如 `age`、`size_mm`、`tbp_lv_A` 等）：

- 中位数填充缺失值
- IQR（四分位距）缩放：`(value - median) / IQR`
- 为每个数值字段生成缺失指示列 `{column}__missing`

**类别特征处理**（4 个字段：`sex`、`anatom_site`、`tbp_lv_location`、`tbp_lv_location_simple`）：

- 缺失值填充为 "UNK"
- 频次低于 `rare_min_count=20` 的类别合并为 "RARE"
- One-hot 编码，词表固定为 `["UNK", "RARE", ...kept_categories]`

**分位数分箱**：

- 年龄（`age`）和尺寸（`size_mm`）按训练集四分位数分箱，用于公平性子群分析
- 分箱边界仅在训练集上拟合

**序列化**：pickle 格式，每折一个文件 `data/processed/preprocessors/fold{k}_tabular.pkl`。

### 2.4 图像校验与质量报告

- `validate_image_paths()`（`src/skin_lesion_risk/data/quality.py`）：支持 `none`/`sample`/`all` 三种模式校验图像路径是否存在
- `manifest_quality_summary()`：生成数据质量摘要报告
- 产物：`reports/data_quality_isic.md`、`reports/data_quality_pad.md`、`reports/tables/split_stats.csv`

### 2.5 运行命令

```bash
# 完整数据准备
python scripts/prepare_data.py --image-check sample

# PAD 数据集可选（如未下载）
python scripts/prepare_data.py --image-check sample --skip-pad
```

---

## 3. 统一模型工厂

所有模型通过统一工厂管线创建：**YAML 配置 → ModelRegistry → BaseModelAdapter 子类**。

### 3.1 核心文件

| 文件 | 作用 |
|---|---|
| `src/skin_lesion_risk/models/base.py` | `ModelBatch`（统一输入）、`PredictionResult`（统一输出）、`BaseModelAdapter` 基类 |
| `src/skin_lesion_risk/models/registry.py` | 类型字符串到适配器类的映射 |
| `src/skin_lesion_risk/models/factory.py` | `ModelFactory.create()` 加载配置创建模型；`default_registry()` 注册所有类型 |

### 3.2 统一输入：ModelBatch

所有模型的训练和推理输入均封装为 `ModelBatch`，包含以下字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_ids` | list[str] | 样本唯一标识，决定 batch 样本顺序 |
| `labels` | np.ndarray | 标签（0=良性，1=恶性），推理时可为空 |
| `image_paths` | list[str] | 图像文件路径，图像/多模态模型使用 |
| `metadata` | DataFrame | 预处理后的表格特征，表格/多模态模型使用 |
| `raw_metadata` | DataFrame | 原始临床字段（age、sex、anatom_site 等），M2 文本提示使用 |
| `graph` | dict | 图结构数据，M5 图模型使用 |
| `groups` | dict | 公平性分组（sex、anatom_site、age_bin、size_bin） |
| `fold` | int | 折号 |
| `source` | str | 数据源名称 |

### 3.3 统一输出：PredictionResult

所有模型通过 `predict_proba()` 返回 `PredictionResult`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_ids` | list[str] | 与输入 batch 一一对应 |
| `scores` | np.ndarray | 每个样本的恶性风险分数 [0, 1] |
| `labels` | np.ndarray | 真实标签 |

### 3.4 模型注册表

`default_registry()` 注册了全部 8 种模型类型：

| 类型键 | 适配器类 | 文件 | torch 依赖 |
|---|---|---|---|
| `constant` | ConstantRiskModel | adapters/tabular.py | 否 |
| `tabular` | PlaceholderTabularModel | adapters/tabular.py | 否 |
| `tabular_lgbm` | LightGBMTabularModel | adapters/tabular.py | 否 |
| `image` | CNNImageModel | adapters/image.py | 是（延迟导入） |
| `image_transformer` | TransformerImageModel | adapters/image.py | 是（延迟导入） |
| `monet_feature` | MonetFeatureModel | adapters/monet.py | 是（延迟导入） |
| `multimodal` | MultimodalFusionModel | adapters/multimodal.py | 是（延迟导入） |
| `graph_multimodal` | LGKEGNNModelAdapter | adapters/graph.py | 是（延迟导入） |

torch 依赖的模型在 `default_registry()` 中通过 `try/except ImportError` 延迟注册，M0 表格模型无需 torch 即可运行。

### 3.5 添加新模型的步骤

1. 在 `src/skin_lesion_risk/models/adapters/` 中实现适配器类，继承 `BaseModelAdapter`
2. 在 `factory.py` 的 `default_registry()` 中注册类型字符串
3. 新增 YAML 配置文件到 `configs/models/`
4. 在 `configs/experiments/baselines.yaml` 的 models 列表中添加条目

---

## 4. 模型实现详情

### 4.1 M0-1 先验常数基线（ConstantRiskModel）

**文件**：`src/skin_lesion_risk/models/adapters/tabular.py`

| 项目 | 说明 |
|---|---|
| 类型键 | `constant` |
| 方法 | 预测所有样本为训练集阳性率常数 |
| 输入 | 仅 `labels` |
| 输出 | 所有样本输出同一分数 = `train.labels.mean()` |
| 保存格式 | pickle (`model.pkl`) |
| 配置文件 | `configs/models/m0_constant.yaml` |
| 硬件需求 | CPU，秒级完成 |

配置参数：

```yaml
type: constant
params:
  score: 0.5          # 初始值，fit 后会被训练集阳性率覆盖
```

设计目的：作为最低基线，验证评估管线端到端正确性。任何有效模型的指标都应优于此基线。

### 4.2 M0-2 LightGBM 表格模型（LightGBMTabularModel）

**文件**：`src/skin_lesion_risk/models/adapters/tabular.py`

| 项目 | 说明 |
|---|---|
| 类型键 | `tabular_lgbm` |
| 方法 | LightGBM 梯度提升树，使用预处理后的表格特征 |
| 输入 | `metadata`（预处理后的数值+类别特征） |
| 输出 | `predict_proba()` 的正类概率 |
| 保存格式 | pickle (`model.pkl`) |
| 配置文件 | `configs/models/m0_lightgbm.yaml` |
| 硬件需求 | CPU，分钟级完成 |

配置参数：

```yaml
type: tabular_lgbm
params:
  preferred_backend: lightgbm         # 优先使用 LightGBM
  allow_sklearn_fallback: true        # LightGBM 不可用时回退到 sklearn HistGradientBoosting
  n_estimators: 300                   # 提升迭代次数
  learning_rate: 0.03                 # 学习率
  num_leaves: 31                      # 叶节点数
  subsample: 0.9                      # 行采样比例
  colsample_bytree: 0.9               # 列采样比例
  class_weight: balanced              # 类别加权（自动按样本比例调整）
  seed: 2026
```

特征处理：

- 自动删除 ID 列（`patient_id`、`sample_id`、`lesion_id`、`image_id`、`isic_id`）
- 所有列强制转为数值类型，缺失值填充 0.0
- fit 时记录特征名列表，transform 时保证特征对齐（缺失列补 0，多余列删除）

回退机制：当 LightGBM 不可用时，自动回退到 scikit-learn 的 `HistGradientBoostingClassifier`，并通过 `backend_used` 字段记录实际使用的后端。

### 4.3 M1 CNN 图像模型（CNNImageModel）

**文件**：`src/skin_lesion_risk/models/adapters/image.py`

| 项目 | 说明 |
|---|---|
| 类型键 | `image` |
| 骨干网络 | EfficientNet-B0（timm，ImageNet pretrained） |
| 特征维度 | 1280 |
| 分类头 | Dropout(0.3) → Linear(1280, 256) → GELU → Dropout(0.15) → Linear(256, 1) |
| 损失函数 | BCEWithLogitsLoss（pos_weight 加权，权重 = 负样本数/正样本数） |
| 优化器 | AdamW |
| 学习率调度 | CosineAnnealingLR |
| 早停策略 | 基于验证集 pAUC@TPR≥0.80，patience=5 |
| 保存格式 | torch checkpoint (`best.ckpt`) |
| 配置文件 | `configs/models/m1_cnn.yaml` |
| 硬件需求 | GPU 推荐，约 20 epochs |

配置参数：

```yaml
type: image
params:
  encoder: efficientnet_b0            # timm 骨干网络名称
  image_size: 224                     # 输入图像尺寸
  pretrained: true                    # 使用 ImageNet 预训练权重
  loss: weighted_bce                  # 加权二元交叉熵
  epochs: 20
  patience: 5
  learning_rate: 1e-4
  weight_decay: 1e-4
  batch_size: 32
  dropout: 0.3
  seed: 2026
```

图像增强（训练时）：

- Resize(224×224)
- RandomHorizontalFlip
- RandomVerticalFlip
- RandomRotation(30°)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)
- ToTensor + Normalize(ImageNet 均值/标准差)

图像增强（验证/测试时）：

- Resize(224×224)
- ToTensor + Normalize(ImageNet 均值/标准差)

训练流程：

1. 构建 `LesionImageDataset`，从 `image_paths` 实时加载图像
2. 每个 epoch 遍历 DataLoader，前向传播 → 计算损失 → 反向传播 → 参数更新
3. 每 epoch 在验证集上计算 pAUC@TPR≥0.80、AUROC、AUPRC、Brier、ECE 等指标
4. 保存最优模型 state_dict 到 `best.ckpt`，同时记录 `train_log.csv` 和 `loss_curve.csv`
5. 早停触发后加载最优权重

### 4.4 M2 MONET 特征模型（MonetFeatureModel）

**文件**：`src/skin_lesion_risk/models/adapters/monet.py`

| 项目 | 说明 |
|---|---|
| 类型键 | `monet_feature` |
| 图像编码器 | frozen EfficientNet-B0（timm，pretrained，参数冻结不参与训练） |
| 图像特征维度 | 1280 |
| 文本特征维度 | 64 |
| 分类头 | MLP: [1280+64] → Linear(1344, 256) → GELU → Dropout(0.3) → Linear(256, 64) → GELU → Dropout(0.15) → Linear(64, 1) |
| 损失函数 | BCEWithLogitsLoss（pos_weight 加权） |
| 优化器 | AdamW（仅训练 MLP head 参数） |
| 学习率调度 | CosineAnnealingLR |
| 早停策略 | 基于验证集 pAUC@TPR≥0.80，patience=5 |
| 保存格式 | torch checkpoint (`best.ckpt`) |
| 配置文件 | `configs/models/m2_monet_feature.yaml` |
| 硬件需求 | GPU 推荐，约 20 epochs |

配置参数：

```yaml
type: monet_feature
params:
  encoder: efficientnet_b0            # 冻结的图像编码器
  image_size: 224
  pretrained: true
  freeze_encoder: true                # 编码器参数冻结
  use_text_prompts: true              # 启用文本提示特征
  classifier: mlp
  epochs: 20
  patience: 5
  learning_rate: 1e-3
  weight_decay: 1e-4
  batch_size: 64
  dropout: 0.3
  seed: 2026
```

训练流程：

1. **批量提取图像特征**：将所有训练/验证图像通过 frozen EfficientNet-B0 前向传播，提取 1280 维特征向量
2. **构建文本提示特征**：使用 `raw_metadata`（原始临床字段，非预处理后的一热编码）构建临床描述文本，通过稳定哈希（`hashlib.md5`）编码为 64 维向量
3. **拼接特征**：[image_features(1280) + text_features(64)] = 1344 维
4. **训练 MLP head**：在拼接特征上训练 3 层 MLP 分类器
5. 保存时同时保存 head state_dict 和 encoder state_dict，确保加载时特征空间一致

文本提示构建（`src/skin_lesion_risk/features/text_prompts.py`）：

- 从 `raw_metadata` 提取原始临床字段：`age`、`sex`、`anatom_site`、`size_mm`
- 可选附加 TBP 视觉元数据摘要：`color_std_mean`、`norm_border`、`norm_color`、`symm_2axis`
- 生成格式：`"Clinical skin lesion image with patient age approximately 55, sex male, anatomical site torso, long diameter 3.5 mm. TBP visual metadata: color variation 2.1, border irregularity 1.8."`

稳定哈希编码：

- 使用 `hashlib.md5` 替代 Python 内置 `hash()`，确保跨进程可复现（Python `hash()` 受 `PYTHONHASHSEED` 随机化影响）
- 实现方式：将 token 通过 md5 哈希映射到 128 维投影矩阵的行索引，累加投影向量后 L2 归一化

> **注意**：当前 M2 使用 frozen EfficientNet-B0 + hash text features，不是真正的 MONET/VLM。报告中标注为 "frozen encoder baseline"，后续接入真实 MONET 权重时替换 encoder 即可。

### 4.5 M3 视觉 Transformer 模型（TransformerImageModel）

**文件**：`src/skin_lesion_risk/models/adapters/image.py`（与 M1 共享文件）

| 项目 | 说明 |
|---|---|
| 类型键 | `image_transformer` |
| 骨干网络 | Swin-Tiny（timm，`swin_tiny_patch4_window7_224`，ImageNet pretrained） |
| 特征维度 | 768 |
| 分类头 | Dropout(0.3) → Linear(768, 256) → GELU → Dropout(0.15) → Linear(256, 1) |
| 损失函数 | Focal Loss（alpha=0.25, gamma=2.0, pos_weight 加权） |
| 优化器 | AdamW |
| 学习率调度 | CosineAnnealingLR |
| 早停策略 | 基于验证集 pAUC@TPR≥0.80，patience=4 |
| 保存格式 | torch checkpoint (`best.ckpt`) |
| 配置文件 | `configs/models/m3_transformer.yaml` |
| 硬件需求 | GPU 推荐，约 15 epochs |

配置参数：

```yaml
type: image_transformer
params:
  encoder: swin_tiny_patch4_window7_224    # Swin Transformer 骨干
  image_size: 224
  pretrained: true
  loss: focal                               # Focal Loss
  focal_alpha: 0.25                         # 正类权重
  focal_gamma: 2.0                          # 难易样本聚焦参数
  epochs: 15
  patience: 4
  learning_rate: 5e-5                       # 较小学习率（Transformer 微调）
  weight_decay: 0.05                        # 较大权重衰减
  batch_size: 16                            # 较小 batch（Swin 显存占用大）
  dropout: 0.3
  seed: 2026
```

Focal Loss 实现（`image.py` 中的 `FocalLoss` 类）：

```
loss = alpha * (1 - p_t)^gamma * BCE_with_logits
```

- `alpha=0.25`：降低正类（恶性）的损失权重，缓解类别不均衡
- `gamma=2.0`：降低易分类样本的损失贡献，聚焦难分类样本
- 结合 `pos_weight` 加权，进一步调整正负样本比例

Swin-Tiny 相比 EfficientNet-B0 的特点：

- 基于移位窗口的自注意力机制，能捕获全局上下文信息
- 参数量更大，需要更小学习率和更大权重衰减
- 显存占用更高，batch_size 设为 16（EfficientNet-B0 为 32）

### 4.6 M4 多模态融合模型（MultimodalFusionModel）

**文件**：`src/skin_lesion_risk/models/adapters/multimodal.py`

| 项目 | 说明 |
|---|---|
| 类型键 | `multimodal` |
| 图像编码器 | ConvNeXt-Tiny（timm，pretrained） |
| 图像特征维度 | 768 |
| 元数据编码器 | MLP: MetadataEncoder（input_dim → 128 → 64） |
| 元数据输出维度 | 64 |
| 融合策略 | Gated Fusion |
| 融合后维度 | 768 + 64 = 832 |
| 分类头 | Linear(832, 256) → GELU → Dropout(0.3) → Linear(256, 64) → GELU → Dropout(0.15) → Linear(64, 1) |
| 损失函数 | BCEWithLogitsLoss（pos_weight 加权） |
| 优化器 | AdamW（所有模块联合优化） |
| 学习率调度 | CosineAnnealingLR |
| 早停策略 | 基于验证集 pAUC@TPR≥0.80，patience=5 |
| 保存格式 | torch checkpoint (`best.ckpt`) |
| 配置文件 | `configs/models/m4_multimodal.yaml` |
| 硬件需求 | GPU 推荐，约 20 epochs |

配置参数：

```yaml
type: multimodal
params:
  image_encoder: convnext_tiny             # 图像编码器
  image_size: 224
  pretrained: true
  metadata_encoder: mlp                    # 元数据编码器类型
  metadata_hidden_dim: 128                 # 元数据 MLP 隐藏层维度
  metadata_output_dim: 64                  # 元数据特征输出维度
  fusion: gated                            # 融合策略
  gate_hidden_dim: 128                     # 门控网络隐藏层维度
  loss: weighted_bce
  epochs: 20
  patience: 5
  learning_rate: 1e-4
  weight_decay: 1e-4
  batch_size: 32
  dropout: 0.3
  seed: 2026
```

**Gated Fusion 机制**（`multimodal.py` 中的 `GatedFusion` 类）：

```
gate_weights = Sigmoid(Linear(GELU(Linear([image_features; metadata_features]))))
gated_image = gate_weights * image_features
fused = [gated_image; metadata_features]
```

- 门控网络接收图像和元数据特征的拼接作为输入
- 输出与图像特征维度相同的 sigmoid 门控权重
- 对图像特征逐元素加权，保留元数据特征的完整信息
- 门控权重让模型学习在哪些维度上依赖图像信息，哪些维度上依赖元数据信息

**MultimodalDataset**：同时返回 `(image, metadata_row, label)` 三元组，保证 DataLoader 的 shuffle 操作不会导致图像和元数据错位。

**联合优化**：图像编码器、元数据编码器、门控模块和分类头的参数全部参与梯度更新（与 M2 的 frozen encoder 策略不同）。

### 4.7 M5 LGKE-GNN（林青澄实现）

**文件**：`src/skin_lesion_risk/models/adapters/graph.py`

| 项目 | 说明 |
|---|---|
| 类型键 | `graph_multimodal` |
| 实现者 | 林青澄 |
| 骨干 | RelationSAGEBlock + PrototypeAttention + 3 层 MLP |
| 图边类型 | same_patient, has_attribute, visual_knn, metadata_knn, visual_prototype, metadata_prototype（6 种） |
| 损失函数 | BCEWithLogitsLoss（pos_weight 加权）+ 可选 graph_smoothing_loss |
| PAD 外部验证 | 已实现 |
| 域适应 | 已实现（source-weighted loss） |

---

## 5. 训练与评估管线

### 5.1 统一训练脚本

`scripts/run_train.py` 统一支持 M0–M5 训练：

```bash
# M0（CPU，秒级/分钟级）
python scripts/run_train.py --model m0_constant --fold 0
python scripts/run_train.py --model m0_lightgbm --fold 0

# M1–M4（需 GPU）
python scripts/run_train.py --model m1_cnn_baseline --fold 0 --device cuda
python scripts/run_train.py --model m2_monet_feature_baseline --fold 0 --device cuda
python scripts/run_train.py --model m3_transformer_baseline --fold 0 --device cuda
python scripts/run_train.py --model m4_multimodal_fusion --fold 0 --device cuda

# M5（需先构建图）
python scripts/build_graph.py --fold 0
python scripts/run_train.py --model m5_lgke_gnn --fold 0 --device cuda
```

训练脚本功能：

- 自动从 `manifest_isic.csv` 和 `folds_isic.csv` 构建各折的 `ModelBatch`
- 根据 `include_metadata` 逻辑决定是否传入 `metadata` 和 `raw_metadata`
  - 需要 metadata 的模型：`tabular_lgbm`、`monet_feature`、`multimodal`
  - 不需要 metadata 的模型：`constant`、`image`、`image_transformer`
- 图像模型自动设置 `device` 和 `out_dir` 参数
- 训练前清理旧的产物文件，避免残留
- 保存 `config_resolved.yaml` 记录实际运行配置
- 训练完成后自动评估验证集和测试集
- 输出 `metrics.json`、`train_log.csv`、`train_summary.csv`、`val_predictions.csv`、`test_predictions.csv`
- 自动更新主结果表 `reports/tables/main_results.csv`

命令行参数覆盖：

```bash
python scripts/run_train.py --model m1_cnn_baseline --fold 0 --device cuda \
  --epochs 30 --patience 7 --learning-rate 5e-5 --batch-size 16
```

### 5.2 统一评估脚本

`scripts/evaluate.py` 支持所有模型的离线评估：

```bash
# M0（无需 torch）
python scripts/evaluate.py --model m0_constant --fold 0
python scripts/evaluate.py --model m0_lightgbm --fold 0

# M1–M4（需 torch）
python scripts/evaluate.py --model m1_cnn_baseline --fold 0 --device cuda
python scripts/evaluate.py --model m2_monet_feature_baseline --fold 0 --device cuda
python scripts/evaluate.py --model m3_transformer_baseline --fold 0 --device cuda
python scripts/evaluate.py --model m4_multimodal_fusion --fold 0 --device cuda

# M5
python scripts/evaluate.py --model m5_lgke_gnn --fold 0 --device cuda
```

评估脚本功能：

- M0 评估无需 torch（torch 和图像 adapter 延迟导入到函数内部）
- 图像/图模型加载后通过 `_move_model_to_device()` 统一迁移所有子模块到目标设备
- 支持 CUDA 训练 → CPU 评估的跨设备场景（模型 load 默认保持 CPU）

### 5.3 产物路径约定

所有模型训练与评估产物统一写入：

```text
data/artifacts/trained_models/{model_name}/fold{f}/
  best.ckpt 或 model.pkl          # 最佳模型权重/传统模型序列化
  config_resolved.yaml            # 实际运行配置
  train_log.csv                   # epoch 级训练日志（深度模型）
  train_summary.csv               # 一行训练摘要
  loss_curve.csv                  # 训练/验证 loss 曲线（深度模型）
  val_predictions.csv             # 验证集预测
  test_predictions.csv            # 测试集预测
  metrics.json                    # 验证集和测试集指标
```

主结果表：`reports/tables/main_results.csv`

### 5.4 汇总报告

```bash
python scripts/make_report_assets.py
```

汇总所有模型的训练结果到主结果表和报告图表。

---

## 6. 评价指标

### 6.1 指标体系

统一评价指标定义在 `src/skin_lesion_risk/evaluation/`：

| 指标 | 模块 | 说明 |
|---|---|---|
| pAUC@TPR≥0.80 | metrics.py | 部分 AUC（主评价指标），在灵敏度 ≥ 0.80 区域内计算 ROC 面积并归一化 |
| AUROC | metrics.py | ROC 曲线下面积（sklearn） |
| AUPRC | metrics.py | PR 曲线下面积（sklearn） |
| Sensitivity | metrics.py | 灵敏度（真阳性率） |
| Specificity | metrics.py | 特异度（真阴性率） |
| Precision | metrics.py | 精确率 |
| F1 | metrics.py | F1 分数 |
| FNR | metrics.py | 假阴性率 |
| Brier | metrics.py | Brier 分数（概率校准） |
| ECE | calibration.py | 期望校准误差（分箱后预测概率与实际频率的偏差） |
| Subgroup Gap | metrics.py | 子群指标差异（公平性评估） |

pAUC@TPR≥0.80 计算方式：

- 使用 sklearn `roc_curve` 获取 FPR/TPR 曲线
- 截取 TPR ≥ 0.80 区域
- 计算该区域内 ROC 曲线与 TPR=0.80 水平线之间的面积
- 归一化：`raw_area / (1.0 - 0.80)`，结果范围 [0, 1]
- numpy 2.0 兼容：使用 `np.trapezoid`（fallback `np.trapz`）

### 6.2 阈值规则

阈值在验证集上确定，然后应用到测试集和外部验证集：

| 规则 | 说明 |
|---|---|
| Youden | 最大 Youden 指数 = max(sensitivity + specificity - 1) 对应的阈值 |
| Sensitivity≥0.90 | 满足验证集灵敏度不低于 0.90 的最低阈值 |

不在测试集或外部验证集上重新选择阈值。

### 6.3 公平性评估

子群指标差异（`subgroup_metric_gaps()`）：

- 按性别（sex）、解剖部位（anatom_site）、年龄分箱（age_bin）、尺寸分箱（size_bin）分组
- 计算每组内的 sensitivity 和 specificity
- 报告 max - min 差异（gap）

---

## 7. 泄漏控制

实现中严格遵守以下规则：

1. **ID 列不入特征**：`patient_id`、`lesion_id`、`sample_id`、`image_id`、`isic_id` 不作为模型输入特征，仅用于划分和审计
2. **预处理仅训练拟合**：scaler、imputer、encoder、分箱边界、类别词表仅在训练折拟合，验证/测试/外部数据只做 transform
3. **诊断字段不入特征**：诊断文本、活检结果、病理确认方式、目标标签同义字段不作为输入
4. **图构建仅用训练折**：kNN 索引、原型节点仅由训练折数据构建；验证/测试样本通过特征接入训练图
5. **阈值仅验证集选择**：决策阈值只在验证集上确定

---

## 8. 测试

```bash
pytest  # 8/8 通过
```

测试文件：

| 文件 | 测试内容 |
|---|---|
| `tests/test_model_factory.py` | 工厂创建模型 smoke test、所有 8 种类型注册验证 |
| `tests/test_data_pipeline.py` | 患者折泄漏断言、预处理器 schema 一致性、pAUC 边界条件 |

---

## 9. 已修复的代码审查问题

| # | 严重度 | 问题 | 修复 |
|---|---|---|---|
| 1 | P1 | M4 shuffle 导致 image/metadata 错位 | 新建 `MultimodalDataset` 同时返回 image+meta+label，DataLoader shuffle 时三者同步 |
| 2 | P1 | M2 text prompt 拿不到原始临床字段 | `ModelBatch` 新增 `raw_metadata`，`_extract_text_features()` 优先使用 `raw_metadata` |
| 3 | P1 | M2/M4 load 后 CUDA 设备不一致 | evaluate.py 添加 `_move_model_to_device()`；load() 默认保持 CPU |
| 4 | P1 | M2 raw_metadata DataFrame 布尔崩溃 | `or batch.metadata` 改为显式 `if raw is None` 判断，避免 DataFrame 的 ambiguous truth value |
| 5 | P1 | M2 encoder 未移到 CUDA | fit() 中创建 encoder 后加 `self.encoder.to(self.device)` |
| 6 | P1 | evaluate.py 缺少 nn 导入 | 延迟导入 torch/nn 到函数内部，M0 评估无需 torch |
| 7 | P1 | M2 checkpoint 加载特征空间不一致 | 保存 `encoder_state_dict`，load 时优先加载保存的权重；若 checkpoint 有 encoder 权重则 `pretrained=False` 跳过下载，否则 fallback |
| 8 | P2 | M2 Python `hash()` 不可复现 | 改用 `hashlib.md5` 的 `_stable_hash()`，不受 PYTHONHASHSEED 影响 |
| 9 | P2 | factory 顶层强依赖 torch | 改为 lazy import + `try/except ImportError` |
| 10 | P2 | pyproject 缺 torchvision/lightgbm | 已添加到 `[project.optional-dependencies] ml` |
| 11 | P2 | `np.trapz` numpy 2.0 兼容 | 改为 `np.trapezoid`（fallback `np.trapz`） |
| 12 | P2 | M2 load() 在 CPU-only 环境尝试 .to("cuda") | load() 默认保持 CPU，设备迁移交给外层 `_move_model_to_device()` |
| 13 | P2 | 训练日志被覆盖 | 摘要写入 `train_summary.csv`，epoch 日志保留在 `train_log.csv`，不再覆盖 |
| 14 | P2 | metrics.json artifacts 路径错误 | 始终包含 `train_summary`，深度模型额外包含 `train_log` 和 `loss_curve` |
| 15 | P2 | M2 load 不必要下载 pretrained 权重 | checkpoint 有 `encoder_state_dict` 时 `pretrained=False`，无则 fallback 到 `pretrained=True` |
| 16 | P3 | evaluate.py 顶层依赖 torch | torch 和图像 adapter 延迟导入到 `evaluate_image_model()`/`evaluate_graph()` 内，M0 评估无需 torch 环境 |