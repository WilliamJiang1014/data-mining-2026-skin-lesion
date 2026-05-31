# Scripts 使用指南与代码走读

本目录保存项目的所有可执行脚本。它们不是孤立运行的工具，而是按“数据准备 -> 图构建 -> 模型训练 -> 模型评估 -> 报告汇总”的流水线组织起来的。

阅读建议：

1. 先看 `prepare_data.py`，理解原始数据如何变成统一 manifest。
2. 再看 `run_train.py`，理解 M0-M4 怎么使用 manifest、folds 和预处理器训练。
3. 如果跑 M5，再看 `build_graph.py`，它负责把样本转成图结构。
4. 最后看 `evaluate.py` 和 `make_report_assets.py`，它们负责复现评估和整理结果表。

---

## 一、快速启动流程

以下命令默认在项目根目录执行：

```bash
cd /path/to/data-mining-2026-skin-lesion
conda activate skin
```

### 1. 准备数据

```bash
python scripts/prepare_data.py
```

该步骤会读取：

- `data/raw/isic2024_permissive/images/`
- `data/raw/isic2024_permissive/metadata.csv`
- `data/raw/isic2024_permissive/labels.csv`
- `data/raw/isic2024_permissive/supplemental_metadata.csv`，如果存在
- `data/raw/pad_ufes_20/images/`
- `data/raw/pad_ufes_20/metadata.csv`，如果存在

然后生成：

- `data/processed/manifest_isic.csv`
- `data/processed/manifest_pad.csv`
- `data/processed/folds_isic.csv`
- `data/processed/preprocessors/fold*_tabular.pkl`
- `reports/tables/split_stats.csv`
- `reports/data_quality_*.md`

### 2. 构建图数据，只有 M5 必须执行

```bash
# M5 需要构建 5 折图数据
for f in 0 1 2 3 4; do
  python scripts/build_graph.py --fold $f --knn-device cuda
done
```

M0-M4 不依赖图文件，可以跳过这一步。

### 3. 训练模型

使用 `--all-folds` 一次性跑 5 折交叉验证：

```bash
# M0（CPU，秒级/分钟级）
python scripts/run_train.py --model m0_constant --all-folds
python scripts/run_train.py --model m0_lightgbm --all-folds

# M1–M4（需 GPU）
python scripts/run_train.py --model m1_cnn_baseline --all-folds --device cuda
python scripts/run_train.py --model m2_monet_feature_baseline --all-folds --device cuda
python scripts/run_train.py --model m3_transformer_baseline --all-folds --device cuda
python scripts/run_train.py --model m4_multimodal_fusion --all-folds --device cuda

# M5（需先构建图）
python scripts/run_train.py --model m5_lgke_gnn --all-folds --device cuda
```

也可以单独跑某个 fold：

```bash
python scripts/run_train.py --model m1_cnn_baseline --fold 0 --device cuda
```

如果没有 GPU，把 `--device cuda` 改成 `--device cpu`。

### 4. 评估模型

```bash
for f in 0 1 2 3 4; do
  python scripts/evaluate.py --model m0_lightgbm --fold $f
done
```

### 5. 汇总报告表格

```bash
python scripts/make_report_assets.py
```

---

## 二、脚本依赖关系

```text
prepare_data.py
  |
  |-- manifest_isic.csv
  |-- manifest_pad.csv
  |-- folds_isic.csv
  |-- preprocessors/fold*_tabular.pkl
  |
  |--> run_train.py 训练 M0-M4
  |
  |--> build_graph.py 构建图
          |
          |-- graph/fold*_train_graph.pt
          |-- graph/fold*_val_graph.pt
          |-- graph/fold*_test_graph.pt
          |
          |--> run_train.py 训练 M5

run_train.py
  |
  |-- data/artifacts/trained_models/{model}/fold{f}/
  |   |-- best.ckpt 或 model.pkl
  |   |-- config_resolved.yaml
  |   |-- val_predictions.csv
  |   |-- test_predictions.csv
  |   |-- metrics.json
  |
  |--> evaluate.py
          |
          |--> make_report_assets.py
```

---

## 三、prepare_data.py 数据准备脚本

### 功能定位

`prepare_data.py` 是全项目的入口脚本。它把不同来源的数据集整理成统一字段格式，并生成训练、验证、测试划分和表格特征预处理器。

### 主要输入

- ISIC 2024 permissive 原始图片和 CSV。
- PAD-UFES-20 原始图片和 CSV，可选。
- 命令行参数，例如 `--folds`、`--val-ratio`、`--image-check`。

### 主要输出

```text
data/processed/
  manifest_isic.csv
  manifest_pad.csv
  folds_isic.csv
  preprocessors/
    fold0_tabular.pkl
    fold1_tabular.pkl
    fold2_tabular.pkl
    fold3_tabular.pkl
    fold4_tabular.pkl

reports/
  tables/split_stats.csv
  data_quality_isic.md
  data_quality_pad.md
```

### 代码执行顺序

1. `main()` 解析命令行参数。
2. `build_isic_manifest()` 读取 ISIC metadata、labels、图片路径，并合并成统一 manifest。
3. 如果没有传 `--skip-pad`，`build_pad_manifest()` 会读取 PAD 数据并生成 PAD manifest。
4. `stratified_sample()` 在 `--max-samples` 被指定时抽小样本，用于 CPU 快速测试。
5. 根据患者 ID 做患者级 fold 划分，避免同一患者同时出现在训练集和测试集。
6. `build_preprocessors()` 为每个 fold 单独拟合表格预处理器，避免验证/测试信息泄露。
7. `write_split_stats()` 写入训练、验证、测试的样本数、阳性数、阳性率。
8. `write_quality_reports()` 写入数据质量报告。
9. 最后在终端打印 JSON 摘要，方便快速确认样本数和图片检查结果。

### 关键函数注释

#### `build_isic_manifest(root: Path) -> pd.DataFrame`

作用：把 ISIC 官网下载的三个文件合并成项目统一格式。

它主要做四件事：

- 读取主元数据 `metadata.csv`，保留年龄、性别、部位、病灶尺寸等字段。
- 读取标签文件 `labels.csv`，把恶性标签统一成 `target`。
- 拼接图片路径，让每一行样本都能找到对应 JPEG。
- 补齐项目统一需要的列，例如 `sample_id`、`patient_id`、`dataset`、`image_path`。

注意：如果 `supplemental_metadata.csv` 存在，脚本会尽量合并补充字段；这些字段主要用于增强表格特征和后续图构建。

#### `build_pad_manifest(root: Path) -> pd.DataFrame`

作用：把 PAD-UFES-20 数据整理成和 ISIC 类似的 manifest。

PAD 数据用于外部验证或域适应，不是 M0-M5 内部交叉验证的主训练集。它的标签分布和 ISIC 不一样，所以不要把 PAD 当成 ISIC fold 的一部分，除非明确跑域适应脚本。

#### `normalize_category(series)`

作用：清洗类别字段。

典型处理包括：

- 缺失值统一为 `unknown`。
- 字符串统一小写。
- 去掉多余空格。

这样可以避免同一个类别因为大小写或空格不同被当成多个类别。

#### `stratified_sample(df, n, seed)`

作用：生成小样本 smoke test。

因为 ISIC 样本非常多，而且阳性极少，简单随机抽样可能抽不到阳性。这个函数会尽量保留正负样本比例，让小样本仍然可以跑通指标计算。

#### `build_preprocessors(isic, folds, out_dir, rare_min_count)`

作用：为每个 fold 训练表格预处理器。

重要点：

- 只用当前 fold 的训练集拟合预处理器。
- 验证集和测试集只调用 transform。
- 这样可以避免把验证/测试集的类别分布提前泄露给模型。

输出的 `fold*_tabular.pkl` 会在 `run_train.py`、`build_graph.py`、`evaluate.py` 中复用。

#### `write_split_stats(...)`

作用：生成 `reports/tables/split_stats.csv`。

这个表主要用于报告中说明：

- 每个 split 有多少样本。
- 有多少恶性样本。
- 阳性率是多少。
- 患者数量是多少。

#### `write_quality_reports(...)`

作用：生成 Markdown 格式的数据质量报告。

报告内容通常包括：

- 缺失字段统计。
- 图片检查失败数量。
- 标签分布。
- 数据集基本统计。

### 常用参数

```bash
python scripts/prepare_data.py --skip-pad
python scripts/prepare_data.py --image-check none
python scripts/prepare_data.py --image-check all
python scripts/prepare_data.py --max-samples 1000 --image-check none
```

参数含义：

- `--skip-pad`：PAD 没下载时跳过 PAD。
- `--image-check none`：不检查图片，速度最快。
- `--image-check sample`：抽样检查图片，默认设置。
- `--image-check all`：检查所有图片，最慢但最完整。
- `--max-samples`：只抽一部分 ISIC 样本，用于快速测试流程。

---

## 四、build_graph.py 图构建脚本

### 功能定位

`build_graph.py` 专门服务于 M5 LGKE-GNN。它把 lesion 样本、患者节点、属性节点、视觉原型节点、元数据原型节点组织成一个 PyTorch 图对象。

M0-M4 不需要运行这个脚本。

### 主要输入

- `data/processed/manifest_isic.csv`
- `data/processed/folds_isic.csv`
- `data/processed/preprocessors/fold{f}_tabular.pkl`
- ISIC 图片文件

### 主要输出

```text
data/processed/graph/
  fold0_train_graph.pt
  fold0_val_graph.pt
  fold0_test_graph.pt
  fold0_train_nodes.parquet
  fold0_train_edges.parquet
  ...

data/processed/embeddings/
  image_stats_features.npz
```

### 代码执行顺序

1. `main()` 解析图构建参数。
2. 读取 manifest、folds 和当前 fold 的表格预处理器。
3. 按 train、val、test 三个 split 分别调用 `build_graph()`。
4. `load_or_build_image_feature_cache()` 为图片提取轻量视觉统计特征，并缓存到 `.npz`。
5. `build_graph()` 创建 lesion 节点特征。
6. `make_patient_nodes()` 创建患者节点。
7. `make_attribute_nodes()` 创建年龄、性别、部位等属性节点。
8. `patient_edges()` 和 `attribute_edges()` 构建语义边。
9. `knn_edges()` 构建视觉 kNN 和元数据 kNN 边。
10. `make_kmeans_prototypes()` 和 `prototype_edges()` 构建原型节点及连接边。
11. 保存 `.pt` 图对象，同时写出 nodes/edges parquet 方便检查。

### 图中节点类型

```text
lesion node      每个皮肤病变样本一个节点
patient node     同一个患者共享的节点
attribute node   年龄、性别、解剖部位等离散属性节点
visual prototype 视觉聚类原型节点
metadata prototype 元数据聚类原型节点
```

### 图中边类型

```text
same_patient        同一患者内的 lesion 连接
has_attribute       lesion 连接到属性节点
visual_knn          视觉特征相近的 lesion 连接
metadata_knn        表格元数据相近的 lesion 连接
visual_prototype    lesion 连接到视觉原型
metadata_prototype  lesion 连接到元数据原型
```

### 关键函数注释

#### `build_graph(...)`

作用：构建单个 split 的完整图。

它会把图所需的所有内容打包进一个 dict，包括：

- `x`：节点特征矩阵。
- `edge_index`：边的起点和终点。
- `edge_type`：每条边的关系类型。
- `edge_weight`：边权重。
- `y`：lesion 节点标签。
- `sample_ids`：样本 ID。
- `node_table`、`edge_table`：用于导出检查的表格。

#### `image_feature(row, cache, image_size)`

作用：从图片提取轻量视觉统计特征。

这里不是 CNN 深度特征，而是为了图构建快速可复现的统计特征，例如颜色和亮度相关特征。它会优先从 cache 中读取，避免重复打开大量图片。

#### `load_or_build_image_feature_cache(...)`

作用：维护图片特征缓存。

如果 `image_stats_features.npz` 已存在且没有传 `--refresh-image-cache`，脚本会复用缓存；否则重新遍历图片生成特征。

#### `make_patient_nodes(df, lesion_x)`

作用：为每个患者创建一个节点。

患者节点特征通常由该患者所有 lesion 特征聚合得到。这样图模型可以利用“同一患者多个病灶之间存在关联”的信息。

#### `make_attribute_nodes(df, lesion_x)`

作用：为重要类别属性创建节点。

例如性别、部位、年龄段等字段会被编码成属性节点，lesion 通过 `has_attribute` 边连接到这些节点。

#### `row_attributes(row)`

作用：从单个样本行中抽取可用于属性节点的字段。

如果某个字段缺失，会尽量使用 `unknown` 或跳过，避免图构建中断。

#### `knn_edges(...)`

作用：根据特征相似度构建 kNN 边。

代码内部会根据 `--knn-device` 决定使用：

- `torch_knn_edges()`：适合 GPU 或大规模张量计算。
- `sklearn_knn_edges()`：适合 CPU 环境。

#### `prototype_edges(...)`

作用：连接 lesion 节点与聚类原型节点。

原型节点可以理解成“典型视觉模式”或“典型元数据模式”，用于给 GNN 提供全局结构信息。

#### `write_nodes_edges_tables(...)`

作用：把图节点和边导出成 parquet。

这些表不是训练必须的，但非常适合排查图是否构建正确，例如检查每种边有多少条、节点数量是否异常。

### 常用参数

```bash
python scripts/build_graph.py --fold 0
python scripts/build_graph.py --fold 0 --knn-device cpu
python scripts/build_graph.py --fold 0 --knn-device cuda
python scripts/build_graph.py --fold 0 --refresh-image-cache
python scripts/build_graph.py --fold 0 --max-samples 500
```

---

## 五、run_train.py 统一训练脚本

### 功能定位

`run_train.py` 是 M0-M5 的统一训练入口。它负责读取配置、划分数据、构造模型、调用模型的 `fit()`、保存预测和指标。

### 支持模型

| 模型名 | 类型 | 主要输入 | 说明 |
| --- | --- | --- | --- |
| `m0_constant` | `constant` | 标签先验 | 只输出训练集阳性率作为预测 |
| `m0_lightgbm` | `tabular_lgbm` | 表格特征 | 表格基线模型 |
| `m1_cnn_baseline` | `image` | 图片 | CNN 图像模型 |
| `m2_monet_feature_baseline` | `monet_feature` | 图片 + 文本提示 | 冻结图像 encoder 后接分类头 |
| `m3_transformer_baseline` | `image_transformer` | 图片 | Transformer 图像模型 |
| `m4_multimodal_fusion` | `multimodal` | 图片 + 表格 | 多模态融合模型 |
| `m5_lgke_gnn` | `graph_multimodal` | 图数据 | LGKE-GNN |

### 主要输入

- `configs/experiments/baselines.yaml`
- `configs/models/*.yaml`
- `data/processed/manifest_isic.csv`
- `data/processed/folds_isic.csv`
- `data/processed/preprocessors/fold{f}_tabular.pkl`
- M5 额外需要 `data/processed/graph/fold{f}_{split}_graph.pt`

### 主要输出

```text
data/artifacts/trained_models/{model_name}/fold{f}/
  best.ckpt 或 model.pkl
  config_resolved.yaml
  train_log.csv
  loss_curve.csv
  val_predictions.csv
  test_predictions.csv
  metrics.json
  thresholds.json
```

### 代码执行顺序

1. `main()` 解析命令行参数。
2. `load_yaml()` 读取实验配置。
3. 如果使用 `--list-models`，直接列出配置中可运行的模型。
4. 根据 `--model` 找到对应模型条目和模型配置文件。
5. 命令行参数覆盖 YAML 中的超参数，例如 `--epochs`、`--learning-rate`。
6. 根据模型类型分流：
   - M0-M4 调用 `train_non_graph_model()`。
   - M5 调用图模型训练逻辑，读取 `.pt` 图文件。
7. 训练结束后写出验证集和测试集预测。
8. `validation_thresholds()` 根据验证集选择阈值。
9. `prediction_metrics()` 计算测试集指标。
10. `write_metrics()` 写出 `metrics.json`。
11. `update_main_results()` 更新 `reports/tables/main_results.csv`。

### 关键函数注释

#### `load_yaml(path: Path) -> dict`

作用：读取 YAML 配置文件。

所有模型超参数都优先来自配置文件，然后由命令行参数覆盖。这样可以保证实验可复现，同时又方便临时调参。

#### `load_graph(path: Path) -> dict`

作用：读取 M5 图文件。

图文件由 `build_graph.py` 生成，是一个 PyTorch 保存的 dict。训练 M5 时会分别读取 train、val、test 图。

#### `prediction_metrics(result, threshold, groups=None)`

作用：把模型预测结果转成评估指标。

常见指标包括：

- AUROC
- average precision
- sensitivity
- specificity
- balanced accuracy
- confusion matrix
- subgroup metrics，如果传入分组信息

#### `write_predictions(path, result)`

作用：把预测结果写成 CSV。

通常包含：

- `sample_id`
- `y_true`
- `y_score`
- `y_pred`

这些文件是后续复查单样本预测、画图、汇总结果的基础。

#### `train_non_graph_model(args, experiment_cfg, model_entry)`

作用：训练 M0-M4。

它的核心流程是：

1. 读取 manifest 和 folds。
2. 找出当前 fold 的 train、val、test 样本。
3. 读取当前 fold 对应的 tabular preprocessor。
4. 调用 `build_manifest_batch()` 把 DataFrame 转成统一的 `ModelBatch`。
5. 通过 model factory 创建模型实例。
6. 调用模型自己的 `fit(train, val)`。
7. 分别对 val、test 调用 `predict_proba()`。
8. 保存模型、配置、预测和指标。

#### `build_manifest_batch(df, preprocessor, include_metadata)`

作用：把 manifest 中的一段样本转成模型统一输入。

根据模型类型不同，batch 里会包含：

- `sample_ids`：样本 ID。
- `image_paths`：图片路径，图像模型需要。
- `metadata`：表格特征，表格模型和多模态模型需要。
- `labels`：二分类标签。
- `groups`：患者 ID，用于患者级指标或分组分析。

这个函数是 M0-M4 共享的数据入口。如果模型训练报字段缺失，优先检查这里。

#### `validation_thresholds(result)`

作用：根据验证集预测选择多个候选阈值。

常见阈值规则包括：

- 默认 `0.5`。
- 满足 sensitivity 至少 0.90 的阈值。
- Youden index 最优阈值。

测试集评估时会把这些阈值写入 `thresholds.json`，方便报告中说明阈值来源。

#### `write_train_log(...)`

作用：保存训练日志。

对 M0 这类非深度模型，日志可能只有一行摘要。对深度模型，模型适配器内部可能还会写 epoch 级别的 loss 曲线。

#### `update_main_results(path, model_name, fold, metrics)`

作用：把当前模型当前 fold 的测试指标追加或更新到主结果表。

如果同一个模型同一个 fold 重复训练，脚本会覆盖对应行，而不是无限追加重复记录。

### 常用参数

```bash
python scripts/run_train.py --list-models
python scripts/run_train.py --model m0_constant --fold 0
python scripts/run_train.py --model m0_lightgbm --all-folds
python scripts/run_train.py --model m1_cnn_baseline --all-folds --device cuda
python scripts/run_train.py --model m5_lgke_gnn --all-folds --device cuda --hidden-dim 128
```

参数说明：

- `--config`：实验总配置，默认 `configs/experiments/baselines.yaml`。
- `--model`：模型名称，必须与配置文件中的模型名一致。
- `--fold`：运行第几个 fold。
- `--all-folds`：一次性跑所有 fold（0 到 `--num-folds - 1`），与 `--fold` 二选一。
- `--num-folds`：fold 总数，默认 5。
- `--device`：`cpu` 或 `cuda`。
- `--epochs`：覆盖模型配置中的训练轮数。
- `--patience`：覆盖早停耐心值。
- `--learning-rate`：覆盖学习率。
- `--dropout`：覆盖 dropout。
- `--edge-chunk-size`：M5 处理大图边时使用，显存不足时可调小。

---

## 六、evaluate.py 模型评估脚本

### 功能定位

`evaluate.py` 用于加载已经训练好的模型，并在当前 fold 的测试集上重新评估。它适合在训练完成后复算指标，或修改指标计算逻辑后重新生成结果。

### 主要输入

- 模型训练产物目录。
- `manifest_isic.csv`
- `folds_isic.csv`
- `preprocessors/fold*_tabular.pkl`
- M5 需要对应 test graph。

### 主要输出

```text
data/artifacts/trained_models/{model_name}/fold{f}/
  test_predictions.csv
  metrics.json

reports/tables/main_results.csv
```

### 代码执行顺序

1. `main()` 解析 `--model`、`--fold`、`--device` 等参数。
2. 读取实验配置，确认模型类型。
3. 根据模型类型分流：
   - 表格/常数模型调用 `evaluate_manifest_model()`。
   - 图像/多模态模型调用 `evaluate_image_model()`。
   - 图模型调用 `evaluate_graph()`。
4. 加载训练时保存的模型文件。
5. 加载测试集数据。
6. 调用模型的 `predict_proba()` 得到预测概率。
7. 读取 `thresholds.json`，优先使用训练阶段选出的阈值。
8. 计算指标并更新 `metrics.json`。
9. 更新 `reports/tables/main_results.csv`。

### 关键函数注释

#### `_move_model_to_device(model, device)`

作用：把模型内部的 PyTorch 模块移动到指定设备。

不同模型适配器保存模块的属性名不一样，所以这个函数会尽量检查常见属性，例如 `model`、`encoder`、`head` 等。

#### `evaluate_manifest_model(...)`

作用：评估不依赖图片 tensor 的模型。

主要用于：

- `m0_constant`
- `m0_lightgbm`

它会使用表格预处理器把测试集元数据转成模型输入。

#### `evaluate_image_model(...)`

作用：评估图片模型或多模态模型。

主要用于：

- `m1_cnn_baseline`
- `m2_monet_feature_baseline`
- `m3_transformer_baseline`
- `m4_multimodal_fusion`

它会重新构造 `ModelBatch`，其中包含图片路径、表格特征和标签。

#### `evaluate_graph(...)`

作用：评估 M5 LGKE-GNN。

它直接读取 test graph，而不是从 manifest 重新构造图。因此评估前必须已经运行过 `build_graph.py`。

#### `load_thresholds(path)`

作用：读取训练阶段保存的阈值。

如果文件不存在，则评估脚本会退回默认阈值。为了报告一致，建议优先使用训练时保存的阈值。

#### `load_existing_json(path)`

作用：读取已有 `metrics.json`。

评估脚本会在原有指标基础上更新测试指标，避免把训练阶段保存的其他信息全部覆盖。

### 常用参数

```bash
python scripts/evaluate.py --model m0_lightgbm --fold 0
python scripts/evaluate.py --model m1_cnn_baseline --fold 0 --device cuda
python scripts/evaluate.py --model m5_lgke_gnn --fold 0 --device cuda
```

---

## 七、make_report_assets.py 报告表格脚本

### 功能定位

`make_report_assets.py` 会扫描训练产物目录，把所有模型所有 fold 的 `metrics.json` 汇总成主结果表。

它不重新训练，也不重新预测，只读取已有结果。

### 主要输入

```text
data/artifacts/trained_models/{model_name}/fold{f}/metrics.json
```

### 主要输出

```text
reports/tables/main_results.csv
```

### 代码执行顺序

1. `main()` 解析 `--artifacts-dir`、`--out`、`--threshold-rule`。
2. `collect_main_results()` 遍历模型目录和 fold 目录。
3. 对每个 `metrics.json` 提取测试指标。
4. 将不同模型、不同 fold 的结果合并成 DataFrame。
5. 按模型名和 fold 排序。
6. 写出主结果表。

### 关键函数注释

#### `collect_main_results(artifacts_dir, threshold_rule)`

作用：从训练产物目录提取所有可汇总的指标。

如果某个 fold 没有 `metrics.json`，脚本会跳过它。这样可以先跑部分模型，再逐步补齐结果。

### 常用参数

```bash
python scripts/make_report_assets.py
python scripts/make_report_assets.py --threshold-rule sensitivity_at_least_0_90
python scripts/make_report_assets.py --out reports/tables/main_results.csv
```

---

## 八、run_pad_external_eval.py PAD 外部验证脚本

### 功能定位

`run_pad_external_eval.py` 用 PAD-UFES-20 作为外部测试集，评估在 ISIC 上训练好的 M5 模型。它回答的问题是：模型从 ISIC 学到的模式，能不能泛化到另一个数据集。

### 主要输入

- `data/processed/manifest_isic.csv`
- `data/processed/manifest_pad.csv`
- `data/processed/folds_isic.csv`
- `data/processed/preprocessors/fold*_tabular.pkl`
- 已训练好的 M5 checkpoint。

### 主要输出

输出目录默认和 checkpoint 根目录相关，通常包含：

```text
pad_external_eval/
  fold*_pad_predictions.csv
  fold*_metrics.json
  fold*_graph_report.md
  summary.csv
  subgroup_metrics.csv
```

### 代码执行顺序

1. `main()` 解析 PAD 外部验证参数。
2. 读取 ISIC 和 PAD manifest。
3. 对每个要评估的 fold：
   - 加载对应 fold 的 ISIC 预处理器。
   - 构建或复用 PAD 外部验证图。
   - 加载 M5 checkpoint。
   - 对 PAD 图做预测。
   - 计算整体指标和分组指标。
4. 如果运行多个 fold，额外计算 ensemble 指标。
5. 写出 summary、subgroup metrics 和 JSON 结果。

### 关键函数注释

#### `build_external_graph(...)`

作用：为 PAD 外部验证构建图。

它会把 ISIC 训练样本和 PAD 测试样本放到同一个图构建逻辑中，使 PAD 样本可以连接到 ISIC 中的视觉/元数据结构。

#### `load_external_image_feature_cache(...)`

作用：维护包含 PAD 图片的视觉统计特征缓存。

如果已有 ISIC 的 `image_stats_features.npz`，脚本会尽量复用它，只为 PAD 图片补充特征。

#### `external_metrics(result, threshold, pad_manifest)`

作用：计算 PAD 外部测试指标。

除了整体 AUROC、AP、sensitivity、specificity，还会结合 PAD manifest 计算分组指标。

#### `ensemble_metrics(predictions, pad_manifest, threshold)`

作用：多个 fold 的预测结果取平均，形成 ensemble 结果。

这种做法可以降低单个 fold checkpoint 的偶然性。

#### `write_graph_report(path, graph, fold)`

作用：记录外部验证图的节点数、边数和边类型分布。

如果 PAD 外部结果异常，先看这个报告确认 PAD 样本是否真的连进图里。

### 常用参数

```bash
python scripts/run_pad_external_eval.py --folds-to-run 0 --device cuda
python scripts/run_pad_external_eval.py --folds-to-run 0,1,2,3,4 --device cuda
python scripts/run_pad_external_eval.py --folds-to-run 0 --reuse-graphs
```

---

## 九、run_pad_domain_adaptation.py PAD 域适应脚本

### 功能定位

`run_pad_domain_adaptation.py` 用于探索 ISIC 到 PAD 的跨域泛化。它可以把 PAD 样本加入训练，或只用 PAD 训练，比较不同域适应策略。

### 主要输入

- ISIC manifest 和 folds。
- PAD manifest。
- M5 模型配置。
- 可选的初始 checkpoint。

### 主要输出

```text
data/artifacts/trained_models/pad_domain_adaptation/{variant_name}/
  fold*_metrics.json
  fold*_predictions.csv
  summary.csv
```

### 代码执行顺序

1. `main()` 解析域适应参数。
2. `load_or_create_pad_folds()` 为 PAD 创建或读取 fold 划分。
3. `fit_or_load_preprocessor()` 根据参数选择：
   - 复用 ISIC 预处理器。
   - 或在混合数据上重新拟合预处理器。
4. `load_or_build_graphs()` 构建或复用 ISIC+PAD 图。
5. `model_params()` 汇总 M5 超参数。
6. 对每个 fold 训练域适应模型。
7. 在 held-out PAD split 上评估。
8. `summarize()` 汇总所有 fold 的结果。
9. `write_summary_csv()` 写出最终对比表。

### 关键函数注释

#### `load_or_create_pad_folds(pad, path, seed)`

作用：为 PAD 数据创建固定 fold。

如果 `folds_pad_adapt.csv` 已存在，直接读取；否则根据 PAD 样本生成并保存，保证后续实验划分一致。

#### `load_or_build_graphs(...)`

作用：准备域适应训练需要的图。

如果传入 `--reuse-graphs`，脚本会优先使用已有图文件，避免重复构建。否则会重新根据当前设置生成图。

#### `fit_or_load_preprocessor(...)`

作用：决定表格预处理器的来源。

两种模式：

- `isic`：直接复用 ISIC fold 预处理器，适合测试跨域泛化。
- `refit`：在训练数据上重新拟合，适合域适应训练。

#### `model_params(args, fold, fold_out)`

作用：把命令行中的超参数整理成 M5 模型初始化参数。

例如 hidden dim、层数、dropout、学习率、权重衰减、边 chunk size 等。

#### `flatten_fold_metrics(...)`

作用：把嵌套指标展平成一行 CSV。

这样后续可以直接用 Excel、pandas 或论文表格读取。

### 常用参数

```bash
python scripts/run_pad_domain_adaptation.py --variant-name mixed --train-mode mixed --device cuda
python scripts/run_pad_domain_adaptation.py --variant-name pad_only --train-mode pad_only --device cuda
python scripts/run_pad_domain_adaptation.py --variant-name mixed_refit --preprocessor-mode refit --device cuda
python scripts/run_pad_domain_adaptation.py --variant-name mixed --reuse-graphs --device cuda
```

---

## 十、常见任务怎么跑

### 只跑 M0，最快检查整体流程

```bash
python scripts/prepare_data.py --image-check sample
python scripts/run_train.py --model m0_constant --all-folds
python scripts/run_train.py --model m0_lightgbm --all-folds
python scripts/make_report_assets.py
```

### 跑一个图像模型（5 折）

```bash
python scripts/run_train.py --model m1_cnn_baseline --all-folds --device cuda
for f in 0 1 2 3 4; do
  python scripts/evaluate.py --model m1_cnn_baseline --fold $f --device cuda
done
```

### 跑 M5 LGKE-GNN（5 折）

```bash
for f in 0 1 2 3 4; do
  python scripts/build_graph.py --fold $f --knn-device cuda
done
python scripts/run_train.py --model m5_lgke_gnn --all-folds --device cuda
python scripts/make_report_assets.py
```

### 后台训练 + 日志

训练时每个 epoch 会打印进度行，格式如下：

```
[m1_cnn_baseline] epoch 3/20  loss=0.1234  pAUC=0.5678  AUROC=0.9012  lr=9.50e-05  best=0.6000  patience_left=5
```

脚本内置日志记录，运行时自动写入 `log/{model}_fold{N}.log`，同时打印到终端。

#### Linux 服务器后台运行（nohup）

在 Linux 服务器上使用 `nohup` 后台运行，配合 `CUDA_VISIBLE_DEVICES` 指定 GPU：

```bash
# 创建日志目录
mkdir -p log

# 后台跑 M1 5 折（使用 GPU 4）
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run_train.py --model m1_cnn_baseline --all-folds --device cuda > log/m1_cnn_baseline.log 2>&1 &

# 后台跑 M2 5 折（使用 GPU 5）
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run_train.py --model m2_monet_feature_baseline --all-folds --device cuda > log/m2_monet.log 2>&1 &

# 后台跑 M3 5 折（使用 GPU 6）
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run_train.py --model m3_transformer_baseline --all-folds --device cuda > log/m3_transformer.log 2>&1 &

# 后台跑 M4 5 折（使用 GPU 7）
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run_train.py --model m4_multimodal_fusion --all-folds --device cuda > log/m4_multimodal.log 2>&1 &
```

查看实时训练日志：

```bash
tail -f log/m1_cnn_baseline.log
```

查看后台任务状态：

```bash
jobs -l
# 或查看进程
ps aux | grep run_train
```

#### 并行训练多个模型

如果服务器有多张 GPU（如 RTX 5090 x 4），可以同时启动 4 个训练任务：

```bash
# 并行启动 M1-M4，每个模型占用一张 GPU
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run_train.py --model m1_cnn_baseline --all-folds --device cuda > log/m1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run_train.py --model m2_monet_feature_baseline --all-folds --device cuda > log/m2.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run_train.py --model m3_transformer_baseline --all-folds --device cuda > log/m3.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run_train.py --model m4_multimodal_fusion --all-folds --device cuda > log/m4.log 2>&1 &

# 等待所有后台任务完成
wait

# 然后跑 M5（需要先构建图）
for f in 0 1 2 3 4; do
  python scripts/build_graph.py --fold $f --knn-device cuda
done
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run_train.py --model m5_lgke_gnn --all-folds --device cuda > log/m5.log 2>&1 &
```

#### Windows 后台运行

Windows 上可以用 `Start-Process`（PowerShell）或 `start`（CMD）：

```powershell
# PowerShell
Start-Process python -ArgumentList "scripts/run_train.py --model m1_cnn_baseline --all-folds --device cuda" -RedirectStandardOutput "log/m1.log" -RedirectStandardError "log/m1_err.log" -NoNewWindow
```

### 调整训练超参数

修改 `configs/training.yaml` 即可调整各模型的 batch_size、epochs、patience 等，无需改模型代码：

```yaml
m1_cnn_baseline:
  batch_size: 64
  epochs: 20
  patience: 5
  learning_rate: 1.0e-4
```

命令行参数优先级最高，可临时覆盖：

```bash
python scripts/run_train.py --model m1_cnn_baseline --all-folds --device cuda --batch-size 128 --epochs 30
```

### CPU smoke test

```bash
python scripts/prepare_data.py --max-samples 1000 --image-check none
python scripts/build_graph.py --fold 0 --max-samples 500 --knn-device cpu
python scripts/run_train.py --model m0_constant --fold 0
```

---

## 十一、输出文件怎么看

### `manifest_isic.csv`

每一行代表一个 ISIC lesion 样本。核心字段通常包括：

- `sample_id`：样本 ID。
- `patient_id`：患者 ID。
- `image_path`：图片路径。
- `target`：二分类标签，1 表示恶性。
- 年龄、性别、部位、尺寸等元数据字段。

### `folds_isic.csv`

记录每个样本属于哪个 fold，以及在该 fold 中是 train、val 还是 test。

这个文件很重要，因为后续所有训练、评估都依赖它保证划分一致。

### `fold*_tabular.pkl`

每个 fold 独立保存的表格预处理器。

不要混用不同 fold 的预处理器，否则会造成特征维度或类别编码不一致。

### `best.ckpt`

深度学习模型的 checkpoint，通常包含：

- 模型权重。
- 模型配置。
- 特征维度。
- 训练时使用的重要参数。

### `model.pkl`

传统机器学习模型的序列化文件，例如常数基线或表格模型。

### `val_predictions.csv`

验证集预测。主要用途：

- 选择阈值。
- 检查模型是否过拟合。
- 做错误样本分析。

### `test_predictions.csv`

测试集预测。主要用途：

- 计算最终报告指标。
- 保存每个样本的预测概率。
- 后续画 ROC、PR 曲线或病例分析。

### `metrics.json`

模型指标汇总。通常包含：

- 验证集指标。
- 测试集指标。
- 阈值。
- confusion matrix。
- subgroup metrics。

### `main_results.csv`

论文或课程报告中的主结果表来源。

如果发现这个表没有更新，优先检查对应模型 fold 下是否已经生成 `metrics.json`。

---

## 十二、排错提示

### 1. 找不到图片

优先检查：

```text
data/raw/isic2024_permissive/images/
data/raw/pad_ufes_20/images/
```

然后重新运行：

```bash
python scripts/prepare_data.py --image-check sample
```

### 2. PAD 数据没下载

可以先跳过 PAD：

```bash
python scripts/prepare_data.py --skip-pad
```

这样仍然可以训练和评估 ISIC 内部验证模型。

### 3. M5 报找不到 graph 文件

说明还没有构建图：

```bash
python scripts/build_graph.py --fold 0
```

如果要跑 5 折，需要对 0-4 都构建。

### 4. CUDA 显存不足

优先尝试：

- 降低 batch size。
- 降低 image size。
- 对 M5 调小 `--edge-chunk-size`。
- 对图构建使用 `--knn-device cpu`。

### 5. 指标表为空

检查是否存在：

```text
data/artifacts/trained_models/{model_name}/fold{f}/metrics.json
```

如果没有，需要先运行 `run_train.py` 或 `evaluate.py`。

### 6. 训练结果重复或被覆盖

同一个 `{model_name}/fold{f}` 目录会被同名实验复用。重复训练同一个模型同一个 fold 时，旧的预测和指标可能被新结果覆盖。

如果想保留多个实验版本，应修改配置里的模型名，或把输出目录改到新的路径。

---

## 十三、推荐阅读顺序

如果你是为了理解代码，建议按这个顺序打开文件：

1. `scripts/prepare_data.py`
2. `src/skin_lesion_risk/data/schema.py`
3. `src/skin_lesion_risk/data/preprocessing.py`
4. `scripts/run_train.py`
5. `src/skin_lesion_risk/models/base.py`
6. `src/skin_lesion_risk/models/factory.py`
7. `src/skin_lesion_risk/models/adapters/tabular.py`
8. `src/skin_lesion_risk/models/adapters/image.py`
9. `src/skin_lesion_risk/models/adapters/monet.py`
10. `src/skin_lesion_risk/models/adapters/multimodal.py`
11. `scripts/build_graph.py`
12. `src/skin_lesion_risk/models/adapters/graph.py`
13. `scripts/evaluate.py`
14. `scripts/make_report_assets.py`

这样看会比较顺：先理解数据格式，再理解模型统一接口，最后理解每个模型适配器。
