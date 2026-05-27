# 统一模型工厂使用说明

本文档说明本项目如何用统一模型工厂管理不同候选模型，并保证所有模型在同一数据集、同一划分策略和同一评价指标下可比较。

## 1. 设计目标

报告模板中的实验需要比较：

- `M0` 表格模型：只使用年龄、性别、解剖部位、尺寸、TBP 数值字段等元数据。
- `M1` CNN 图像模型：只使用病灶图像。
- `M2` MONET 或兼容视觉语言特征模型：使用冻结特征和分类头。
- `M3` 视觉 Transformer 模型：ViT、Swin 等图像模型。
- `M4` 多模态融合模型：图像编码器 + 元数据编码器。
- `M5` LGKE-GNN：图像、元数据和病灶图结构联合建模。

这些模型的内部实现差异很大，但实验比较需要统一以下内容：

- 数据来源：统一从 `manifest_isic.csv`、`manifest_pad.csv` 和每折图文件读取。
- 样本顺序：所有输入按 `sample_ids` 对齐。
- 训练接口：统一 `fit(train, valid)`。
- 推理接口：统一 `predict_proba(batch)`。
- 输出格式：统一输出每个样本的恶性风险分数 `score`。
- 评价指标：统一从 `PredictionResult` 计算或附加 `MetricBundle`。

对于图模型，输入确实比普通模型多一个 `graph` 字段；因此工厂只统一“外层训练/推理协议”和“输出评价协议”，不强行把所有模型内部输入压成同一种张量。

## 2. 文件位置

```text
src/skin_lesion_risk/models/
  base.py                 # ModelBatch、PredictionResult、LesionRiskModel 协议
  factory.py              # ModelFactory 和默认注册表
  registry.py             # 模型类型到适配器类的映射
  adapters/
    tabular.py            # 表格模型槽位和 ConstantRiskModel
    image.py              # CNN / Transformer 图像模型槽位
    monet.py              # MONET / 视觉语言特征模型槽位
    multimodal.py         # 常规多模态融合模型槽位
    graph.py              # LGKE-GNN 图模型槽位
evaluation/
  metrics.py              # 统一评价指标
configs/
  experiments/baselines.yaml
  models/*.yaml
```

## 3. 统一输入：ModelBatch

所有模型的训练和推理输入都使用 `ModelBatch`：

```python
from skin_lesion_risk.models.base import ModelBatch

batch = ModelBatch(
    sample_ids=["ISIC_001", "ISIC_002"],
    labels=[0, 1],
    image_paths=["data/raw/isic2024_permissive/images/ISIC_001.jpg", "..."],
    metadata=metadata_df,
    graph=graph_object_or_paths,
    groups={
        "sex": ["female", "male"],
        "anatom_site": ["torso", "lower_extremity"],
    },
    fold=0,
    source="isic2024_slice3d_permissive",
)
```

字段约定：

- `sample_ids`：必须提供，且决定本 batch 的样本顺序。
- `labels`：训练、验证、测试时提供；无标签外部推理可以为空。
- `image_paths` / `images`：图像模型、多模态模型使用。
- `metadata`：表格模型、多模态模型、图模型使用。
- `graph`：只由 LGKE-GNN 等图模型使用，可以是 PyG/DGL 对象，也可以是封装后的节点表和边表路径。
- `groups`：公平性评估使用，例如性别、年龄段、解剖部位、Fitzpatrick 类型。
- `fold`：患者级交叉验证折号。
- `source`：数据源名称，例如 `isic2024_slice3d_permissive` 或 `pad_ufes_20`。

## 4. 统一输出：PredictionResult

所有模型必须通过 `predict_proba` 返回 `PredictionResult`：

```python
from skin_lesion_risk.models.base import PredictionResult

result = PredictionResult(
    sample_ids=["ISIC_001", "ISIC_002"],
    scores=np.array([0.12, 0.83]),
    labels=np.array([0, 1]),
).with_metrics(threshold=0.5, threshold_rule="fixed_0_5")
```

字段约定：

- `sample_ids`：必须与输入 batch 一一对应。
- `scores`：每个样本的恶性风险分数，范围建议为 `[0, 1]`。
- `labels`：如果输入中有标签，则输出中保留标签。
- `threshold`：当前评价阈值。
- `metrics`：统一评价指标包。
- `metadata`：可保存模型版本、fold、checkpoint 路径、特征缓存路径等补充信息。

## 5. 统一评价指标

基础指标在 `skin_lesion_risk.evaluation.metrics` 中：

- `sensitivity`
- `specificity`
- `precision`
- `f1`
- `fnr`
- `brier`
- `auroc`，安装 `scikit-learn` 时计算
- `auprc`，安装 `scikit-learn` 时计算

报告中还需要补充的指标建议继续放在同一模块中：

- `partial_auc_high_sensitivity`
- `specificity_at_sensitivity_0_90`
- `ece`
- `subgroup_gap`

阈值规则应在验证集上确定，再应用到测试集和外部验证集：

- `youden`：最大 Youden 指数。
- `sensitivity_at_least_0_90`：满足验证集灵敏度不低于 0.90 的最低阈值。

不要在测试集或 PAD-UFES-20 外部验证集上重新选择阈值，除非单独作为补充实验报告。

## 6. 从配置创建模型

单模型配置位于 `configs/models/*.yaml`，例如：

```yaml
type: multimodal
backend: torch
class_name: PlaceholderMultimodalModel
params:
  image_encoder: convnext_tiny
  metadata_encoder: mlp
  fusion: gated
  loss: weighted_bce
```

创建模型：

```python
from skin_lesion_risk.models.factory import ModelFactory

factory = ModelFactory()
model = factory.create("configs/models/m4_multimodal.yaml", model_name="m4_multimodal_fusion")
```

查看已注册类型：

```python
factory.available()
```

当前默认注册类型：

- `constant`
- `tabular`
- `image`
- `image_transformer`
- `monet_feature`
- `multimodal`
- `graph_multimodal`

命令行检查实验配置：

```bash
python3 scripts/run_train.py --config configs/experiments/baselines.yaml --list-models
python3 scripts/run_train.py --config configs/experiments/baselines.yaml --model m4_multimodal_fusion
```

## 7. 添加一个新模型

以添加 `LightGBMTabularModel` 为例：

1. 在 `src/skin_lesion_risk/models/adapters/tabular.py` 中新增类。

```python
class LightGBMTabularModel(BaseModelAdapter):
    model_type = "tabular_lgbm"

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None):
        # 从 train.metadata 读取特征，从 train.labels 读取标签
        # 只在训练折拟合 imputer、encoder、scaler
        return self

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        # 保证输出顺序与 batch.sample_ids 一致
        return PredictionResult(sample_ids=batch.sample_ids, scores=scores, labels=batch.labels)
```

2. 在 `factory.py` 的 `default_registry()` 中注册。

```python
registry.register("tabular_lgbm", LightGBMTabularModel)
```

3. 新增配置文件。

```yaml
type: tabular_lgbm
backend: lightgbm
params:
  num_leaves: 31
  learning_rate: 0.03
  class_weight: balanced
```

4. 在 `configs/experiments/baselines.yaml` 中加入该模型。

```yaml
- name: m0_lightgbm
  type: tabular_lgbm
  config: configs/models/m0_lightgbm.yaml
```

## 8. 各模型应消费哪些字段

| 模型类型 | image_paths/images | metadata | graph | labels | groups |
|---|---:|---:|---:|---:|---:|
| `tabular` | 否 | 是 | 否 | 是 | 可选 |
| `image` | 是 | 否 | 否 | 是 | 可选 |
| `image_transformer` | 是 | 否 | 否 | 是 | 可选 |
| `monet_feature` | 是，可结合文本描述 | 可选 | 否 | 是 | 可选 |
| `multimodal` | 是 | 是 | 否 | 是 | 可选 |
| `graph_multimodal` | 通常是特征或路径 | 是 | 是 | 是 | 可选 |

`patient_id`、`lesion_id`、`sample_id` 默认不进入普通表格特征。它们可以用于：

- 患者级划分。
- 图模型的同患者边。
- 结果回写和错误分析。
- 数据质量审计。

## 9. 推荐训练流程

完整训练流水线后续应按以下顺序实现：

1. `build_manifest`：从原始图像、元数据、补充元数据和标签生成统一 manifest。
2. `split_folds`：使用 `StratifiedGroupKFold` 按 `patient_id` 构造患者级折。
3. `fit_preprocessor`：每折只在训练子集拟合表格预处理器、分箱规则、类别词表。
4. `extract_features`：需要时抽取冻结图像特征或 MONET 特征。
5. `build_graph`：只用训练折统计量构建图边和知识原型。
6. `train`：通过 `ModelFactory` 创建模型并调用统一 `fit`。
7. `evaluate`：统一调用 `predict_proba`，输出预测、指标、子群指标和校准结果。
8. `report`：把 `reports/tables/*.csv`、`reports/figures/*.png` 写入报告。

## 10. 泄漏控制

实现具体模型时必须遵守：

- 不把 `patient_id`、`lesion_id`、`sample_id` 当作普通监督特征。
- 不在验证、测试或外部验证集上拟合 scaler、imputer、类别词表、分箱边界、kNN 索引或阈值。
- 不用诊断文本、活检结果、病理确认方式、目标标签同义字段作为输入。
- 图边权、视觉近邻、元数据近邻只由训练折拟合；验证和测试只能接入训练图或使用自身特征构造不含标签信息的推理图。

## 11. 目前的可运行 smoke test

当前仓库提供 `ConstantRiskModel` 用于验证工厂和指标链路：

```python
import numpy as np
from skin_lesion_risk.models.base import ModelBatch
from skin_lesion_risk.models.factory import ModelFactory

model = ModelFactory().create({"type": "constant"}, model_name="smoke")
train = ModelBatch(sample_ids=["a", "b"], labels=np.array([0, 1]))
test = ModelBatch(sample_ids=["x"], labels=np.array([1]))

model.fit(train)
result = model.predict_proba(test).with_metrics(threshold=0.5)
print(result.to_records())
print(result.metrics.to_dict())
```

运行测试：

```bash
pytest
```
