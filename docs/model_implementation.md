# M0–M4 模型实现文档

本文档记录本项目中负责的核心工作：M0–M4 模型实现、统一训练/评估管线、训练优化。

---

## 1. 最终结果

5 折交叉验证平均指标（主评价指标：pAUC@TPR≥0.80）：

| 模型 | pAUC↑ | AUROC | AUPRC | Sensitivity | Specificity |
|---|---|---|---|---|---|
| M0 Constant | 0.1000 | 0.5000 | 0.0013 | 1.0000 | 0.0000 |
| M0 LightGBM | 0.7303 | 0.9324 | 0.0667 | 0.9176 | 0.7677 |
| M1 CNN | 0.5408 | 0.8734 | 0.0440 | 0.8514 | 0.6537 |
| M2 MONET | 0.5892 | 0.8826 | 0.0408 | 0.7942 | 0.8089 |
| M3 Transformer | 0.6418 | 0.9011 | 0.0471 | 0.8486 | 0.7956 |
| **M4 Multimodal** | **0.7393** | **0.9328** | **0.0765** | 0.8680 | 0.8445 |

**M4 多模态融合模型取得最优结果**，pAUC 比次优的 M0 LightGBM 高 0.009。

---

## 2. 统一模型框架

所有模型通过统一工厂管线创建：**YAML 配置 → ModelRegistry → BaseModelAdapter 子类**。

### 2.1 核心接口

**统一输入 `ModelBatch`**（`src/skin_lesion_risk/models/base.py`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_ids` | list[str] | 样本唯一标识 |
| `labels` | np.ndarray | 标签（0=良性，1=恶性） |
| `image_paths` | list[str] | 图像文件路径，图像/多模态模型使用 |
| `metadata` | DataFrame | 预处理后的表格特征，表格/多模态模型使用 |
| `raw_metadata` | DataFrame | 原始临床字段（age、sex、anatom_site 等），M2 文本提示使用 |

**统一输出 `PredictionResult`**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_ids` | list[str] | 与输入 batch 一一对应 |
| `scores` | np.ndarray | 每个样本的恶性风险分数 [0, 1] |
| `labels` | np.ndarray | 真实标签 |

**适配器协议 `LesionRiskModel`**：每个适配器实现 `fit()`、`predict_proba()`、`save()`、`load()` 四个方法。

### 2.2 模型注册表

`default_registry()` 注册 8 种模型类型，torch 依赖的模型通过 `try/except ImportError` 延迟注册，M0 表格模型无需 torch 即可运行：

| 类型键 | 适配器类 | 文件 |
|---|---|---|
| `constant` | ConstantRiskModel | adapters/tabular.py |
| `tabular_lgbm` | LightGBMTabularModel | adapters/tabular.py |
| `image` | CNNImageModel | adapters/image.py |
| `image_transformer` | TransformerImageModel | adapters/image.py |
| `monet_feature` | MonetFeatureModel | adapters/monet.py |
| `multimodal` | MultimodalFusionModel | adapters/multimodal.py |

### 2.3 关键文件

| 文件 | 作用 |
|---|---|
| `src/skin_lesion_risk/models/base.py` | `ModelBatch`、`PredictionResult`、`BaseModelAdapter` 基类 |
| `src/skin_lesion_risk/models/registry.py` | 类型字符串到适配器类的映射 |
| `src/skin_lesion_risk/models/factory.py` | `ModelFactory.create()` 加载配置创建模型 |

---

## 3. 模型实现详情

### 3.1 M0-1 先验常数基线（ConstantRiskModel）

**文件**：`src/skin_lesion_risk/models/adapters/tabular.py`

预测所有样本为训练集阳性率常数。设计目的：作为最低基线，验证评估管线端到端正确性。

- 类型键：`constant`
- 输入：仅 `labels`
- 输出：所有样本输出同一分数 = `train.labels.mean()`
- 保存格式：pickle (`model.pkl`)
- 硬件需求：CPU，秒级完成

### 3.2 M0-2 LightGBM 表格模型（LightGBMTabularModel）

**文件**：`src/skin_lesion_risk/models/adapters/tabular.py`

| 项目 | 说明 |
|---|---|
| 类型键 | `tabular_lgbm` |
| 方法 | LightGBM 梯度提升树，使用预处理后的表格特征 |
| 输入 | `metadata`（预处理后的数值+类别特征） |
| 输出 | `predict_proba()` 的正类概率 |
| 保存格式 | pickle (`model.pkl`) |
| 硬件需求 | CPU，分钟级完成 |

特征处理（`_metadata_frame()`）：
- 自动删除 ID 列（`patient_id`、`sample_id`、`lesion_id`、`image_id`、`isic_id`）
- 所有列强制转为数值类型，缺失值填充 0.0
- fit 时记录特征名列表，transform 时保证特征对齐（缺失列补 0，多余列删除）

核心参数：`n_estimators=300`、`learning_rate=0.03`、`num_leaves=31`、`class_weight=balanced`

回退机制：LightGBM 不可用时自动回退到 sklearn `HistGradientBoostingClassifier`。

### 3.3 M1 CNN 图像模型（CNNImageModel）

**文件**：`src/skin_lesion_risk/models/adapters/image.py`

| 项目 | 说明 |
|---|---|
| 类型键 | `image` |
| 骨干网络 | EfficientNet-B0（timm，ImageNet pretrained） |
| 特征维度 | 1280 |
| 分类头 | Dropout(0.3) → Linear(1280, 256) → GELU → Dropout(0.15) → Linear(256, 1) |
| 损失函数 | BCEWithLogitsLoss（pos_weight 加权，上限 100.0） |
| 优化器 | AdamW（backbone lr × 0.1，head lr × 1.0） |
| 学习率调度 | LinearLR warmup（5 epoch）→ CosineAnnealingLR |
| 早停策略 | 基于验证集 pAUC@TPR≥0.80，patience=10 |
| 保存格式 | torch checkpoint (`best.ckpt`) |

训练优化：
- **AMP 混合精度**：`torch.amp.autocast("cuda")` + `GradScaler`，训练和推理均启用，速度提升 ~30%
- **Differential LR**：backbone 用 `lr × backbone_lr_scale`（0.1×），head 用原始 lr
- **Gradient Clipping**：`clip_grad_norm_=1.0`

图像增强（训练时）：
- Resize(384×384) → RandomHorizontalFlip → RandomVerticalFlip → RandomRotation(30°) → ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.04) → ToTensor → Normalize(ImageNet) → RandomErasing(p=0.3)

图像增强（验证/测试时）：
- Resize(384×384) → ToTensor → Normalize(ImageNet)

### 3.4 M2 MONET 特征模型（MonetFeatureModel）

**文件**：`src/skin_lesion_risk/models/adapters/monet.py`

| 项目 | 说明 |
|---|---|
| 类型键 | `monet_feature` |
| 图像编码器 | EfficientNet-B0（timm，pretrained） |
| 图像特征维度 | 1280 |
| 文本特征维度 | 64（稳定哈希编码） |
| 拼接特征维度 | 1280 + 64 = 1344 |
| 分类头 | Linear(1344, 256) → GELU → LayerNorm(256) → Dropout(0.3) → Linear(256, 64) → GELU → Dropout(0.15) → Linear(64, 1) |
| 损失函数 | BCEWithLogitsLoss（pos_weight 加权，上限 100.0） |
| 优化器 | AdamW |
| 早停策略 | 基于验证集 pAUC@TPR≥0.80，patience=10 |

**Progressive Unfreeze 训练策略**（核心优化）：

1. **前 5 epoch**：冻结 encoder 所有参数（`requires_grad=False`），只训练 MLP head
   - Optimizer 仅含 head 参数
   - LinearLR warmup（5 epoch）→ CosineAnnealingLR
2. **第 6 epoch**：解冻 encoder，重建 optimizer 加入 encoder 参数
   - Encoder 用 `lr × encoder_lr_scale`（0.1×）的低学习率
   - 新的 LinearLR warmup（3 epoch）→ CosineAnnealingLR
   - 重建 GradScaler

效果：在小样本极度不平衡场景下，backbone 早期梯度干扰 head 学习，progressive unfreeze 让 head 先稳定收敛，再让 backbone 低 lr 微调，避免 warmup 不稳定。

**文本提示特征**（`src/skin_lesion_risk/features/text_prompts.py`）：

- 从 `raw_metadata`（原始临床字段）构建临床描述文本
- 格式：`"Clinical skin lesion image with patient age approximately 55, sex male, anatomical site torso, long diameter 3.5 mm. TBP visual metadata: color variation 2.1, border irregularity 1.8."`
- 通过 `hashlib.md5` 稳定哈希编码为 64 维向量（不受 `PYTHONHASHSEED` 影响）
- 编码方式：token → md5 hash → 映射到 128×64 投影矩阵行索引 → 累加投影向量 → L2 归一化

> **注意**：当前 M2 使用 frozen EfficientNet-B0 + hash text features，不是真正的 MONET/VLM。后续接入真实 MONET 权重时替换 encoder 即可。

### 3.5 M3 视觉 Transformer 模型（TransformerImageModel）

**文件**：`src/skin_lesion_risk/models/adapters/image.py`（与 M1 共享文件）

| 项目 | 说明 |
|---|---|
| 类型键 | `image_transformer` |
| 骨干网络 | Swin-Tiny（timm，`swin_tiny_patch4_window7_224`，ImageNet pretrained） |
| 特征维度 | 768 |
| 分类头 | Dropout(0.3) → Linear(768, 256) → GELU → Dropout(0.15) → Linear(256, 1) |
| 损失函数 | Focal Loss（alpha=0.25, gamma=2.0, pos_weight 加权） |
| 优化器 | AdamW（backbone lr × 0.1，head lr × 1.0） |
| 学习率调度 | LinearLR warmup（5 epoch）→ CosineAnnealingLR |
| 早停策略 | 基于验证集 pAUC@TPR≥0.80，patience=10 |
| 保存格式 | torch checkpoint (`best.ckpt`) |

**Focal Loss** 实现（`image.py` 中的 `FocalLoss` 类）：

```
loss = alpha_factor × (1 - p_t)^gamma × BCE_with_logits
```

- `alpha=0.25`：降低正类（恶性）的损失权重，缓解类别不均衡
- `gamma=2.0`：降低易分类样本的损失贡献，聚焦难分类样本
- 结合 `pos_weight` 加权，进一步调整正负样本比例

训练优化：
- **AMP 混合精度**：与 M1 相同的实现模式
- **Differential LR**：backbone 用 0.1× lr
- 较小学习率 8e-5，较大权重衰减 0.05（Transformer 微调需要更强正则化）
- batch_size=96（在 32GB GPU 上）

### 3.6 M4 多模态融合模型（MultimodalFusionModel）

**文件**：`src/skin_lesion_risk/models/adapters/multimodal.py`

| 项目 | 说明 |
|---|---|
| 类型键 | `multimodal` |
| 图像编码器 | ConvNeXt-Tiny（timm，pretrained） |
| 图像特征维度 | 768 |
| 元数据编码器 | MetadataEncoder: MLP（input_dim → 128 → 64），含 LayerNorm |
| 元数据输出维度 | 64 |
| 融合策略 | Gated Fusion |
| 融合后维度 | 768 + 64 = 832 |
| 分类头 | Linear(832, 256) → GELU → LayerNorm(256) → Dropout(0.3) → Linear(256, 64) → GELU → LayerNorm(64) → Dropout(0.15) → Linear(64, 1) |
| 损失函数 | BCEWithLogitsLoss（pos_weight 加权，上限 100.0） |
| 优化器 | AdamW（所有模块联合优化） |
| 早停策略 | 基于验证集 pAUC@TPR≥0.80，patience=10 |
| 保存格式 | torch checkpoint (`best.ckpt`) |

**Gated Fusion 机制**（`GatedFusion` 类）：

```
gate_weights = Sigmoid(Linear(GELU(Linear([image_features; metadata_features]))))
gated_image = gate_weights * image_features
scaled_metadata = metadata_scale * metadata_features   # 可学习缩放
fused = [gated_image; scaled_metadata]
```

- 门控网络接收图像和元数据特征的拼接作为输入
- 输出与图像特征维度相同的 sigmoid 门控权重
- 对图像特征逐元素加权，保留元数据特征的完整信息
- `metadata_scale` 是可学习参数，初始化为 1.0

**Progressive Unfreeze 训练策略**（核心优化）：

1. **前 5 epoch**：冻结 image_encoder 所有参数（`requires_grad=False`），只训练 metadata_encoder + gate + head
   - Optimizer 仅含 metadata_encoder、gate、head 参数
   - LinearLR warmup（5 epoch）→ CosineAnnealingLR
2. **第 6 epoch**：解冻 image_encoder，重建 optimizer 加入 encoder 参数
   - Encoder 用 `lr × backbone_lr_scale`（0.1×）的低学习率
   - 新的 LinearLR warmup（3 epoch）→ CosineAnnealingLR
   - 重建 GradScaler

效果：M4 val pAUC 从 ~0.71 提升到 ~0.74，超越 M2。

**MultimodalDataset**：同时返回 `(image, metadata_row, label)` 三元组，保证 DataLoader 的 shuffle 操作不会导致图像和元数据错位。

训练优化：
- **AMP 混合精度**：训练和推理均启用
- **clip_grad_norm_ 优化**：冻结阶段跳过 image_encoder 参数，只 clip metadata_encoder + gate + head 的梯度
- **Differential LR**：encoder 解冻后用 0.1× lr
- batch_size=128，epochs=40，patience=10

---

## 4. 训练配置

统一配置文件 `configs/training.yaml`，各模型超参数：

| 参数 | M1 CNN | M2 MONET | M3 Transformer | M4 Multimodal |
|---|---|---|---|---|
| batch_size | 80 | 80 | 96 | 128 |
| epochs | 30 | 30 | 30 | 40 |
| learning_rate | 1.3e-4 | 2.2e-4 | 8e-5 | 1.5e-4 |
| backbone_lr_scale | 0.1 | 0.1 | 0.1 | 0.1 |
| weight_decay | 1e-4 | 1e-4 | 0.05 | 1e-4 |
| dropout | 0.3 | 0.3 | 0.3 | 0.3 |
| warmup_epochs | 5 | 5 | 5 | 5 |
| unfreeze_after | — | 5 | — | 5 |
| max_pos_weight | 100.0 | 100.0 | 100.0 | 100.0 |
| grad_clip_norm | 1.0 | 1.0 | 1.0 | 1.0 |
| image_size | 384 | 384 | 384 | 384 |

命令行参数可覆盖 YAML 配置：

```bash
python scripts/run_train.py --model m4_multimodal_fusion --fold 0 --device cuda \
  --epochs 50 --patience 15 --learning-rate 2e-4 --batch-size 64
```

---

## 5. 训练与评估命令

### 5.1 训练

```bash
# M0（CPU）
python scripts/run_train.py --model m0_constant --fold 0
python scripts/run_train.py --model m0_lightgbm --fold 0

# M1–M4（GPU）
python scripts/run_train.py --model m1_cnn_baseline --fold 0 --device cuda
python scripts/run_train.py --model m2_monet_feature_baseline --fold 0 --device cuda
python scripts/run_train.py --model m3_transformer_baseline --fold 0 --device cuda
python scripts/run_train.py --model m4_multimodal_fusion --fold 0 --device cuda
```

训练脚本功能：
- 自动从 `manifest_isic.csv` 和 `folds_isic.csv` 构建各折的 `ModelBatch`
- 训练前清理旧的产物文件，避免残留
- 保存 `config_resolved.yaml` 记录实际运行配置
- 训练完成后自动评估验证集和测试集
- 输出 `metrics.json`、`train_log.csv`、`train_summary.csv`、`val_predictions.csv`、`test_predictions.csv`

### 5.2 评估

```bash
python scripts/evaluate.py --model m4_multimodal_fusion --fold 0 --device cuda
```

- M0 评估无需 torch（torch 和图像 adapter 延迟导入到函数内部）
- 支持 CUDA 训练 → CPU 评估的跨设备场景

### 5.3 产物路径

```
data/artifacts/trained_models/{model_name}/fold{f}/
  best.ckpt 或 model.pkl          # 最佳模型权重
  config_resolved.yaml            # 实际运行配置
  train_log.csv                   # epoch 级训练日志
  train_summary.csv               # 一行训练摘要
  loss_curve.csv                  # 训练/验证 loss 曲线
  val_predictions.csv             # 验证集预测
  test_predictions.csv            # 测试集预测
  metrics.json                    # 验证集和测试集指标
```

主结果表：`reports/tables/main_results.csv`

---

## 6. 数据管线

### 6.1 数据划分

患者级分层 5 折划分（`src/skin_lesion_risk/data/splits.py`）：

1. **外层划分**：`StratifiedGroupKFold(n_splits=5)` 按 `patient_id` 分组，保证同一患者的所有病灶样本不跨折泄漏
2. **内层划分**：从训练+验证池中按 `val_ratio=0.1` 使用 `StratifiedShuffleSplit` 划分验证集
3. **泄漏断言**：每折均通过 `assert_no_group_overlap()` 检查

各折样本分布：

| Fold | 训练集 | 验证集 | 测试集 |
|---|---|---|---|
| 0 | 154,204 (70.9%) | 19,778 (9.1%) | 43,495 (20.0%) |
| 1 | 157,975 (72.6%) | 16,007 (7.4%) | 43,495 (20.0%) |
| 2 | 159,530 (73.4%) | 14,453 (6.6%) | 43,494 (20.0%) |
| 3 | 154,527 (71.1%) | 19,451 (8.9%) | 43,499 (20.0%) |
| 4 | 152,966 (70.3%) | 21,017 (9.7%) | 43,494 (20.0%) |

产物：`data/processed/folds_isic.csv`

### 6.2 数据预处理

`FoldTabularPreprocessor`（`src/skin_lesion_risk/data/preprocessing.py`），每折独立拟合：

- **数值特征**（35 个 TBP 字段）：中位数填充 → IQR 缩放 → 缺失指示列
- **类别特征**（4 个字段：sex、anatom_site 等）：缺失填 "UNK" → 低频合并 "RARE" → one-hot 编码
- **分位数分箱**：age 和 size_mm 按训练集四分位数分箱，用于公平性子群分析

---

## 7. 评价指标

| 指标 | 说明 |
|---|---|
| pAUC@TPR≥0.80 | 部分 AUC（主评价指标），在灵敏度 ≥ 0.80 区域内计算 ROC 面积并归一化 |
| AUROC | ROC 曲线下面积 |
| AUPRC | PR 曲线下面积 |
| Sensitivity | 灵敏度（真阳性率） |
| Specificity | 特异度（真阴性率） |
| Brier | Brier 分数（概率校准） |
| ECE | 期望校准误差 |

pAUC@TPR≥0.80 计算方式：使用 sklearn `roc_curve` 获取 FPR/TPR 曲线 → 截取 TPR ≥ 0.80 区域 → 计算该区域内 ROC 曲线与 TPR=0.80 水平线之间的面积 → 归一化（`raw_area / 0.20`），结果范围 [0, 1]。

阈值规则（仅在验证集上确定，直接应用到测试集）：
- **Youden**：最大 Youden 指数 = max(sensitivity + specificity - 1) 对应的阈值
- **Sensitivity≥0.90**：满足验证集灵敏度不低于 0.90 的最低阈值

---

## 8. 泄漏控制

1. **患者级划分**：StratifiedGroupKFold 按 `patient_id` 分组，同一患者不跨折
2. **预处理仅训练拟合**：scaler、imputer、encoder、分箱边界、类别词表仅在训练折学习参数
3. **ID 列不入特征**：`patient_id`、`sample_id`、`lesion_id`、`image_id`、`isic_id` 不作为模型输入
4. **阈值仅验证集选择**：决策阈值不在测试集上重新选择
5. **诊断字段不入特征**：诊断文本、活检结果、病理确认方式不作为输入
