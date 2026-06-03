# Scripts 使用指南

项目脚本按流水线组织：数据准备 → 训练 → 评估 → 报告汇总。

---

## 快速启动

```bash
cd <project_root>
conda activate skin
```

### 1. 准备数据

```bash
python scripts/prepare_data.py
```

输入：`data/raw/isic2024_permissive/` 下的图片和 CSV

输出：
- `data/processed/manifest_isic.csv` — 统一格式样本 manifest
- `data/processed/folds_isic.csv` — 5 折划分
- `data/processed/preprocessors/fold*_tabular.pkl` — 各折表格预处理器
- `reports/tables/split_stats.csv` — 划分统计
- `reports/data_quality_*.md` — 数据质量报告

参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--isic-root` | `data/raw/isic2024_permissive` | ISIC 数据根目录 |
| `--pad-root` | `data/raw/pad_ufes_20` | PAD 数据根目录 |
| `--skip-pad` | 否 | 跳过 PAD-UFES-20 |
| `--folds` | 5 | 折数 |
| `--val-ratio` | 0.1 | 验证集比例 |
| `--seed` | 2026 | 随机种子 |
| `--rare-min-count` | 20 | 类别特征低频合并阈值 |
| `--image-check` | `sample` | 图片检查级别：`none`/`sample`/`all` |
| `--image-check-sample-size` | 200 | `sample` 模式检查数量 |
| `--max-samples` | 全部 | 抽小样本做 CPU smoke test |

### 2. 训练模型

```bash
# M0（CPU）
python scripts/run_train.py --model m0_constant --all-folds
python scripts/run_train.py --model m0_lightgbm --all-folds

# M1–M4（GPU）
python scripts/run_train.py --model m1_cnn_baseline --all-folds --device cuda
python scripts/run_train.py --model m2_monet_feature_baseline --all-folds --device cuda
python scripts/run_train.py --model m3_transformer_baseline --all-folds --device cuda
python scripts/run_train.py --model m4_multimodal_fusion --all-folds --device cuda

# 单折训练
python scripts/run_train.py --model m1_cnn_baseline --fold 0 --device cuda
```

参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `configs/experiments/baselines.yaml` | 实验配置文件 |
| `--model` | 必填 | 模型名称，需与配置文件中一致 |
| `--fold N` | — | 运行第 N 折（与 `--all-folds` 二选一） |
| `--all-folds` | — | 运行全部折 |
| `--num-folds` | 5 | 总折数 |
| `--device` | `cpu` | `cpu` 或 `cuda` |
| `--training-config` | `configs/training.yaml` | 训练超参数配置 |
| `--log-dir` | `log` | 日志目录 |
| `--out-dir` | `data/artifacts/trained_models` | 模型输出目录 |
| `--manifest` | `data/processed/manifest_isic.csv` | manifest 文件 |
| `--folds` | `data/processed/folds_isic.csv` | 划分文件 |
| `--list-models` | — | 列出配置中所有可用模型 |

命令行覆盖参数（优先级高于 `configs/training.yaml`）：

| 参数 | 说明 |
|---|---|
| `--epochs` | 训练轮数 |
| `--patience` | 早停耐心值 |
| `--learning-rate` | 学习率 |
| `--weight-decay` | 权重衰减 |
| `--dropout` | Dropout 率 |
| `--batch-size` | 批大小 |

M5 专用覆盖参数：

| 参数 | 说明 |
|---|---|
| `--hidden-dim` | 隐藏层维度 |
| `--num-layers` | GNN 层数 |
| `--lambda-smooth` | 图平滑损失权重 |
| `--edge-chunk-size` | 大图边分块大小（OOM 时调小） |

训练产物（每折）：
```
data/artifacts/trained_models/{model_name}/fold{f}/
  best.ckpt 或 model.pkl      # 模型权重
  config_resolved.yaml         # 实际运行配置（含所有参数来源）
  train_log.csv                # epoch 级训练日志（深度模型）
  train_summary.csv            # 一行训练摘要
  loss_curve.csv               # loss 曲线
  val_predictions.csv          # 验证集预测
  test_predictions.csv         # 测试集预测
  metrics.json                 # 验证集和测试集指标 + 阈值
```

训练时每个 epoch 输出进度行：
```
[m4_multimodal_fusion] epoch 6/40  loss=0.0012  pAUC=0.7645  AUROC=0.9431  lr=1.50e-04  best=0.7645  patience_left=10
```

日志自动写入 `log/{model}_fold{N}.log`，同时打印到终端。

### 3. 评估模型

```bash
python scripts/evaluate.py --model m0_lightgbm --fold 0
python scripts/evaluate.py --model m4_multimodal_fusion --fold 0 --device cuda

# 5 折全评估
for f in 0 1 2 3 4; do
  python scripts/evaluate.py --model m4_multimodal_fusion --fold $f --device cuda
done
```

参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | 必填 | 模型名称 |
| `--fold` | 0 | 折号 |
| `--device` | `cpu` | `cpu` 或 `cuda` |
| `--config` | `configs/experiments/baselines.yaml` | 实验配置文件 |
| `--artifacts-dir` | `data/artifacts/trained_models` | 模型产物目录 |
| `--reports-dir` | `reports` | 报告目录 |

输出：更新对应 fold 的 `metrics.json` 和 `reports/tables/main_results.csv`

### 4. 汇总报告

```bash
python scripts/make_report_assets.py
```

扫描所有 `metrics.json`，汇总到 `reports/tables/main_results.csv`。

### 5. 论文 Table 4

```bash
python scripts/make_table4.py                    # CSV，mean ± std
python scripts/make_table4.py --use-ci           # 95% CI 代替 std
python scripts/make_table4.py --format latex     # LaTeX 格式
python scripts/make_table4.py --format both      # CSV + LaTeX
```

参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` | `reports/tables/main_results.csv` | 输入结果表 |
| `--output` | `reports/tables/table4.csv` | 输出文件 |
| `--format` | `csv` | 输出格式：`csv`/`latex`/`both` |
| `--use-ci` | 否 | 使用 95% CI 代替标准差 |

输出：`reports/tables/table4.csv`、`reports/tables/table4.tex`

表格列：Model、pAUC、AUPRC、AUROC、Sens.、Spec.，排除 M0 Constant，保留 LightGBM、M1–M5。

---

## M5 图构建

M5 LGKE-GNN 需要先构建图数据，M0–M4 不需要。

```bash
for f in 0 1 2 3 4; do
  python scripts/build_graph.py --fold $f --knn-device cuda
done
```

参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--fold` | 0 | 折号 |
| `--k-visual` | 10 | 视觉 kNN 邻居数 |
| `--k-metadata` | 10 | 元数据 kNN 邻居数 |
| `--n-visual-prototypes` | 64 | 视觉原型节点数 |
| `--n-metadata-prototypes` | 64 | 元数据原型节点数 |
| `--knn-device` | `auto` | kNN 设备：`auto`/`cuda`/`cpu` |
| `--knn-chunk-size` | 4096 | kNN 分块大小 |
| `--refresh-image-cache` | 否 | 重建图片特征缓存 |
| `--max-samples` | 全部 | 每图样本上限（smoke test 用） |

输出：`data/processed/graph/fold{f}_{split}_graph.pt`

---

## 多 GPU 并行训练

可同时在不同 GPU 上跑不同模型：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run_train.py --model m1_cnn_baseline --all-folds --device cuda > log/m1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run_train.py --model m2_monet_feature_baseline --all-folds --device cuda > log/m2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run_train.py --model m3_transformer_baseline --all-folds --device cuda > log/m3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run_train.py --model m4_multimodal_fusion --all-folds --device cuda > log/m4.log 2>&1 &
wait
```

查看训练日志：`tail -f log/m1.log`

---

## 脚本依赖关系

```
prepare_data.py → manifest_isic.csv, folds_isic.csv, preprocessors/
                     │
                     ├── run_train.py → trained_models/{model}/fold{f}/
                     │                      │
                     │                      ├── evaluate.py → metrics.json, main_results.csv
                     │                      │
                     │                      └── make_report_assets.py → main_results.csv
                     │
                     └── build_graph.py → graph/fold*_{split}_graph.pt  (仅 M5 需要)
                            │
                            └── run_train.py → m5_lgke_gnn
```

---

## 其他脚本

| 脚本 | 用途 |
|---|---|
| `run_pad_external_eval.py` | PAD-UFES-20 外部验证 |
| `run_pad_domain_adaptation.py` | ISIC→PAD 域适应实验 |

---

## 排错

| 问题 | 解决方案 |
|---|---|
| 找不到图片 | 运行 `prepare_data.py --image-check sample` |
| PAD 没下载 | 运行 `prepare_data.py --skip-pad` |
| M5 找不到 graph | 先运行 `build_graph.py --fold {f}` |
| CUDA OOM | 降低 `--batch-size` 或 image_size；M5 调小 `--edge-chunk-size` |
| 指标表为空 | 检查对应 fold 下是否有 `metrics.json` |
| 训练结果被覆盖 | 同 model+fold 会复用目录，改模型名或输出路径可保留多版本 |