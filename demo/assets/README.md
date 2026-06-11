# Demo Assets

本目录为 **Streamlit Demo 的预计算结果快照**，已随仓库提交。

他人 `git clone` 后可直接运行：

```bash
bash scripts/run_demo.sh
```

无需本地 `训练数据/`、`training_results_bundle` 或 ISIC 原始数据。

## 内容说明

| 路径 | 用途 |
|------|------|
| `sample_predictions_manifest.json` + `predictions/` + `metrics/` | 样本查看：M0/M4/M5 共 100 条对齐样本 |
| `m5_summary_snippet.json` | M5 五折汇总卡片 |
| `fairness_subgroup.csv` | 公平性子群表 |
| `graph_ablation_summary.csv` | 图结构消融 |
| `pad_adaptation_summary.csv` | PAD 外部验证 |
| `figures/` | 流程图、森林图等配图 |

主实验汇总表见仓库内 `reports/tables/`（`table4.csv`、`main_results.csv`、`split_stats.csv`）。

## 维护者：重新生成（可选）

仅在需要刷新 Demo 数据时，从本地实验归档重新抽样：

```bash
python scripts/prepare_demo_assets.py \
  --m0-m4-root /path/to/artifacts \
  --bundle-root /path/to/training_results_bundle
```

生成后请运行 `pytest` 并提交 `demo/assets/` 与 `reports/tables/` 的变更。
