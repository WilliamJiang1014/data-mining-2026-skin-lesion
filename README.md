# 多模态皮肤病变风险筛查与公平性评估

课程研究型代码仓库，围绕 **ISIC 2024 SLICE-3D Permissive** 主数据集与 **PAD-UFES-20** 外部验证，实现表格 / 图像 / 多模态融合 / LGKE-GNN 等模型的可复现实验流水线。

**仓库地址：** https://github.com/WilliamJiang1014/data-mining-2026-skin-lesion

> **免责声明：** 本项目仅用于课程研究与方法验证，**不构成医学诊断建议**，不可用于临床决策。

---

## Clone 后三步（验收 / 答辩）

仓库已内置 `demo/assets/` 与 `reports/tables/`，**不依赖仓库外的 `训练数据/` 目录**。

```bash
git clone https://github.com/WilliamJiang1014/data-mining-2026-skin-lesion.git
cd data-mining-2026-skin-lesion
conda env create -f environment.yml && conda activate skin
bash scripts/run_all.sh --smoke    # 一键自检：pytest + 报表 smoke
bash scripts/run_demo.sh           # 启动 Streamlit Demo
```

成功判据：`pytest` 全部通过；终端出现 `smoke done`；浏览器可打开 `http://localhost:8501` 并看到四个 Tab。

---

## 环境安装

推荐使用 Conda（环境名 `skin`）：

```bash
cd data-mining-2026-skin-lesion
conda env create -f environment.yml
conda activate skin
```

或使用 pip：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

GPU 训练需自行安装 CUDA 版 PyTorch，参见 [`requirements.txt`](requirements.txt) 顶部说明。

验证安装：

```bash
pytest
```

---

## 最快路径：启动 Demo（答辩推荐）

`demo/assets/` 已包含精简实验结果快照，**clone 后无需本地 `训练数据/` 或 `training_results_bundle` 即可演示**：

```bash
conda activate skin   # 或 source .venv/bin/activate
bash scripts/run_demo.sh
# 等价于：streamlit run src/skin_lesion_risk/demo/app.py
```

浏览器打开终端提示的本地地址（默认 `http://localhost:8501`）。

---

## 可选：Smoke 测试与更新 Demo 资产

### Smoke 测试（本地 CPU，无需 GPU）

```bash
bash scripts/run_all.sh --smoke
```

该命令会执行 `pytest`，并在有本地 raw 数据时跑一轮 M0 单折 smoke；无 raw 数据时会自动跳过训练步骤。

运行成功判据（关键输出）：

- 终端出现 `10 passed`（或更高）测试通过提示
- 终端出现 `smoke done. Launch demo` 提示
- `reports/smoke/tables/table4.csv` 被更新生成

### 从本地实验归档更新 Demo 数据

若需在本地刷新 Demo 展示数据（维护者场景），可从仓库外训练目录重新抽样：

```bash
python scripts/prepare_demo_assets.py \
  --m0-m4-root ../训练数据/artifacts \
  --bundle-root ../训练数据/training_results_bundle
```

`--bundle-root` 可省略：脚本会自动尝试 `../训练数据/training_results_bundle` 与 `../training_results_bundle`。

生成内容写入 [`demo/assets/`](demo/assets/)，详见 [`demo/assets/README.md`](demo/assets/README.md)。

### Demo 启动成功判据

```bash
bash scripts/run_demo.sh
```

若启动成功，应满足：

- 终端出现 `Local URL: http://localhost:8501`
- 浏览器可访问 `http://localhost:8501`
- 页面可看到四个 Tab：`概览` / `模型结果` / `样本查看` / `公平性与图结构`

说明：日常答辩演示仅依赖仓库内 `demo/assets/` 与 `reports/tables/`。本地 `训练数据/` 仅在“刷新 demo 资产”时需要。

---

## 完整实验复现

完整流水线（数据准备 → M0–M5 训练 → 评估 → 报告汇总）见 **[`scripts/README.md`](scripts/README.md)**。

简要顺序：

1. 下载 ISIC / PAD 数据至 `data/raw/`（见 [`data/README.md`](data/README.md)）
2. `python scripts/prepare_data.py`
3. `python scripts/run_train.py --model <name> --fold 0 --device cuda`（M1–M5 需 GPU）
4. `python scripts/evaluate.py --model <name> --fold 0`
5. `python scripts/make_report_assets.py && python scripts/make_table4.py`

M5 需先构建图：`python scripts/build_graph.py --fold 0`

---

## 已有结果（无需重训即可查看）

| 文件 | 说明 |
|------|------|
| [`reports/tables/table4.csv`](reports/tables/table4.csv) | M0–M4 五折汇总（论文 Table 4） |
| [`reports/tables/main_results.csv`](reports/tables/main_results.csv) | M0–M5 各折详细指标 |
| [`reports/tables/split_stats.csv`](reports/tables/split_stats.csv) | 患者级划分统计 |
| [`demo/assets/`](demo/assets/) | Demo 用 M0/M4/M5 样本预测与阈值快照、公平性、图消融、PAD 摘要 |

M5 五折汇总见 `demo/assets/m5_summary_snippet.json`；若本地有训练 bundle，可用 `python scripts/append_m5_main_results.py` 从真实 `metrics.json` 刷新 M5 各折行。

---

## 项目结构

```
├── configs/           # 数据集、模型、实验配置
├── demo/assets/       # Demo 展示用精简结果（已提交 Git）
├── docs/              # 项目文档
├── scripts/           # 命令行入口（详见 scripts/README.md）
├── src/skin_lesion_risk/  # Python 包
│   ├── data/          # 数据处理
│   ├── models/        # 模型工厂与适配器
│   ├── evaluation/    # 评估指标
│   └── demo/          # Streamlit Demo 入口
├── reports/tables/    # 实验结果表
└── tests/             # 单元测试
```

更完整的目录说明见 [`docs/project_structure.md`](docs/project_structure.md)。

---

## 数据来源

| 数据集 | 用途 | 链接 |
|--------|------|------|
| ISIC 2024 SLICE-3D Permissive | 主训练 / 验证 / 测试 | https://challenge.isic-archive.com/data/ |
| PAD-UFES-20 | 外部验证 / Fitzpatrick 公平性 | https://data.mendeley.com/datasets/zr7vgbcyr2/1 |

---

## 相关文档

- 脚本详细用法：[`scripts/README.md`](scripts/README.md)
- 模型实现说明：[`docs/model_implementation.md`](docs/model_implementation.md)
- M5 训练说明：[`docs/m5_lgke_gnn_training.md`](docs/m5_lgke_gnn_training.md)
