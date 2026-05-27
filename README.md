# Multimodal Skin Lesion Risk Screening

本仓库用于课程项目报告中的可复现实验设计，围绕 ISIC 2024 SLICE-3D Permissive 主数据集和 PAD-UFES-20 外部验证数据集组织代码。

当前版本只搭建项目结构与统一模型工厂，不实现具体深度模型训练代码。后续每个成员可以在统一接口下补充表格模型、CNN、Transformer、MONET 特征模型、多模态融合模型和 LGKE-GNN。

核心入口：

- `src/skin_lesion_risk/models/factory.py`：统一模型工厂。
- `src/skin_lesion_risk/models/base.py`：统一输入、输出和模型协议。
- `src/skin_lesion_risk/evaluation/metrics.py`：统一评价指标定义。
- `docs/model_factory.md`：模型工厂使用说明。
- `configs/experiments/baselines.yaml`：候选模型实验配置样例。

推荐工作流：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml]"
pytest
```
