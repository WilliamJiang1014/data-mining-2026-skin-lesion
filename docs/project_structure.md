# 项目文件树说明

```text
.
├── configs/
│   ├── datasets/          # 数据源、字段、路径和泄漏控制配置
│   ├── experiments/       # 主实验、消融、外部验证等实验配置
│   └── models/            # 每类模型的独立配置
├── data/
│   ├── raw/               # 原始数据，不提交 Git
│   ├── processed/         # manifest、fold、图边、schema
│   ├── interim/           # 每折预处理器、特征缓存
│   ├── external/          # 外部下载或补充资源
│   └── artifacts/         # checkpoint、预测文件、指标
├── docs/                  # 项目文档
├── notebooks/             # 探索分析 notebook
├── reports/
│   ├── figures/           # 报告图
│   └── tables/            # 报告表
├── scripts/               # 命令行入口
├── app/                   # 交互式展示预留入口
├── src/skin_lesion_risk/  # Python 包源码
│   ├── data/              # manifest schema 和数据处理工具
│   ├── evaluation/        # 指标、阈值、子群公平性
│   ├── features/          # 图像、元数据、文本特征抽取与缓存
│   ├── models/            # 模型工厂和适配器
│   ├── pipelines/         # build_manifest、build_graph、train、evaluate
│   ├── reporting/         # 结果表和图生成
│   ├── demo/              # demo 应用内部模块
│   └── utils/             # 随机种子等通用工具
└── tests/                 # 单元测试和 smoke tests
```

该结构对应报告模板中的可复现 Pipeline：数据下载、样本清单、患者级划分、元数据编码、图像增强、类别不均衡处理、图结构构建、模型训练、统计评估和结果展示。
