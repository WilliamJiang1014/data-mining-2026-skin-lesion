# Data Layout

数据文件不纳入 Git，只保留目录结构和字段约定。

```text
data/
  raw/
    isic2024_permissive/
      images/*.jpg
      metadata.csv
      supplemental_metadata.csv
      labels.csv
    pad_ufes_20/
      images/*.png
      metadata.csv
  processed/
    manifest_isic.csv
    manifest_pad.csv
    folds_isic.csv
    graph_edges_fold0.parquet
    ...
    tabular_schema.json
  interim/
    preprocessor_fold0.pkl
    image_features_fold0.parquet
  artifacts/
    trained_models/
    predictions/
    metrics/
```

统一 `manifest` 最小字段：

- `sample_id`
- `image_path`
- `target`
- `patient_id`
- `lesion_id`
- `age`
- `sex`
- `anatom_site`
- `size_mm`
- `source`
- `fold`

`patient_id`、`lesion_id`、`sample_id` 只用于分组、审计和图边构建，默认不作为普通表格特征输入。

