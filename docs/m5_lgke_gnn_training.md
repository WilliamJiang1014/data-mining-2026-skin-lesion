# M5 LGKE-GNN Training Guide

This document describes the reproducible path from raw ISIC/PAD data to the M5 LGKE-GNN artifacts used by the report tables.

## 1. Local CPU Preprocessing

Run the full preprocessing pipeline from the repository root:

```bash
python3 scripts/prepare_data.py --image-check sample
```

Main outputs:

- `data/processed/manifest_isic.csv`
- `data/processed/manifest_pad.csv`
- `data/processed/folds_isic.csv`
- `data/processed/preprocessors/fold{k}_tabular.pkl`
- `reports/data_quality_isic.md`
- `reports/data_quality_pad.md`
- `reports/tables/split_stats.csv`

For a quick local smoke test, use a stratified subset:

```bash
python3 scripts/prepare_data.py \
  --max-samples 256 \
  --image-check sample \
  --image-check-sample-size 20
```

`--max-samples` is only for pipeline validation. Do not use it for final report numbers.

## 2. Build Fold Graphs

Build graph files after preprocessing:

```bash
python3 scripts/build_graph.py --fold 0
```

For a faster local graph smoke test:

```bash
python3 scripts/build_graph.py \
  --fold 0 \
  --max-samples 256 \
  --k-visual 3 \
  --k-metadata 3 \
  --n-visual-prototypes 8 \
  --n-metadata-prototypes 8
```

Main outputs:

- `data/processed/graph/fold0_train_graph.pt`
- `data/processed/graph/fold0_val_graph.pt`
- `data/processed/graph/fold0_test_graph.pt`
- `data/processed/graph/fold0_train_nodes.parquet`
- `data/processed/graph/fold0_train_edges.parquet`
- `data/processed/graph/fold0_graph_report.json`

The native graph contains lesion, patient, attribute, visual prototype and metadata prototype nodes. Validation/test query lesions connect to the training graph through KNN/prototype edges; thresholds remain selected only on validation predictions.

## 3. Local CPU Smoke Training

Run one epoch on CPU:

```bash
python3 scripts/run_train.py \
  --model m5_lgke_gnn \
  --fold 0 \
  --device cpu \
  --epochs 1 \
  --patience 1
```

Expected outputs:

- `data/artifacts/trained_models/m5_lgke_gnn/fold0/best.ckpt`
- `data/artifacts/trained_models/m5_lgke_gnn/fold0/train_log.csv`
- `data/artifacts/trained_models/m5_lgke_gnn/fold0/loss_curve.csv`
- `data/artifacts/trained_models/m5_lgke_gnn/fold0/val_predictions.csv`
- `data/artifacts/trained_models/m5_lgke_gnn/fold0/test_predictions.csv`
- `data/artifacts/trained_models/m5_lgke_gnn/fold0/metrics.json`
- `data/artifacts/trained_models/m5_lgke_gnn/fold0/config_resolved.yaml`

`metrics.json` includes report-aligned pAUC@TPR>=0.80, AUPRC, AUROC, sensitivity, specificity, FNR, Brier and ECE for both Youden and high-sensitivity validation thresholds.

## 4. Cluster Training

On the compute cluster, install the project with ML dependencies and run the same commands without `--max-samples`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml]"

python3 scripts/prepare_data.py --image-check sample

for fold in 0 1 2 3 4; do
  python3 scripts/build_graph.py --fold "$fold"
  python3 scripts/run_train.py \
    --model m5_lgke_gnn \
    --fold "$fold" \
    --device cuda
done
```

If the cluster has limited memory, reduce `--k-visual`, `--k-metadata`, `--n-visual-prototypes`, or `--n-metadata-prototypes` during graph construction. Keep the final values recorded in `config_resolved.yaml` and `fold{k}_graph_report.json`.

## 5. Important Rules

- Do not fit preprocessing, thresholds, calibration or KNN indices on validation/test/PAD data.
- Do not use `patient_id`, `lesion_id`, `sample_id`, diagnosis fields or pathology fields as ordinary tabular features.
- PAD-UFES-20 remains external validation by default; it is not mixed into ISIC training.
- Local CPU runs verify pipeline correctness only. Final report metrics should come from full folds on the cluster.
