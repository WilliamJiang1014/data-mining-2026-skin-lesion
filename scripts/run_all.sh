#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="smoke"
BUNDLE_ROOT=""
M0_M4_ROOT=""
SKIP_DEMO_ASSETS=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_all.sh [--smoke|--full] [--bundle-root PATH] [--m0-m4-root PATH] [--skip-demo-assets]

Modes:
  --smoke            CPU smoke workflow (default)
  --full             Full workflow; GPU steps run only when CUDA is available

Options:
  --bundle-root PATH Optional path for prepare_demo_assets.py (maintainers only)
  --m0-m4-root PATH Optional M0-M4 artifacts root for prepare_demo_assets.py (maintainers only)
  --skip-demo-assets Skip calling scripts/prepare_demo_assets.py
  -h, --help         Show help
EOF
}

log() {
  printf '[run_all] %s\n' "$1"
}

run_cmd() {
  printf '[run_all] $ %s\n' "$*"
  "$@"
}

exists_file() {
  [[ -f "$1" ]]
}

demo_assets_ready() {
  python3 - <<'PY'
import json
from pathlib import Path
import sys

root = Path("demo/assets")
manifest_path = root / "sample_predictions_manifest.json"
if not manifest_path.exists():
    sys.exit(1)

try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)

models = manifest.get("models", [])
if not isinstance(models, list) or not models:
    sys.exit(1)

for item in models:
    if not isinstance(item, dict):
        sys.exit(1)
    pred_rel = str(item.get("predictions", "")).strip()
    if not pred_rel or not (root / pred_rel).exists():
        sys.exit(1)
    metrics_rel = str(item.get("metrics", "")).strip()
    if metrics_rel and not (root / metrics_rel).exists():
        sys.exit(1)

sys.exit(0)
PY
}

has_raw_isic() {
  exists_file "data/raw/isic2024_permissive/metadata.csv"
}

has_cuda() {
  python3 - <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(1)
sys.exit(0 if torch.cuda.is_available() else 1)
PY
}

prepare_demo_assets_if_needed() {
  if [[ "$SKIP_DEMO_ASSETS" -eq 1 ]]; then
    log "skip demo assets generation by flag"
    return
  fi

  if demo_assets_ready; then
    log "demo/assets already present, skip generation"
    return
  fi

  if [[ -z "$M0_M4_ROOT" && -z "$BUNDLE_ROOT" ]]; then
    log "ERROR: demo/assets incomplete. Clone should include committed demo/assets/."
    log "Maintainers may regenerate with:"
    log "  python scripts/prepare_demo_assets.py --m0-m4-root <artifacts> --bundle-root <bundle>"
    exit 1
  fi

  local cmd=(python3 scripts/prepare_demo_assets.py)
  if [[ -n "$M0_M4_ROOT" ]]; then
    cmd+=(--m0-m4-root "$M0_M4_ROOT")
  fi
  if [[ -n "$BUNDLE_ROOT" ]]; then
    cmd+=(--bundle-root "$BUNDLE_ROOT")
  fi
  run_cmd "${cmd[@]}"
}

run_smoke() {
  log "start smoke workflow"
  run_cmd python3 -m pytest -q
  prepare_demo_assets_if_needed

  local smoke_processed="data/processed/smoke"
  local smoke_reports="reports/smoke"
  local smoke_artifacts="data/artifacts/trained_models/smoke"

  mkdir -p "$smoke_reports/tables"

  if has_raw_isic; then
    run_cmd python3 scripts/prepare_data.py \
      --skip-pad \
      --image-check none \
      --max-samples 500 \
      --out-dir "$smoke_processed" \
      --reports-dir "$smoke_reports"

    if exists_file "$smoke_processed/manifest_isic.csv" && exists_file "$smoke_processed/folds_isic.csv"; then
      run_cmd python3 scripts/run_train.py \
        --model m0_lightgbm \
        --fold 0 \
        --device cpu \
        --manifest "$smoke_processed/manifest_isic.csv" \
        --folds "$smoke_processed/folds_isic.csv" \
        --preprocessor-dir "$smoke_processed/preprocessors" \
        --graph-dir "$smoke_processed/graph" \
        --out-dir "$smoke_artifacts" \
        --reports-dir "$smoke_reports"

      run_cmd python3 scripts/evaluate.py \
        --model m0_lightgbm \
        --fold 0 \
        --device cpu \
        --manifest "$smoke_processed/manifest_isic.csv" \
        --folds "$smoke_processed/folds_isic.csv" \
        --preprocessor-dir "$smoke_processed/preprocessors" \
        --graph-dir "$smoke_processed/graph" \
        --artifacts-dir "$smoke_artifacts" \
        --reports-dir "$smoke_reports"

      run_cmd python3 scripts/make_report_assets.py \
        --artifacts-dir "$smoke_artifacts" \
        --out "$smoke_reports/tables/main_results.csv"

      run_cmd python3 scripts/make_table4.py \
        --input "$smoke_reports/tables/main_results.csv" \
        --output "$smoke_reports/tables/table4.csv"
    else
      log "smoke manifest/folds missing, skip train/eval"
    fi
  else
    log "raw ISIC not found, skip data prep and training"
    if exists_file "reports/tables/main_results.csv"; then
      run_cmd python3 scripts/make_table4.py \
        --input "reports/tables/main_results.csv" \
        --output "$smoke_reports/tables/table4.csv"
    fi
  fi

  log "smoke done. Launch demo: streamlit run src/skin_lesion_risk/demo/app.py"
}

run_full() {
  log "start full workflow"
  run_cmd python3 -m pytest -q
  prepare_demo_assets_if_needed

  if has_raw_isic; then
    run_cmd python3 scripts/prepare_data.py
  else
    log "raw ISIC not found, skip data prep/training. See data/README.md."
    return
  fi

  run_cmd python3 scripts/run_train.py --model m0_constant --all-folds --device cpu
  run_cmd python3 scripts/run_train.py --model m0_lightgbm --all-folds --device cpu

  if has_cuda; then
    run_cmd python3 scripts/run_train.py --model m1_cnn_baseline --all-folds --device cuda
    run_cmd python3 scripts/run_train.py --model m2_monet_feature_baseline --all-folds --device cuda
    run_cmd python3 scripts/run_train.py --model m3_transformer_baseline --all-folds --device cuda
    run_cmd python3 scripts/run_train.py --model m4_multimodal_fusion --all-folds --device cuda

    for fold in 0 1 2 3 4; do
      run_cmd python3 scripts/build_graph.py --fold "$fold" --knn-device cuda
    done
    run_cmd python3 scripts/run_train.py --model m5_lgke_gnn --all-folds --device cuda
  else
    log "CUDA unavailable; skip M1-M5 training and graph build."
    log "Run these later in GPU environment:"
    log "  python3 scripts/run_train.py --model m1_cnn_baseline --all-folds --device cuda"
    log "  python3 scripts/run_train.py --model m2_monet_feature_baseline --all-folds --device cuda"
    log "  python3 scripts/run_train.py --model m3_transformer_baseline --all-folds --device cuda"
    log "  python3 scripts/run_train.py --model m4_multimodal_fusion --all-folds --device cuda"
    log "  for f in 0 1 2 3 4; do python3 scripts/build_graph.py --fold \$f --knn-device cuda; done"
    log "  python3 scripts/run_train.py --model m5_lgke_gnn --all-folds --device cuda"
  fi

  run_cmd python3 scripts/make_report_assets.py
  run_cmd python3 scripts/make_table4.py
  log "full workflow done"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      MODE="smoke"
      ;;
    --full)
      MODE="full"
      ;;
    --bundle-root)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Missing value after --bundle-root" >&2
        exit 1
      fi
      BUNDLE_ROOT="$1"
      ;;
    --m0-m4-root)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Missing value after --m0-m4-root" >&2
        exit 1
      fi
      M0_M4_ROOT="$1"
      ;;
    --skip-demo-assets)
      SKIP_DEMO_ASSETS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ "$MODE" == "full" ]]; then
  run_full
else
  run_smoke
fi
