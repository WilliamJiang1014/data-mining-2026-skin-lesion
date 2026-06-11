#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APP="src/skin_lesion_risk/demo/app.py"

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx skin; then
  exec conda run --no-capture-output -n skin streamlit run "$APP" "$@"
fi

if command -v streamlit >/dev/null 2>&1; then
  exec streamlit run "$APP" "$@"
fi

cat <<'EOF' >&2
未找到 streamlit。请先激活环境：

  conda activate skin
  bash scripts/run_demo.sh

或创建环境：

  conda env create -f environment.yml
  conda activate skin
EOF
exit 1
