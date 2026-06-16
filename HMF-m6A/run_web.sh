#!/bin/bash
# m6A HMF Web 应用启动脚本
# 用法: bash run_web.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MPLCONFIGDIR="/tmp/matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"

PYTHON="/home/dnt/anaconda3/envs/m6A/bin/python"

echo "========================================="
echo "  m6A Methylation Predictor HMF Web"
echo "========================================="
echo "  Python: $PYTHON"
echo "  Working dir: $SCRIPT_DIR"
echo ""

$PYTHON -m streamlit run app.py --server.port 8501 --server.headless true
