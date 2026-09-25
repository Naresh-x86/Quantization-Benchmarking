#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_benchmark.sh — Manual trigger for the quantization benchmark suite
#
# Execute this script to run the benchmarks:
#
#   docker exec -it quant-bench bash /app/run_benchmark.sh
#
# main.py handles:
#  - Discovering enabled agents and models from config.ini
#  - Launching the vLLM server container per model on demand
#  - Mounting local model directories and the host HF cache
#  - Running benchmark trials with accurate VRAM measurements
#  - Piping all output to <results>/benchmark_<timestamp>/output.log
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

python3 /app/main.py "$@"
