#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_benchmark.sh — Manual trigger for the quantization benchmark suite
#
# Run this script INSIDE the benchmark container to execute a full benchmark:
#
#   bash /app/run_benchmark.sh
#
# Or from the host while the container is running:
#
#   docker exec -it quant-bench bash /app/run_benchmark.sh
#
# What this script does
# ─────────────────────
#  1. Reads config.ini to discover which models are enabled and inference
#     settings (gpu_memory_utilization, max_model_len, etc.).
#  2. For every enabled model it:
#       a. Starts a vLLM server (vllm/vllm-openai container) for that model.
#       b. Waits until the server is ready.
#       c. Runs main.py to benchmark all agents against that model.
#       d. Stops the vLLM server.
#  3. All benchmark output is tee'd by main.py into
#     /results/benchmark_<timestamp>/output.log automatically.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CONFIG="${BENCHMARK_CONFIG:-/app/config.ini}"
VLLM_PORT=8000
VLLM_CONTAINER_NAME="vllm-server"

# ── Helper: read a value from config.ini ─────────────────────────────────────
ini_get() {
    local section="$1" key="$2" default="${3:-}"
    python3 - "$CONFIG" "$section" "$key" "$default" <<'PYEOF'
import sys, configparser
cfg = configparser.ConfigParser()
cfg.optionxform = str
cfg.read(sys.argv[1])
section, key, default = sys.argv[2], sys.argv[3], sys.argv[4]
print(cfg.get(section, key, fallback=default).strip())
PYEOF
}

# ── Helper: collect enabled models from a section ────────────────────────────
enabled_models() {
    local section="$1" dir_key="$2"
    python3 - "$CONFIG" "$section" "$dir_key" <<'PYEOF'
import sys, configparser, os
cfg = configparser.ConfigParser()
cfg.optionxform = str
cfg.read(sys.argv[1])
section, dir_key = sys.argv[2], sys.argv[3]
if section not in cfg:
    sys.exit(0)
base_dir = cfg[section].get(dir_key, "").strip()
for key, val in cfg[section].items():
    if key == dir_key:
        continue
    if val.strip().lower() == "true":
        full_path = os.path.join(base_dir, key) if base_dir else key
        print(f"{key}|{full_path}")
PYEOF
}

# ── Read inference settings ───────────────────────────────────────────────────
GPU_MEM_UTIL=$(ini_get INFERENCE gpu_memory_utilization "0.05")
MAX_MODEL_LEN=$(ini_get INFERENCE max_model_len "4096")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        Quantization Benchmark — Manual Run                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Config                : $CONFIG"
echo "vLLM port             : $VLLM_PORT"
echo "GPU memory util       : $GPU_MEM_UTIL  (keeps VRAM metrics accurate)"
echo "Max model len (tokens): $MAX_MODEL_LEN"
echo ""

# ── Collect all enabled models (Qwen then Llama) ─────────────────────────────
mapfile -t QWEN_MODELS  < <(enabled_models QWEN_MODELS  qwen_models_dir)
mapfile -t LLAMA_MODELS < <(enabled_models LLAMA_MODELS llama_models_dir)
ALL_MODELS=("${QWEN_MODELS[@]}" "${LLAMA_MODELS[@]}")

if [[ ${#ALL_MODELS[@]} -eq 0 ]]; then
    echo "No models enabled in config.ini. Set at least one model to True."
    exit 1
fi

echo "Enabled models (${#ALL_MODELS[@]}):"
for entry in "${ALL_MODELS[@]}"; do
    model_name="${entry%%|*}"
    echo "  • $model_name"
done
echo ""

# ── Benchmark each model ──────────────────────────────────────────────────────
for entry in "${ALL_MODELS[@]}"; do
    MODEL_NAME="${entry%%|*}"
    MODEL_PATH="${entry##*|}"

    echo "────────────────────────────────────────────────────────────────"
    echo "  Model : $MODEL_NAME"
    echo "  Path  : $MODEL_PATH"
    echo "────────────────────────────────────────────────────────────────"

    # ── Start vLLM server ────────────────────────────────────────────────────
    echo "  [1/3] Starting vLLM server..."
    docker run -d \
        --name "$VLLM_CONTAINER_NAME" \
        --gpus all \
        --network host \
        -v /models:/models:ro \
        --ipc=host \
        vllm/vllm-openai:latest \
            --model "$MODEL_PATH" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --port "$VLLM_PORT" \
            --disable-log-requests

    # ── Wait for readiness ───────────────────────────────────────────────────
    echo "  [2/3] Waiting for vLLM to be ready..."
    MAX_WAIT=300   # 5 minutes
    ELAPSED=0
    until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        if [[ $ELAPSED -ge $MAX_WAIT ]]; then
            echo "  ✗ vLLM did not start within ${MAX_WAIT}s. Dumping logs:"
            docker logs "$VLLM_CONTAINER_NAME" | tail -40
            docker rm -f "$VLLM_CONTAINER_NAME" || true
            exit 1
        fi
        echo "    still waiting… (${ELAPSED}s)"
    done
    echo "  ✓ vLLM ready."

    # ── Run benchmarks for this model ────────────────────────────────────────
    echo "  [3/3] Running benchmarks..."
    export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export BENCHMARK_CONFIG="$CONFIG"

    python3 /app/main.py || true   # 'true' so one model failure doesn't abort the whole run

    # ── Tear down vLLM ───────────────────────────────────────────────────────
    echo "  Stopping vLLM server..."
    docker rm -f "$VLLM_CONTAINER_NAME" || true
    # Give the GPU a moment to fully free VRAM before the next model
    sleep 5
    echo ""
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  All models benchmarked.  Results in /results              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
