# ──────────────────────────────────────────────────────────────────────────────
# Quantization Benchmarking — benchmark container
#
# This image contains only the Python benchmark harness (main.py, agent.py,
# inference.py, etc.).  It does NOT contain the LLM weights or vLLM itself;
# vLLM runs in a separate container (vllm/vllm-openai) that is started on
# demand by run_benchmark.sh before each model is tested.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps: gcc is needed by some packages; curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        curl \
        docker-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy benchmark source
COPY agent.py \
     inference.py \
     main.py \
     metrics.py \
     run_single.py \
     tools.py \
     dataset.json \
     config.ini \
     run_benchmark.sh \
     ./

# Results are written to /results which is bind-mounted from the host.
# Creating it here ensures it exists even without a mount.
RUN mkdir -p /results

# The container starts an interactive shell by default.
# The benchmark is triggered manually with:
#   docker exec -it quant-bench bash -c "python main.py"
# OR by attaching and running the command inside.
CMD ["/bin/bash"]
