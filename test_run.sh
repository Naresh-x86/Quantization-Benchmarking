#!/bin/bash

echo "Creating dummy models directory for testing..."
mkdir -p ./test_models/Qwen2.5-0.5B-Instruct_FP16_Baseline

echo "Downloading tiny Qwen model for pipeline testing (requires huggingface_hub)..."
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct', local_dir='./test_models/Qwen2.5-0.5B-Instruct_FP16_Baseline')
"

echo "Running benchmark test..."
python3 main.py --models_dir ./test_models --dataset dataset.json --output test_results.csv

echo "Test complete. Check test_results.csv for output."
