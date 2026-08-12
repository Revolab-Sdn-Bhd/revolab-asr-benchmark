#!/usr/bin/env bash
# Benchmark: Revolab/Malaysian-Qwen3-ASR-1.7B
# Usage: bash run_malaysian_qwen3.sh [max_samples] [public|private]
set -euo pipefail

MAX_SAMPLES="${1:-}"
BENCHMARK="${2:-public}"
MAX_SAMPLES_ARG="${MAX_SAMPLES:+--max-samples ${MAX_SAMPLES}}"

if [[ "${BENCHMARK}" == "private" ]]; then
    DATASET="Revolab/ASR-Benchmark-Private"
    OUTPUT_DIR="results/private"
else
    DATASET="Revolab/ASR-Benchmark-Public"
    OUTPUT_DIR="results/public"
fi

uv run python run_eval.py \
    --model-type qwen \
    --model-id knoveleng/polyglot-lion-1.7b-v1.5 \
    --dataset "${DATASET}" \
    --splits train \
    --text-column text \
    --normalized-text-column normalized_text \
    --metadata-columns category \
    --language ms \
    --batch-size 4 \
    --output-dir "${OUTPUT_DIR}" \
    --no-streaming \
    ${MAX_SAMPLES_ARG}
