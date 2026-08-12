#!/usr/bin/env bash
# Benchmark: Deepgram Nova-3
# Requires: DEEPGRAM_API_KEY env var + pip install -r requirements/deepgram.txt
# Usage: bash run_deepgram_nova3.sh [max_samples] [public|private]
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
    --model-type deepgram \
    --model-id nova-3 \
    --dataset "${DATASET}" \
    --splits train \
    --text-column text \
    --normalized-text-column normalized_text \
    --metadata-columns category \
    --language ms \
    --output-dir "${OUTPUT_DIR}" \
    --no-streaming \
    ${MAX_SAMPLES_ARG}
