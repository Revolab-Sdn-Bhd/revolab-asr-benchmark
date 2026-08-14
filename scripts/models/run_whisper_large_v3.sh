#!/usr/bin/env bash
# Benchmark: openai/whisper-large-v3
# Usage: bash scripts/models/run_whisper_large_v3.sh [max_samples] [public|private]
set -euo pipefail

# Always run from the repo root, wherever this script was invoked from.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

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

# A capped run is a smoke test — keep it out of the published manifests.
if [[ -n "${MAX_SAMPLES}" ]]; then
    OUTPUT_DIR="results/smoke"
fi

python run_eval.py \
    --model-type whisper \
    --model-id openai/whisper-large-v3 \
    --dataset "${DATASET}" \
    --splits train \
    --text-column text \
    --normalized-text-column normalized_text \
    --metadata-columns category \
    --language ms \
    --output-dir "${OUTPUT_DIR}" \
    --no-streaming \
    ${MAX_SAMPLES_ARG}
