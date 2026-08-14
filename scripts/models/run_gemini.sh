#!/usr/bin/env bash
# Benchmark: Google Gemini (via google-genai SDK)
# Requires: GEMINI_API_KEY in env + pip install -r requirements/gemini.txt
# Usage: bash scripts/models/run_gemini.sh [model_id] [max_samples] [public|private]
#   e.g. bash run_gemini.sh gemini-2.5-flash
#        bash run_gemini.sh gemini-2.5-pro 50 private
set -euo pipefail

# Always run from the repo root, wherever this script was invoked from.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL_ID="${1:-gemini-2.5-flash}"
MAX_SAMPLES="${2:-}"
BENCHMARK="${3:-public}"
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

uv run python run_eval.py \
    --model-type gemini \
    --model-id "${MODEL_ID}" \
    --dataset "${DATASET}" \
    --splits train \
    --text-column text \
    --normalized-text-column normalized_text \
    --metadata-columns category \
    --language ms \
    --output-dir "${OUTPUT_DIR}" \
    --no-streaming \
    ${MAX_SAMPLES_ARG}
