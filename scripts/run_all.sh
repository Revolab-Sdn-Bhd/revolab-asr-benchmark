
#!/usr/bin/env bash
# Run the full benchmark suite across all configured (model, dataset) pairs.
# Edit the arrays below to match your target models and datasets.
# Usage: bash scripts/run_all.sh [--max-samples N]

set -euo pipefail

MAX_SAMPLES="${1:-}"
MAX_SAMPLES_ARG=""
if [[ -n "${MAX_SAMPLES}" ]]; then
    MAX_SAMPLES_ARG="--max-samples ${MAX_SAMPLES}"
fi

DATASET="mozilla-foundation/common_voice_13_0"
SUBSET="en"
SPLIT="test"
OUTPUT_DIR="results"

declare -A MODELS=(
    ["whisper"]="openai/whisper-large-v3"
    ["qwen"]="Qwen/Qwen2-Audio-7B-Instruct"
    ["zipformer"]="zipformer-en-2023-06-21"
    ["gemini"]="gemini-1.5-flash"
    ["elevenlabs"]="scribe_v1"
)

for MODEL_TYPE in "${!MODELS[@]}"; do
    MODEL_ID="${MODELS[$MODEL_TYPE]}"
    echo "============================================================"
    echo "Running: ${MODEL_TYPE} / ${MODEL_ID}"
    echo "============================================================"
    python run_eval.py \
        --model-type "${MODEL_TYPE}" \
        --model-id "${MODEL_ID}" \
        --dataset "${DATASET}" \
        --subset "${SUBSET}" \
        --split "${SPLIT}" \
        --output-dir "${OUTPUT_DIR}" \
        ${MAX_SAMPLES_ARG}
done

echo ""
echo "All done. Manifests written to: ${OUTPUT_DIR}/"
python scripts/summarize_results.py --results-dir "${OUTPUT_DIR}"
