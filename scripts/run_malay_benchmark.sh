#!/usr/bin/env bash
# Run all models for the Malay ASR benchmark.
#
# Usage:
#   bash scripts/run_malay_benchmark.sh                    # public, full run
#   bash scripts/run_malay_benchmark.sh 50                 # public, smoke test
#   bash scripts/run_malay_benchmark.sh "" private         # private, full run
#   bash scripts/run_malay_benchmark.sh 50 private         # private, smoke test

set -euo pipefail

MAX_SAMPLES="${1:-}"
BENCHMARK="${2:-public}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " Benchmark: ${BENCHMARK^^}  |  Max samples: ${MAX_SAMPLES:-all}"
echo "============================================================"

bash "${SCRIPT_DIR}/models/run_revolab_aisyah_flash.sh"  "${MAX_SAMPLES}" "${BENCHMARK}"
bash "${SCRIPT_DIR}/models/run_qwen3_1.7b.sh"           "${MAX_SAMPLES}" "${BENCHMARK}"
bash "${SCRIPT_DIR}/models/run_qwen3_0.6b.sh"           "${MAX_SAMPLES}" "${BENCHMARK}"
bash "${SCRIPT_DIR}/models/run_revolab_zipformer_ms.sh"  "${MAX_SAMPLES}" "${BENCHMARK}"
bash "${SCRIPT_DIR}/models/run_whisper_large_v3.sh"      "${MAX_SAMPLES}" "${BENCHMARK}"
bash "${SCRIPT_DIR}/models/run_deepgram_nova3.sh"        "${MAX_SAMPLES}" "${BENCHMARK}"
bash "${SCRIPT_DIR}/models/run_elevenlabs.sh"            "${MAX_SAMPLES}" "${BENCHMARK}"
bash "${SCRIPT_DIR}/models/run_assemblyai.sh"           "${MAX_SAMPLES}" "${BENCHMARK}"

OUTPUT_DIR="results/${BENCHMARK}"

echo ""
echo "============================================================"
echo " All models done. Generating leaderboard (${BENCHMARK})..."
echo "============================================================"
uv run python scripts/summarize_results.py \
    --results-dir "${OUTPUT_DIR}" \
    --csv "${OUTPUT_DIR}/leaderboard.csv"
