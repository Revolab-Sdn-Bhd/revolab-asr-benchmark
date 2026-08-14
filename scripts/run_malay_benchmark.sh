#!/usr/bin/env bash
# Run all models for the Malay ASR benchmark.
#
# Usage:
#   bash scripts/run_malay_benchmark.sh                    # public, full run
#   bash scripts/run_malay_benchmark.sh 50                 # public, smoke test
#   bash scripts/run_malay_benchmark.sh "" private         # private, full run
#   bash scripts/run_malay_benchmark.sh 50 private         # private, smoke test

set -euo pipefail

# Always run from the repo root, wherever this script was invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

MAX_SAMPLES="${1:-}"
BENCHMARK="${2:-public}"

echo "============================================================"
echo " Benchmark: ${BENCHMARK^^}  |  Max samples: ${MAX_SAMPLES:-all}"
echo "============================================================"

bash "${SCRIPT_DIR}/models/run_revolab_aisyah_flash.sh"  "${MAX_SAMPLES}" "${BENCHMARK}"
bash "${SCRIPT_DIR}/models/run_qwen3_1.7b.sh"           "${MAX_SAMPLES}" "${BENCHMARK}"
bash "${SCRIPT_DIR}/models/run_qwen3_0.6b.sh"           "${MAX_SAMPLES}" "${BENCHMARK}"
bash "${SCRIPT_DIR}/models/run_whisper_large_v3.sh"      "${MAX_SAMPLES}" "${BENCHMARK}"
bash "${SCRIPT_DIR}/models/run_deepgram_nova3.sh"        "${MAX_SAMPLES}" "${BENCHMARK}"
bash "${SCRIPT_DIR}/models/run_elevenlabs.sh"            "${MAX_SAMPLES}" "${BENCHMARK}"
bash "${SCRIPT_DIR}/models/run_assemblyai.sh"           "${MAX_SAMPLES}" "${BENCHMARK}"

# Must match the OUTPUT_DIR the per-model scripts picked, or the leaderboard
# would be built from a different directory than the one just written to.
if [[ -n "${MAX_SAMPLES}" ]]; then
    OUTPUT_DIR="results/smoke"
else
    OUTPUT_DIR="results/${BENCHMARK}"
fi

echo ""
echo "============================================================"
echo " All models done. Generating leaderboard (${BENCHMARK})..."
echo "============================================================"
uv run python scripts/summarize_results.py \
    --results-dir "${OUTPUT_DIR}" \
    --csv "${OUTPUT_DIR}/leaderboard.csv"
