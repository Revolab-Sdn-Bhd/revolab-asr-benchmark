#!/usr/bin/env bash
# Run every model in the benchmark, then build the leaderboard.
#
# Usage:
#   bash scripts/run_malay_benchmark.sh                    # public, full run
#   bash scripts/run_malay_benchmark.sh 50                 # smoke test → results/smoke
#   bash scripts/run_malay_benchmark.sh "" private         # private, full run
#   bash scripts/run_malay_benchmark.sh 50 private         # private, smoke test
#
# Models whose API key is missing are skipped; a model that errors out is
# reported at the end instead of aborting the whole sweep.

# Deliberately no -e: one model failing must not kill the remaining models.
set -uo pipefail

# Always run from the repo root, wherever this script was invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

MAX_SAMPLES="${1:-}"
BENCHMARK="${2:-public}"

# Must match the OUTPUT_DIR the per-model launchers pick, or the leaderboard
# would be built from a different directory than the one just written to.
if [[ -n "${MAX_SAMPLES}" ]]; then
    OUTPUT_DIR="results/smoke"
else
    OUTPUT_DIR="results/${BENCHMARK}"
fi

# Load .env so the API-key checks below see the same values run_eval.py will.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

RAN=()
SKIPPED=()
FAILED=()

# run_model <label> <api_key_var|-> <script> [args...]
run_model() {
    local label="$1" key_var="$2" script="$3"
    shift 3

    if [[ "${key_var}" != "-" && -z "${!key_var:-}" ]]; then
        echo "-- skipping ${label}: ${key_var} not set"
        SKIPPED+=("${label} (no ${key_var})")
        return 0
    fi

    echo ""
    echo "============================================================"
    echo " ${label}"
    echo "============================================================"
    if bash "${SCRIPT_DIR}/${script}" "$@"; then
        RAN+=("${label}")
    else
        echo "!! ${label} failed — continuing with the remaining models"
        FAILED+=("${label}")
    fi
}

echo "============================================================"
echo " Benchmark: ${BENCHMARK^^}  |  Max samples: ${MAX_SAMPLES:-all}"
echo " Output:    ${OUTPUT_DIR}"
echo "============================================================"

# --- Hosted APIs -----------------------------------------------------------
# run_gemini.sh takes [model_id] first; every other launcher takes
# [max_samples] [public|private].
run_model "aisyah-1.0-flash"         REVOLAB_API_KEY    models/run_revolab_aisyah_flash.sh        "${MAX_SAMPLES}" "${BENCHMARK}"
run_model "aisyah-1.0-pro"           REVOLAB_API_KEY    models/run_revolab_aisyah_pro.sh          "${MAX_SAMPLES}" "${BENCHMARK}"
run_model "gemini-2.5-flash"         GEMINI_API_KEY     models/run_gemini.sh     gemini-2.5-flash "${MAX_SAMPLES}" "${BENCHMARK}"
run_model "gemini-2.5-pro"           GEMINI_API_KEY     models/run_gemini.sh     gemini-2.5-pro   "${MAX_SAMPLES}" "${BENCHMARK}"
run_model "gemini-3.6-flash"         GEMINI_API_KEY     models/run_gemini.sh     gemini-3.6-flash "${MAX_SAMPLES}" "${BENCHMARK}"
run_model "scribe_v2"                ELEVENLABS_API_KEY models/run_elevenlabs.sh                  "${MAX_SAMPLES}" "${BENCHMARK}"
run_model "nova-3"                   DEEPGRAM_API_KEY   models/run_deepgram_nova3.sh              "${MAX_SAMPLES}" "${BENCHMARK}"
run_model "universal-2"              ASSEMBLYAI_API_KEY models/run_assemblyai.sh                  "${MAX_SAMPLES}" "${BENCHMARK}"
run_model "universal-3"              ASSEMBLYAI_API_KEY models/run_assemblyai_universal_3.sh      "${MAX_SAMPLES}" "${BENCHMARK}"
run_model "universal-3-5-pro"        ASSEMBLYAI_API_KEY models/run_assemblyai_universal_3_5_pro.sh "${MAX_SAMPLES}" "${BENCHMARK}"
run_model "ilmu-asr-v4.2"            ILMU_API_KEY       models/run_ilmu_asr.sh                    "${MAX_SAMPLES}" "${BENCHMARK}"
run_model "qwen-audio-3.0-asr-flash" DASHSCOPE_API_KEY  models/run_qwen_audio.sh                  "${MAX_SAMPLES}" "${BENCHMARK}"

# --- Local GPU models (downloads weights on first run) ---------------------
run_model "Qwen3-ASR-1.7B"           -                  models/run_qwen3_1.7b.sh                  "${MAX_SAMPLES}" "${BENCHMARK}"
run_model "Qwen3-ASR-0.6B"           -                  models/run_qwen3_0.6b.sh                  "${MAX_SAMPLES}" "${BENCHMARK}"
run_model "whisper-large-v3"         -                  models/run_whisper_large_v3.sh            "${MAX_SAMPLES}" "${BENCHMARK}"
run_model "polyglot-lion-1.7b-v1.5"  -                  models/run_polyglot_asr.sh                "${MAX_SAMPLES}" "${BENCHMARK}"

echo ""
echo "============================================================"
echo " Sweep finished — ran ${#RAN[@]}, skipped ${#SKIPPED[@]}, failed ${#FAILED[@]}"
echo "============================================================"
for m in "${SKIPPED[@]}"; do echo "  skipped: ${m}"; done
for m in "${FAILED[@]}"; do echo "  FAILED : ${m}"; done

if [[ ${#RAN[@]} -eq 0 ]]; then
    echo ""
    echo "No model produced results — not touching the leaderboard."
    exit 1
fi

echo ""
echo "============================================================"
echo " Generating leaderboard from ${OUTPUT_DIR}"
echo "============================================================"
uv run python scripts/summarize_results.py \
    --results-dir "${OUTPUT_DIR}" \
    --csv "${OUTPUT_DIR}/leaderboard.csv"

[[ ${#FAILED[@]} -eq 0 ]]
