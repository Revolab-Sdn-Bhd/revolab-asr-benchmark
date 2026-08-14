
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run evaluation for one model across one or more splits
uv run python run_eval.py \
    --model-type qwen \
    --model-id Qwen/Qwen3-ASR-1.7B \
    --dataset Revolab/ASR-Benchmark-test \
    --splits train short_inputs \
    --text-column text \
    --normalized-text-column normalized_text \
    --metadata-columns category \
    --language ms \
    --no-streaming \
    --output-dir results/malay-benchmark

# Smoke test with limited samples
uv run python run_eval.py --model-type whisper --model-id openai/whisper-large-v3 \
    --dataset Revolab/ASR-Benchmark-test --splits train --max-samples 20

# Run a specific model via its script
bash scripts/models/run_qwen3_1.7b.sh          # full run
bash scripts/models/run_qwen3_1.7b.sh 50       # 50 samples (smoke test)

# Run all models sequentially (generates leaderboard at the end)
bash scripts/run_malay_benchmark.sh            # full
bash scripts/run_malay_benchmark.sh 50         # smoke test

# Summarize results into a leaderboard table
uv run python scripts/summarize_results.py \
    --results-dir results/malay-benchmark \
    --csv results/malay-benchmark/leaderboard.csv

# Install model-specific dependencies
pip install -r requirements/qwen.txt
# Options: base.txt  whisper.txt  qwen.txt  gemini.txt  elevenlabs.txt  deepgram.txt  assemblyai.txt
```

API keys for non-local models: `export GEMINI_API_KEY=...` / `export ELEVENLABS_API_KEY=...`

# Run the Benchmark Explorer web app
pip install -r explorer/requirements.txt
python explorer/server.py            # serves at http://localhost:7860
# or: uvicorn explorer.server:app --reload --port 7860

## Architecture

### Model-oriented evaluation

The model is loaded **once** in `runner.py:Evaluator`, then all requested splits run sequentially before the process exits. This avoids repeated GPU memory allocation for large models. `ModelConfig` carries the model settings; `DatasetConfig` carries dataset + split list.

### Dual-reference scoring

Each dataset row has two references: the raw `text` column and the dataset-supplied `normalized_text` column. Both are normalized with `BasicTextNormalizer` (strips `<tag>` `[tag]` `(tag)`, punctuation, lowercases — no number/abbreviation expansion). WER is computed against both per row; the lower-WER reference wins for that row. Final corpus WER/CER/Sub/Ins/Del are computed from the aggregated per-row winners. This is intentional: we don't know which reference the model output is closer to without seeing the audio.

Key manifest fields per record: `reference`, `reference_normalized_ours`, `reference_normalized_dataset`, `reference_normalized_final`, `ref_source` (`"ours"` or `"dataset"`), `prediction`, `prediction_normalized`, `wer_ours`, `wer_dataset`, word-level edit stats, `audio_length_s`, `transcription_time_s`, `rtfx`, `category`, `model_id`, `split`.

### Adding a new model

1. Create `asr_benchmark/models/my_model.py` subclassing `BaseASRModel`
2. Implement `_load_model(self) -> None` and `transcribe_batch(self, audio_arrays, sample_rates, audio_lengths_s) -> list[TranscriptionResult]`
3. Register in `asr_benchmark/models/__init__.py` under `MODEL_REGISTRY`

`BaseASRModel.__init__` sets `self.model_id`, `self.language`, `self.kwargs`. Use `self._timed_call(fn, *args)` for timing. `TranscriptionResult` has fields `prediction`, `audio_length_s`, `transcription_time_s`.

### Audio loading (`asr_benchmark/utils/data.py`)

`Revolab/ASR-Benchmark-test` yields `torchcodec.AudioDecoder` objects (not standard HF `{"array": ..., "sampling_rate": ...}` dicts). `_decode_audio()` detects the type and dispatches accordingly. When adding support for new datasets, check which format the audio column uses.

### Manifest JSONL

`utils/manifest.py` provides `write_manifest(path, records)` and `read_manifest(path)`. Files are named `{model_id_slug}_{dataset_slug}_{split}.jsonl` and land in `--output-dir`. `summarize_results.py` reads all `.jsonl` files in a directory and aggregates them into a leaderboard.

### Benchmark Explorer (`explorer/`)

FastAPI + vanilla JS/HTML single-page app. `explorer/server.py` reads all JSONL manifests from `results/malay-benchmark/` at startup and exposes:
- `GET /api/meta` — splits, categories, model IDs, sample count
- `GET /api/leaderboard` — per-model WER/CER/RTFx with per-category breakdown
- `GET /api/samples?split=&category=` — filtered sample list
- `GET /api/sample?split=&ref=` — full sample with all model predictions (word-level stats)
- `GET /api/audio?split=&ref=` — streams WAV audio from HF dataset (lazy-loaded per split)

Audio is loaded on demand from `Revolab/ASR-Benchmark-test` using `_decode_audio()` (same torchcodec handling as eval). First access per split downloads + indexes the dataset; subsequent accesses use the in-memory cache. Audio requires HF access (`HF_TOKEN` env var or `huggingface-cli login`).
