# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The repo has its own venv. Activate it before running anything — the launcher
scripts call `python`, not an absolute interpreter path:

```bash
source .venv/bin/activate
```

To rebuild it from scratch:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements/base.txt -e .
pip install -r requirements/whisper.txt        # plus whichever backends you need
```

Dependencies are deliberately **unpinned** — install the latest. Two traps worth
remembering:

- `torchcodec` imports `torch` but doesn't declare it, so `torch` sits in
  `base.txt` even for API-only backends. It also needs `ffmpeg` on the system.
- `uv run python` does **not** work here: with no `pyproject.toml`, uv has no
  project to resolve and silently falls back to a bare interpreter with none of
  the dependencies installed.

Backend deps live in `requirements/<backend>.txt`, mirrored by the `BACKENDS`
dict in `setup.py`. Backends import their SDK lazily inside `_load_model()`, so
a missing dep only surfaces at run time — check the import when adding one.

API keys go in `.env` (copy `.env.example`); `run_eval.py` loads it via
`python-dotenv`.

## Commands

```bash
# One model, one or more splits
python run_eval.py \
    --model-type qwen \
    --model-id Qwen/Qwen3-ASR-1.7B \
    --dataset Revolab/ASR-Benchmark-Public \
    --splits train \
    --text-column text \
    --normalized-text-column normalized_text \
    --metadata-columns category \
    --language ms \
    --no-streaming \
    --output-dir results/public

# A specific model via its launcher: [max_samples] [public|private]
bash scripts/models/run_qwen3_1.7b.sh          # full run  → results/public
bash scripts/models/run_qwen3_1.7b.sh 50       # smoke     → results/smoke

# Every model, then the leaderboard
bash scripts/run_malay_benchmark.sh            # full
bash scripts/run_malay_benchmark.sh 50         # smoke

# Leaderboard from a directory of manifests
python scripts/summarize_results.py \
    --results-dir results/public \
    --csv results/public/leaderboard.csv

# Analysis (all read the same manifests)
python scripts/error_analysis.py --model-id nova-3 --results-dir results/public
python scripts/compute_confusion.py --results-dir results/public
python scripts/noise_analysis.py --results-dir results/public      # needs noise_tags.json
python scripts/tag_noise.py --results-dir results/public           # regenerates it
python scripts/analyze_models.py --results-dirs results/public     # LLM, needs GEMINI_API_KEY
```

`scripts/summarize_results.py` imports `asr_benchmark`, so the package must be
installed (`pip install -e .`) — running it from `scripts/` alone won't work.

## Architecture

### Model-oriented evaluation

The model is loaded **once** in `runner.py:Evaluator`, then all requested splits
run sequentially before the process exits. This avoids repeated GPU memory
allocation for large models. `ModelConfig` carries the model settings;
`DatasetConfig` carries dataset + split list.

Rows are batched to `--batch-size` (default 1) and flushed through
`transcribe_batch()`. Results with `skipped=True` are dropped, not scored.

### Dataset shape

`Revolab/ASR-Benchmark-Public` is 820 samples in a **single `train` split**. The
12 domains are values of the `category` column, not separate splits — hence
`--metadata-columns category`, and `--category <name>` to score just one.
`Revolab/ASR-Benchmark-Private` is the same shape.

The four flags `--text-column text`, `--normalized-text-column normalized_text`,
`--metadata-columns category` and `--no-streaming` are mandatory for this
dataset; the CLI defaults target Common Voice and will fail or silently
mis-score.

### Dual-reference scoring

Each row has two references: the raw `text` column and the dataset-supplied
`normalized_text` column. Both are normalized with the language's normalizer,
WER is computed against each per row, and the lower-WER reference wins for that
row (ties go to the dataset reference). Final corpus WER/CER/Sub/Ins/Del are
aggregated from the per-row winners. This is intentional: we don't know which
reference the model output is closer to without hearing the audio.

The normalizer is chosen by `--language` in `runner.py`: `en` →
`EnglishTextNormalizer`, **`ms`/`id` → `MalayTextNormalizer`**, everything else →
`BasicTextNormalizer`. `MalayTextNormalizer` extends the basic one (which strips
`<tag>` `[tag]` `(tag)` and punctuation, lowercases, and does no number or
abbreviation expansion) with canonical-variant folding — `okay/okey → ok`,
`tu → itu`, `ni/nih → ini`, `jugak → juga`, `takde → tiada` — plus clitic and
particle splitting.

Key manifest fields per record: `reference`, `reference_normalized_ours`,
`reference_normalized_dataset`, `reference_normalized_final`, `ref_source`
(`"ours"` or `"dataset"`), `prediction`, `prediction_normalized`, `wer_ours`,
`wer_dataset`, word-level edit stats, `audio_length_s`, `transcription_time_s`,
`rtfx`, `category`, `model_id`, `split`.

RTFx is measured end to end inside the backend, so it includes API latency for
hosted models.

### Adding a new model

1. Create `asr_benchmark/models/my_model.py` subclassing `BaseASRModel`
2. Implement `_load_model(self) -> None` and `transcribe_batch(self, audio_arrays, sample_rates, audio_lengths_s) -> list[TranscriptionResult]`
3. Register in `asr_benchmark/models/__init__.py` under `MODEL_REGISTRY`
4. Add its deps to `requirements/<name>.txt` and the `BACKENDS` dict in `setup.py`
5. Add a launcher in `scripts/models/` and a line in `scripts/run_malay_benchmark.sh`

`BaseASRModel.__init__` sets `self.model_id`, `self.language`, `self.kwargs`.
Use `self._timed_call(fn, *args)` for timing. `TranscriptionResult` has
`prediction`, `audio_length_s`, `transcription_time_s`, `metadata`, `skipped`,
and an `rtfx` property.

### Launcher scripts

Each `scripts/models/run_*.sh` takes `[max_samples] [public|private]` and cds to
the repo root first, so it can be invoked from anywhere. **A capped run writes to
`results/smoke/` instead of `results/public/`** so smoke tests can't overwrite
published manifests; `results/smoke/` is gitignored.

`run_gemini.sh` is the exception: its first argument is `model_id`, so it takes
`[model_id] [max_samples] [public|private]`.

`scripts/run_malay_benchmark.sh` runs every launcher. It deliberately does *not*
use `set -e`: a model whose API key is unset is skipped by name, a model that
errors is recorded and the sweep continues, and it ends with a ran/skipped/failed
summary plus a non-zero exit if anything failed. It picks the same output
directory the launchers picked before summarizing.

### Audio loading (`asr_benchmark/utils/data.py`)

The Revolab datasets yield `torchcodec.AudioDecoder` objects (not standard HF
`{"array": ..., "sampling_rate": ...}` dicts). `_decode_audio()` detects the type
and dispatches accordingly. When adding support for new datasets, check which
format the audio column uses.

### Manifest JSONL

`utils/manifest.py` provides `write_manifest(path, records)` and
`read_manifest(path)`. Files are named
`{model_id_slug}__{dataset_slug}__{split}.jsonl` (slashes become `__`) and land
in `--output-dir`. Every downstream tool — the leaderboard and all the analysis
scripts — reads these files and nothing else, so a manifest is the unit of
exchange between runs.

`noise_tags.json` (written by `tag_noise.py`, consumed by `noise_analysis.py`) is
keyed by **reference text**, so it cannot distinguish two clips that share a
transcript, and tags from a different benchmark will silently sit unused in the
file.
