# Revolab Malaysian ASR Benchmark

How well do speech recognition models actually transcribe Malaysian speech?

This repo runs any ASR model over 820 real Malaysian audio clips — phone calls,
parliament, TV drama, singing, street interviews — and reports word error rate,
character error rate and speed, broken down by domain.

Dataset: [`Revolab/ASR-Benchmark-Public`](https://huggingface.co/datasets/Revolab/ASR-Benchmark-Public)

## Results

820 samples, language `ms`. **Lower WER is better; higher RTFx is faster.**

| # | Model | WER | CER | RTFx |
|---:|---|---:|---:|---:|
| 1 | `scribe_v2` (ElevenLabs) | **4.79%** | 2.00% | 7.3 |
| 2 | `aisyah-1.0-pro` (Revolab) | 4.89% | **1.97%** | 15.7 |
| 3 | `gemini-2.5-pro` | 5.30% | 2.43% | 1.1 |
| 4 | `qwen-audio-3.0-asr-flash` | 7.22% | 3.15% | 13.0 |
| 5 | `ilmu-asr-v4.2` | 7.78% | 3.15% | 12.1 |
| 6 | `aisyah-1.0-flash` (Revolab) | 7.98% | 3.91% | **45.6** |
| 7 | `gemini-2.5-flash` | 9.05% | 4.66% | 1.4 |
| 8 | `universal-2` (AssemblyAI) | 14.86% | 8.91% | 1.8 |
| 9 | `universal-3-5-pro` (AssemblyAI) | 14.88% | 8.92% | 1.9 |
| 10 | `gemini-3.6-flash` | 15.01% | 9.31% | 1.0 |
| 11 | `Qwen/Qwen3-ASR-1.7B` | 15.26% | 7.28% | 21.0 |
| 12 | `openai/whisper-large-v3` | 15.62% | 10.18% | 3.2 |
| 13 | `Qwen/Qwen3-ASR-0.6B` | 20.73% | 8.82% | 12.8 |
| 14 | `nova-3` (Deepgram) | 25.63% | 15.35% | 18.0 |

Per-domain numbers, error breakdowns and rare-word WER come from the same command:

```bash
python scripts/summarize_results.py --results-dir results/public
```

Every figure above is regenerated from the transcript manifests committed in
`results/public/` — nothing is hand-entered.

## How it works

Two commands, one file in between:

```
   run_eval.py                    JSONL manifest                summarize_results.py
┌────────────────────┐      ┌──────────────────────────┐      ┌─────────────────────┐
│ load model once,   │ ───► │ one line per utterance:  │ ───► │ leaderboard table   │
│ transcribe, score  │      │ reference, prediction,   │      │ overall + per-domain│
│ each utterance     │      │ WER stats, timing        │      │ + optional CSV      │
└────────────────────┘      └──────────────────────────┘      └─────────────────────┘
```

The manifest is the unit of exchange: run a model today, summarise it next week,
and compare it against everyone else's runs by dropping the JSONL files into the
same directory.

## Quick start

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements/base.txt      # core
pip install -e .                          # makes `asr_benchmark` importable
pip install -r requirements/whisper.txt   # the backend you want to run
```

### 2. Authenticate with Hugging Face

The dataset is downloaded from the Hub:

```bash
huggingface-cli login          # or: export HF_TOKEN=hf_...
```

### 3. Run a model

Every model has a launcher that passes the right flags for you:

```bash
bash scripts/models/run_whisper_large_v3.sh 20   # 20 samples → results/smoke
bash scripts/models/run_whisper_large_v3.sh      # all 820    → results/public
```

A capped run is treated as a smoke test and writes to `results/smoke/`, so it can
never overwrite published results.

To sweep every model and build the leaderboard in one go:

```bash
bash scripts/run_malay_benchmark.sh          # all models → results/public
bash scripts/run_malay_benchmark.sh 50       # quick pass → results/smoke
```

Models whose API key isn't set are skipped, and one model erroring out doesn't
abort the rest — you get a ran / skipped / failed summary at the end.

### 4. Build the leaderboard

```bash
python scripts/summarize_results.py \
    --results-dir results/public \
    --csv results/public/leaderboard.csv
```

It reads *every* `.jsonl` in the directory and ranks the models it finds.

### Writing the command yourself

The launchers are thin wrappers around this:

```bash
python run_eval.py \
    --model-type whisper \
    --model-id openai/whisper-large-v3 \
    --dataset Revolab/ASR-Benchmark-Public \
    --splits train \
    --text-column text \
    --normalized-text-column normalized_text \
    --metadata-columns category \
    --language ms \
    --no-streaming \
    --max-samples 20 \
    --output-dir results/smoke
```

Drop `--max-samples` for the full run. Four of those flags are **required** —
the defaults target Common Voice, and omitting them here either fails outright or
silently reports the wrong number:

| Flag | What breaks without it |
|---|---|
| `--text-column text` | the default `sentence` column doesn't exist here → run fails |
| `--normalized-text-column normalized_text` | dual-reference scoring is disabled → WER comes out too high and isn't comparable to the table above |
| `--metadata-columns category` | no per-domain breakdown |
| `--no-streaming` | streaming can't decode this dataset's audio |

## Scoring

### Dual-reference WER

Each clip ships with two reference transcripts: the raw `text` and the
dataset-supplied `normalized_text`. Both are normalised the same way, WER is
computed against each, and **the lower one wins for that utterance**:

```
reference (raw)         : "Dr. Ahmad kata 20% je"
reference (normalized)  : "doktor ahmad kata dua puluh peratus je"
prediction              : "doktor ahmad kata dua puluh peratus je"
                          → WER 60% vs raw, 0% vs normalized → scores 0%
```

Without the audio in front of us we can't say which spelling a model *should*
have produced, so we don't penalise it for picking the other valid one. The
corpus WER aggregates the winning per-utterance counts, and the manifest records
which reference won (`ref_source`).

### Metrics

| Metric | Meaning |
|---|---|
| **WER** | word error rate, dual-reference as above. Lower is better |
| **CER** | character edit distance ÷ total reference characters |
| **Sub / Ins / Del** | the WER split into substitutions, insertions, deletions — useful for spotting a model that hallucinates (high Ins) vs. one that drops audio (high Del) |
| **RTFx** | audio duration ÷ transcription time. 10 = 10× faster than realtime. Includes API latency for hosted models |
| **RareWER** | WER over reference words outside the corpus's 500 most common. Only comparable within a single summarize run, since the "common" list depends on which manifests are present |

Normalisation depends on `--language`: `ms`/`id` use `MalayTextNormalizer`, `en`
uses `EnglishTextNormalizer`, anything else falls back to `BasicTextNormalizer`.

All of them strip `<tags>`, `[tags]`, `(tags)` and punctuation, and lowercase —
with no number or abbreviation expansion. `MalayTextNormalizer` additionally
folds common spelling variants to one canonical form (`okay/okey → ok`,
`tu → itu`, `ni/nih → ini`, `jugak → juga`, `takde → tiada`) and splits clitics
and particles, so a model isn't penalised for writing a valid alternative
spelling.

## Digging into the results

WER tells you *how much* a model gets wrong. These read the same manifests to
show *what* it gets wrong — all of them take `--results-dir`:

```bash
# Where the errors are: per-category WER, worst rows, error patterns
python scripts/error_analysis.py --model-id nova-3 --results-dir results/public

# Same for every model at once, as JSON
python scripts/error_analysis.py --all-models --results-dir results/public \
    --json-out results/public/error_analysis.json

# Which words get swapped for which
python scripts/compute_confusion.py --results-dir results/public

# WER bucketed by audio noise level, ranked by clean → noisy degradation
python scripts/noise_analysis.py --results-dir results/public
```

`noise_analysis.py` needs per-sample SNR tags. `results/public/noise_tags.json`
is committed, so it works out of the box; regenerate it (or tag a new dataset)
with:

```bash
python scripts/tag_noise.py --results-dir results/public
```

Noise robustness is where the leaderboard ordering changes — a model can look
strong on clean audio and collapse on noisy:

```
model                        clean      moderate    noisy     clean->noisy
gemini-2.5-pro           5.15%/600    4.13%/130   8.89%/90        +3.74pp
aisyah-1.0-flash         5.49%/600    5.68%/130  33.07%/90       +27.58pp
```

Finally, `scripts/analyze_models.py` asks an LLM to write per-model strengths and
weaknesses, each backed by a specific benchmark sample. It needs `GEMINI_API_KEY`
(or `--provider claude`).

## TODO

- **Better prompt for `gemini-3.6-flash`.** It refuses 9/820 short clips with
  *"no audio file was provided"* instead of transcribing, inflating its WER.
  Tune `_TRANSCRIBE_PROMPT` in `asr_benchmark/models/gemini_model.py`, then
  re-run the model.

## Dataset

820 samples in **one `train` split** — that's just the Hugging Face split name,
not a training set. The 12 domains are values of the `category` column, which is
why `--metadata-columns category` matters:

| `category` | N | What it is |
|---|---:|---|
| `fleurs` | 153 | FLEURS (ms), public benchmark |
| `commonvoice` | 141 | Common Voice (ms), public benchmark |
| `telephony` | 66 | Phone call audio |
| `drama` | 52 | TV drama |
| `news` | 52 | Broadcast news |
| `parliament` | 52 | Parliamentary proceedings |
| `podcast` | 52 | Podcast recordings |
| `short-inputs` | 52 | Short utterances |
| `read-speech` | 51 | Scripted reading |
| `singing` | 51 | Sung audio |
| `animation` | 50 | Animated / cartoon content |
| `street interview` | 48 | Spontaneous outdoor speech |

Score a single domain with `--category`, e.g. `--category telephony`.

## Supported models

| `--model-type` | Example `--model-id` | Extra deps | API key |
|---|---|---|---|
| `whisper` | `openai/whisper-large-v3` | `requirements/whisper.txt` | — |
| `qwen` | `Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ASR-0.6B` | `requirements/qwen.txt` | — |
| `gemini` | `gemini-2.5-flash`, `gemini-2.5-pro` | `requirements/gemini.txt` | `GEMINI_API_KEY` |
| `elevenlabs` | `scribe_v2` | `requirements/elevenlabs.txt` | `ELEVENLABS_API_KEY` |
| `assemblyai` | `universal-2`, `universal-3-pro`, `universal-3-5-pro` | `requirements/assemblyai.txt` | `ASSEMBLYAI_API_KEY` |
| `deepgram` | `nova-3` | `requirements/deepgram.txt` | `DEEPGRAM_API_KEY` |
| `qwen-audio` | `qwen-audio-3.0-asr-flash` | none | `DASHSCOPE_API_KEY` |
| `ilmu` | `ilmu-asr-v4.2` | `requirements/ilmu.txt` | `ILMU_API_KEY` |
| `revolab-api` | `aisyah-1.0-flash`, `aisyah-1.0-pro` | none | `REVOLAB_API_KEY` |

Copy `.env.example` to `.env` and fill in the keys you need — `run_eval.py` loads
it automatically when `python-dotenv` is installed. Each model has a launcher in
`scripts/models/`.

## Adding your own model

Full guide in [CONTRIBUTING.md](CONTRIBUTING.md). The short version:

1. Subclass `BaseASRModel` in `asr_benchmark/models/my_model.py`
2. Implement `_load_model()` and `transcribe_batch()`
3. Register it in `asr_benchmark/models/__init__.py` under `MODEL_REGISTRY`
4. Run the benchmark and include the output JSONL in your PR

```python
from asr_benchmark.models.base import BaseASRModel, TranscriptionResult

class MyModel(BaseASRModel):
    def _load_model(self) -> None:
        # load weights, processor, etc. — called once per process
        ...

    def transcribe_batch(self, audio_arrays, sample_rates, audio_lengths_s) -> list[TranscriptionResult]:
        results = []
        for audio, sr, length in zip(audio_arrays, sample_rates, audio_lengths_s):
            text, elapsed = self._timed_call(lambda: self.model(audio))
            results.append(TranscriptionResult(
                prediction=text,
                audio_length_s=length,
                transcription_time_s=elapsed,
            ))
        return results
```

## Repository layout

```
run_eval.py              # CLI entry-point — one model, one or more splits
asr_benchmark/
  runner.py              # Evaluator: loads the model once, scores every split
  models/
    base.py              # BaseASRModel + TranscriptionResult
    whisper_model.py  qwen_model.py  gemini_model.py  elevenlabs_model.py
    deepgram_model.py  assemblyai_model.py  qwen_audio_model.py  ...
  normalizer/
    basic.py             # BasicTextNormalizer
    malay.py             # Malay-specific normalisation
  utils/
    data.py              # HF dataset loader
    metrics.py           # WER / CER / rare-WER
    manifest.py          # JSONL read/write
scripts/
  models/                # one launcher per model
  run_malay_benchmark.sh # every model in sequence, then the leaderboard
  summarize_results.py   # manifests → leaderboard table + CSV
  error_analysis.py  analyze_models.py  noise_analysis.py  ...
results/public/          # committed manifests behind the table above
requirements/            # base.txt + one file per backend
```

## License

Evaluation code: [MIT](LICENSE).

Dataset annotations (reference transcripts): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
see [`Revolab/ASR-Benchmark-Public`](https://huggingface.co/datasets/Revolab/ASR-Benchmark-Public).
This covers the annotations only; the underlying audio retains the license of its
original source.
