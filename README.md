# Revolab Malaysian ASR Benchmark

Open evaluation of speech recognition models on real Malaysian audio, using [`Revolab/ASR-Benchmark-Public`](https://huggingface.co/datasets/Revolab/ASR-Benchmark-Public).

## Quick start

End-to-end: install, run one model, get numbers. Run everything from the repo root.

```bash
# 1. Install core deps + make `asr_benchmark` importable, then the model backend
pip install -r requirements/base.txt
pip install -e .
pip install -r requirements/whisper.txt

# 2. Hugging Face auth (needed to download the dataset)
huggingface-cli login          # or: export HF_TOKEN=hf_...

# 3. Smoke test — 20 samples
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

# 4. Full run (820 samples) — drop --max-samples, write to a fresh directory
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
    --output-dir results/public

# 5. Leaderboard — overall + per-domain WER/CER/RTFx across every manifest in the dir
python scripts/summarize_results.py \
    --results-dir results/public \
    --csv results/public/leaderboard.csv
```

Or use the ready-made per-model script, which passes all of the above for you:

```bash
bash scripts/models/run_whisper_large_v3.sh 20        # 20 samples
bash scripts/models/run_whisper_large_v3.sh           # full run
bash scripts/run_malay_benchmark.sh                   # every model + leaderboard
```

The scripts use `uv run python`; if you don't have `uv`, run the `python run_eval.py`
form above instead.

### Flags that matter

The four flags below are **not** optional for this dataset — the defaults target
Common Voice and will silently produce wrong numbers here:

| Flag | Why |
|---|---|
| `--text-column text` | default is `sentence`, which does not exist in this dataset |
| `--normalized-text-column normalized_text` | enables dual-reference scoring (see [methodology](#evaluation-methodology)); without it, WER is measured against the raw reference only and is not comparable to the leaderboard |
| `--metadata-columns category` | required for the per-domain breakdown |
| `--no-streaming` | this dataset ships torchcodec audio; streaming mode is not supported |

### What you get

`run_eval.py` prints a JSON summary per split and writes one manifest per
(model, dataset, split):

```
results/public/openai__whisper-large-v3__Revolab__ASR-Benchmark-Public__train.jsonl
```

Each line is one utterance with `reference`, `prediction`, both normalised
references, the winning `ref_source`, word/char edit counts, `audio_length_s`,
`transcription_time_s`, `rtfx`, and `category`.

`scripts/summarize_results.py` reads every `.jsonl` in a directory and prints:

```
  Model / Category                              N      WER      CER    Sub%    Ins%    Del%    RTFx   RareWER
-------------------------------------------------------------------------------------------------------------
  OVERALL                                     820   15.62%   10.18%   8.04%   3.56%   4.02%    3.16    17.79%
-------------------------------------------------------------------------------------------------------------
  animation                                    50    7.56%    3.65%   5.90%   0.61%   1.06%    2.39    14.75%
  commonvoice                                 141    5.47%    2.31%   2.28%   0.20%   2.99%    3.50     9.95%
  drama                                        52   12.96%    8.20%   8.97%   0.78%   3.21%    1.99    19.44%
  ...

── QUICK VIEW (sorted by WER) ──
  Model                                         Split               WER      CER    Sub%    Ins%    Del%    RTFx   RareWER
--------------------------------------------------------------------------------------------------------------------------
  scribe_v2                                     train             4.79%    2.00%   2.81%   0.55%   1.44%    7.27     6.05%
  openai/whisper-large-v3                       train            15.62%   10.18%   8.04%   3.56%   4.02%    3.16    17.79%
```

`--csv` additionally writes the same figures as a flat CSV (one row per model,
one column group per domain).

## Dataset

[`Revolab/ASR-Benchmark-Public`](https://huggingface.co/datasets/Revolab/ASR-Benchmark-Public)
— 820 samples in a **single `train` split**. Domains are not separate splits;
they are values of the `category` column, which is why `--metadata-columns category`
matters:

| `category` | N | Description |
|---|---|---|
| `telephony` | 66 | Phone call audio |
| `short-inputs` | 52 | Short utterances |
| `animation` | 50 | Animated / cartoon content |
| `drama` | 52 | TV drama |
| `news` | 52 | Broadcast news |
| `parliament` | 52 | Parliamentary proceedings |
| `podcast` | 52 | Podcast recordings |
| `read-speech` | 51 | Scripted reading |
| `singing` | 51 | Sung audio |
| `street interview` | 48 | Spontaneous outdoor speech |
| `fleurs` | 153 | FLEURS (ms) |
| `commonvoice` | 141 | Common Voice (ms) |

To score a single domain, use `--category` (e.g. `--category telephony`).

## Evaluation methodology

- **WER** — dual-reference scoring: both raw and normalised references are scored per utterance; the lower-WER reference wins. Final corpus WER aggregates the per-utterance winners.
- **CER** — character Levenshtein edit distance summed over all utterances, divided by total reference character count.
- **Normalisation** — `BasicTextNormalizer` strips tags, punctuation, lowercases; no number or abbreviation expansion.
- **RTFx** — `audio_duration / transcription_time` (higher = faster). Measured end-to-end including API latency.
- **RareWER** — WER restricted to reference words outside the 500 most frequent words across the corpus.

## Supported models

| `--model-type` | Example `--model-id` | Deps | Env var |
|---|---|---|---|
| `whisper` | `openai/whisper-large-v3` | `requirements/whisper.txt` | — |
| `qwen` | `Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ASR-0.6B` | `requirements/qwen.txt` | — |
| `gemini` | `gemini-2.5-flash`, `gemini-2.5-pro` | `requirements/gemini.txt` | `GEMINI_API_KEY` |
| `elevenlabs` | `scribe_v2` | `requirements/elevenlabs.txt` | `ELEVENLABS_API_KEY` |
| `assemblyai` | `universal-2`, `universal-3`, `universal-3-5-pro` | `requirements/assemblyai.txt` | `ASSEMBLYAI_API_KEY` |
| `deepgram` | `nova-3` | `requirements/deepgram.txt` | `DEEPGRAM_API_KEY` |
| `qwen-audio` | `qwen-audio-3.0-asr-flash` | base only | `DASHSCOPE_API_KEY` |
| `ilmu` | `ilmu-asr-v4.2` | base only | `ILMU_API_KEY` |
| `revolab-api` | `aisyah-1.0-flash`, `aisyah-1.0-pro` | base only | `REVOLAB_API_KEY` |

Copy `.env.example` to `.env` and fill in the keys you need — `run_eval.py` loads
it automatically if `python-dotenv` is installed. Each model also has a launcher
in `scripts/models/`.

## Contributing a model

Want to add your model to the leaderboard? See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

**Short version:**

1. Subclass `BaseASRModel` in `asr_benchmark/models/my_model.py`
2. Implement `_load_model()` and `transcribe_batch()`
3. Register in `asr_benchmark/models/__init__.py` under `MODEL_REGISTRY`
4. Run the benchmark, include the output JSONL in your PR

```python
from asr_benchmark.models.base import BaseASRModel, TranscriptionResult

class MyModel(BaseASRModel):
    def _load_model(self) -> None:
        # load weights, processor, etc.
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
asr_benchmark/
  models/
    base.py              # BaseASRModel + TranscriptionResult
    whisper_model.py  qwen_model.py  gemini_model.py  elevenlabs_model.py
    deepgram_model.py  assemblyai_model.py  qwen_audio_model.py  ...
  normalizer/
    basic.py             # BasicTextNormalizer
    malay.py             # Malay-specific normalisation
  utils/
    data.py              # HF dataset loader (handles torchcodec audio)
    metrics.py           # WER / CER / rare-WER
    manifest.py          # JSONL read/write
  runner.py              # Evaluator + ModelConfig / DatasetConfig
run_eval.py              # CLI entry-point
scripts/
  models/                # one launcher per model
  run_malay_benchmark.sh # all models, then the leaderboard
  summarize_results.py   # manifests → leaderboard table + CSV
  error_analysis.py  analyze_models.py  noise_analysis.py  ...
requirements/
  base.txt  whisper.txt  qwen.txt  gemini.txt
  elevenlabs.txt  deepgram.txt  assemblyai.txt
```

## License

The evaluation code in this repository is licensed under [MIT](LICENSE).

The benchmark dataset's annotations (reference transcripts) are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see
[`Revolab/ASR-Benchmark-Public`](https://huggingface.co/datasets/Revolab/ASR-Benchmark-Public)
on Hugging Face. This covers the annotations only; the underlying audio
retains the license of its original source.
