# Revolab Malaysian ASR Benchmark

Open evaluation of speech recognition models on real Malaysian audio, using [`Revolab/ASR-Benchmark-Public`](https://huggingface.co/datasets/Revolab/ASR-Benchmark-Public).

## Dataset

Public benchmark: [`Revolab/ASR-Benchmark-Public`](https://huggingface.co/datasets/Revolab/ASR-Benchmark-Public) — 820 samples across 10 domains:

| Domain | Description |
|---|---|
| Telephony | Phone call audio |
| Short inputs | Short utterances |
| Cartoon | Animated content |
| Drama | TV drama |
| News | Broadcast news |
| Parliament | Parliamentary proceedings |
| Podcast | Podcast recordings |
| Read speech | Scripted reading |
| Singing | Sung audio |
| Street interview | Spontaneous outdoor speech |

Plus two standard public benchmarks: FLEURS (ms) and CommonVoice (ms).

## Evaluation methodology

- **WER** — dual-reference scoring: both raw and normalised references are scored per utterance; the lower-WER reference wins. Final corpus WER aggregates the per-utterance winners.
- **CER** — character Levenshtein edit distance summed over all utterances, divided by total reference character count.
- **Normalisation** — `BasicTextNormalizer` strips tags, punctuation, lowercases; no number or abbreviation expansion.
- **RTFx** — `audio_duration / transcription_time` (higher = faster). Measured end-to-end including API latency.

## Running the benchmark

```bash
# Install
pip install -r requirements/base.txt

# Smoke test (20 samples)
python run_eval.py \
    --model-type whisper \
    --model-id openai/whisper-large-v3 \
    --dataset Revolab/ASR-Benchmark-Public \
    --splits train \
    --language ms \
    --max-samples 20 \
    --output-dir results/smoke

# Full run
python run_eval.py \
    --model-type whisper \
    --model-id openai/whisper-large-v3 \
    --dataset Revolab/ASR-Benchmark-Public \
    --splits train \
    --language ms \
    --output-dir results/public
```

Model-specific dependencies:

```bash
pip install -r requirements/whisper.txt     # Whisper
pip install -r requirements/qwen.txt        # Qwen3-ASR
pip install -r requirements/gemini.txt      # Gemini
pip install -r requirements/elevenlabs.txt  # ElevenLabs Scribe
```

API keys for non-local models:

```bash
export GEMINI_API_KEY="..."
export ELEVENLABS_API_KEY="..."
```

## Supported models

| Key | Model | Notes |
|---|---|---|
| `whisper` | Whisper (any checkpoint) | HF transformers pipeline |
| `qwen` | Qwen3-ASR (0.6B, 1.7B) | HF transformers |
| `gemini` | Gemini 2.5 Flash / Pro | Google AI API |
| `elevenlabs` | Scribe v2 | ElevenLabs API |
| `assemblyai` | Universal-2 | AssemblyAI API |
| `deepgram` | Nova-3 | Deepgram API |

## Contributing a model

Want to add your model to the leaderboard? See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

**Short version:**

1. Subclass `BaseASRModel` in `asr_benchmark/models/my_model.py`
2. Implement `_load_model()` and `transcribe_batch()`
3. Register in `asr_benchmark/models/__init__.py`
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
    whisper_model.py
    qwen_model.py
    gemini_model.py
    elevenlabs_model.py
    ...
  normalizer/
    basic.py             # BasicTextNormalizer
    malay.py             # Malay-specific normalisation
  utils/
    data.py              # HF dataset loader
    metrics.py           # WER / CER
    manifest.py          # JSONL read/write
  runner.py              # Evaluator + EvalConfig
run_eval.py              # CLI entry-point
scripts/
  summarize_results.py
  run_malay_benchmark.sh
requirements/
  base.txt
  whisper.txt
  qwen.txt
  gemini.txt
  elevenlabs.txt
```
