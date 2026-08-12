# Contributing a Model

We welcome contributions of new ASR models to the Revolab Malaysian ASR Benchmark. Follow the steps below to add your model and open a pull request.

## Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- Access to the benchmark dataset: `Revolab/ASR-Benchmark-Public` on Hugging Face

## Steps

### 1. Fork and clone

```bash
git clone https://github.com/aisyah-revo/revolab-asr-benchmark.git
cd revolab-asr-benchmark
pip install -r requirements/base.txt
```

### 2. Implement your model

Create `asr_benchmark/models/my_model.py` that subclasses `BaseASRModel`:

```python
from asr_benchmark.models.base import BaseASRModel, TranscriptionResult

class MyModel(BaseASRModel):
    def _load_model(self) -> None:
        # Load model weights, tokenizer, processor, etc.
        ...

    def transcribe_batch(self, audio_arrays, sample_rates, audio_lengths_s) -> list[TranscriptionResult]:
        results = []
        for audio, sr, length in zip(audio_arrays, sample_rates, audio_lengths_s):
            def _infer():
                return "transcribed text"
            text, duration = self._timed_call(_infer)
            results.append(TranscriptionResult(
                prediction=text,
                audio_length_s=length,
                transcription_time_s=duration,
            ))
        return results
```

Key points:
- `self.model_id`, `self.language`, `self.kwargs` are set by `BaseASRModel.__init__`
- Use `self._timed_call(fn, *args)` for automatic timing — it returns `(result, elapsed_seconds)`
- `TranscriptionResult` fields: `prediction` (str), `audio_length_s` (float), `transcription_time_s` (float)
- `audio_arrays` are NumPy float32 arrays at the native sample rate for each clip

### 3. Register the model

Add your model to `asr_benchmark/models/__init__.py`:

```python
from asr_benchmark.models.my_model import MyModel

MODEL_REGISTRY = {
    ...
    "my-model": MyModel,
}
```

### 4. Add dependencies (if any)

If your model needs packages beyond `requirements/base.txt`, create `requirements/my_model.txt` and document it in your PR.

### 5. Smoke test

Run a quick sanity check with 20 samples before the full run:

```bash
uv run python run_eval.py \
    --model-type my-model \
    --model-id org/my-model-name \
    --dataset Revolab/ASR-Benchmark-Public \
    --splits train \
    --language ms \
    --max-samples 20 \
    --output-dir results/smoke
```

### 6. Full benchmark run

Run the full public split evaluation:

```bash
uv run python run_eval.py \
    --model-type my-model \
    --model-id org/my-model-name \
    --dataset Revolab/ASR-Benchmark-Public \
    --splits train \
    --language ms \
    --output-dir results/public
```

The output JSONL file should appear at:
`results/public/<model_slug>__Revolab__ASR-Benchmark-Public__train.jsonl`

### 7. Open a pull request

Open a PR against `main`. Fill in the PR template completely — particularly the model details table, benchmark results, and the exact reproduction command.

## Evaluation methodology

- **Dataset**: `Revolab/ASR-Benchmark-Public` — real Malaysian audio across telephony, broadcast, spontaneous speech, and public sets
- **WER**: dual-reference scoring — both raw and normalised references are evaluated per utterance; the lower-WER reference wins
- **Normalisation**: `BasicTextNormalizer` strips tags, punctuation, and lowercases; no number or abbreviation expansion
- **RTFx**: audio duration ÷ transcription time (higher = faster). Not applicable for API models

## Code of conduct

- Keep changes scoped to your model. Do not modify shared evaluation logic, existing model files, or dataset handling unless fixing a genuine bug (open a separate issue for that).
- Results must be reproducible. Include all required environment variables and setup steps in your PR.
- API-based models must be clearly labeled as `API` in the model type.
