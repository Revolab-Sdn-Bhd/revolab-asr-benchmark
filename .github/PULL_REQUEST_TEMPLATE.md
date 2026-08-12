# Model Submission

Thank you for contributing to the Revolab Malaysian ASR Benchmark! Please fill in the details below before opening your PR.

---

## Submission checklist

- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ ] My model implementation lives in `asr_benchmark/models/my_model.py` and subclasses `BaseASRModel`
- [ ] `_load_model()` and `transcribe_batch()` are implemented
- [ ] The model is registered in `asr_benchmark/models/__init__.py` under `MODEL_REGISTRY`
- [ ] I have run the benchmark on the public split and attached results (see below)
- [ ] Results JSONL file is included at `results/public/<model_slug>__Revolab__ASR-Benchmark-Public__train.jsonl`
- [ ] I have verified my implementation works with a smoke test (`--max-samples 20`)
- [ ] Any new dependencies are documented (package name + version, and why it is needed)
- [ ] I have not modified any existing model files or shared evaluation logic (unless fixing a real bug)

---

## Model details

| Field | Value |
|---|---|
| **Model name** | <!-- e.g. Whisper large-v3-turbo --> |
| **Model ID / HF repo** | <!-- e.g. openai/whisper-large-v3-turbo --> |
| **Model type** | <!-- open-source / API --> |
| **Model size** | <!-- parameter count, e.g. 1.5B --> |
| **Language(s)** | <!-- ms / ms+en / multilingual --> |
| **License** | <!-- MIT / Apache-2.0 / proprietary --> |
| **Paper / blog** | <!-- link if available --> |
| **`--model-type` key** | <!-- the key registered in MODEL_REGISTRY --> |

---

## Benchmark results (public split)

Paste the output of `scripts/summarize_results.py` for your model, or fill in the table manually:

| Metric | Value |
|---|---|
| Overall WER | |
| Overall CER | |
| RTFx (if applicable) | |
| Telephony WER | |
| Short-inputs WER | |

---

## Reproducing the run

Provide the exact command used to generate results:

```bash
uv run python run_eval.py \
    --model-type <your-model-type> \
    --model-id <your-model-id> \
    --dataset Revolab/ASR-Benchmark-Public \
    --splits train \
    --language ms \
    --output-dir results/public
```

List any environment variables, API keys, or additional setup steps required:

```
# e.g.
export MY_API_KEY=...
pip install -r requirements/my_model.txt
```

---

## Notes

<!-- Anything else reviewers should know: quantization, special hardware, known limitations, etc. -->
