#!/usr/bin/env python3
"""
CLI entry-point for the ASR benchmark.

Model-oriented: the model is loaded ONCE and all requested splits are
evaluated sequentially before the process exits. This avoids repeated
GPU memory allocation for large models.

Usage examples
--------------
# Whisper large-v3 on Common Voice English (100 samples, streaming)
python run_eval.py \
    --model-type whisper \
    --model-id openai/whisper-large-v3 \
    --dataset mozilla-foundation/common_voice_13_0 \
    --subset en \
    --splits test \
    --max-samples 100

# Qwen3-ASR on the Revolab Malay benchmark.
# This dataset has a single `train` split; its 12 domains are values of the
# `category` column, so pass --metadata-columns category for the breakdown
# (or --category telephony to score one domain).
python run_eval.py \
    --model-type qwen \
    --model-id Qwen/Qwen3-ASR-1.7B \
    --dataset Revolab/ASR-Benchmark-Public \
    --splits train \
    --text-column text \
    --normalized-text-column normalized_text \
    --metadata-columns category \
    --language ms \
    --no-streaming

# Gemini (API model)
GEMINI_API_KEY=xxx python run_eval.py \
    --model-type gemini \
    --model-id gemini-2.5-flash \
    --dataset mozilla-foundation/common_voice_13_0 \
    --subset en \
    --splits test \
    --max-samples 50

# ElevenLabs (API model)
ELEVENLABS_API_KEY=xxx python run_eval.py \
    --model-type elevenlabs \
    --model-id scribe_v2 \
    --dataset mozilla-foundation/common_voice_13_0 \
    --subset en \
    --splits test \
    --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from asr_benchmark.models import MODEL_REGISTRY
from asr_benchmark.runner import ModelConfig, DatasetConfig, Evaluator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run ASR benchmark for one model across one or more dataset splits. "
            "The model is loaded once and reused for all splits."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument(
        "--model-type",
        required=True,
        choices=list(MODEL_REGISTRY),
        help="Model backend to use.",
    )
    p.add_argument("--model-id", required=True, help="HF repo id or local path / checkpoint name.")

    # Dataset
    p.add_argument(
        "--dataset",
        default="mozilla-foundation/common_voice_13_0",
        help="HuggingFace dataset repo id.",
    )
    p.add_argument("--subset", default=None, help="Dataset config/subset (e.g. language code).")
    p.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        metavar="SPLIT",
        help="One or more dataset splits to evaluate (model is loaded once for all).",
    )
    p.add_argument("--audio-column", default="audio", help="Column name containing audio.")
    p.add_argument("--text-column", default="sentence", help="Column name containing reference text.")
    p.add_argument("--normalized-text-column", default=None,
                   help="Dataset column with pre-normalized text (used as-is for a second WER pass).")
    p.add_argument("--max-samples", type=int, default=None, help="Limit number of samples per split.")
    p.add_argument("--no-streaming", action="store_true", help="Disable HF streaming mode.")
    p.add_argument(
        "--metadata-columns",
        nargs="*",
        default=[],
        metavar="COL",
        help="Extra dataset columns to store in manifests (e.g. category).",
    )
    p.add_argument(
        "--category",
        default=None,
        help="Only evaluate rows whose 'category' column matches this value.",
    )

    # Eval
    p.add_argument("--language", default="en", help="BCP-47 language tag (drives normalizer and model).")
    p.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
    p.add_argument("--output-dir", default="results", help="Directory to write manifest JSONL files.")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args(argv)

    model_cfg = ModelConfig(
        model_type=args.model_type,
        model_id=args.model_id,
        language=args.language,
    )

    dataset_cfg = DatasetConfig(
        dataset_name=args.dataset,
        splits=args.splits,
        audio_column=args.audio_column,
        text_column=args.text_column,
        normalized_text_column=args.normalized_text_column,
        metadata_columns=args.metadata_columns,
        subset=args.subset,
        max_samples=args.max_samples,
        streaming=not args.no_streaming,
        category_filter=args.category,
    )

    evaluator = Evaluator(
        model_config=model_cfg,
        dataset_config=dataset_cfg,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    summaries = evaluator.run()

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
