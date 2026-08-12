#!/usr/bin/env python3
"""
Tag all benchmark audio samples with SNR-based noise level.

Downloads audio from HF, estimates SNR per sample using an energy-based
percentile method, and saves noise_tags.json used by the Explorer.

Buckets:
    clean     → SNR > 25 dB
    moderate  → 15–25 dB
    noisy     → < 15 dB

Usage:
    uv run python scripts/tag_noise.py
    uv run python scripts/tag_noise.py --splits telephony_revolab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asr_benchmark.utils.data import _decode_audio
from asr_benchmark.utils.manifest import read_manifest

DATASET_NAME = "Revolab/ASR-Benchmark-Private"


def estimate_snr(audio: np.ndarray) -> float:
    """Energy-based SNR via percentile method on 25 ms frames."""
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
        

    # 25 ms frames at 16 kHz = 400 samples; 10 ms hop = 160 samples
    frame_len, hop_len = 400, 160
    if len(audio) < frame_len:
        return 40.0  # too short to estimate — treat as clean

    rms = np.array([
        np.sqrt(np.mean(audio[s : s + frame_len] ** 2))
        for s in range(0, len(audio) - frame_len, hop_len)
    ])

    noise_floor  = np.percentile(rms, 10)
    signal_level = np.percentile(rms, 90)

    if noise_floor < 1e-8:
        return 40.0  # silent background → very clean

    return float(round(20 * np.log10(signal_level / noise_floor), 2))


def snr_to_level(snr_db: float) -> str:
    if snr_db > 25:
        return "clean"
    if snr_db > 15:
        return "moderate"
    return "noisy"


def get_refs_for_split(results_dir: Path, split: str) -> set[str]:
    refs: set[str] = set()
    for path in results_dir.glob("*.jsonl"):
        for r in read_manifest(path):
            if r.get("split") == split:
                refs.add(r["reference"])
    return refs


def tag_split(split: str, want_refs: set[str], dataset_name: str = DATASET_NAME) -> dict[str, dict]:
    from datasets import load_dataset

    print(f"  Loading {dataset_name} split={split} …")
    ds = load_dataset(dataset_name, split=split, streaming=False)

    tags: dict[str, dict] = {}
    total = len(ds)

    for i, row in enumerate(ds):
        ref = row.get("text", "")
        if ref not in want_refs:
            continue
        arr, sr = _decode_audio(row["audio"], 16000)
        snr = estimate_snr(arr)
        tags[ref] = {"snr_db": snr, "noise_level": snr_to_level(snr)}

        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total} rows scanned, {len(tags)} tagged …")

    return tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag benchmark audio with noise levels")
    parser.add_argument("--results-dir", default="results/public")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument(
        "--splits", nargs="+",
        default=["train"],
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "noise_tags.json"

    # Load existing tags so we can resume if interrupted
    existing: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)

    all_tags = dict(existing)

    for split in args.splits:
        print(f"\n── {split} ──")
        want_refs = get_refs_for_split(results_dir, split)
        already   = set((existing.get(split) or {}).keys())
        remaining = want_refs - already
        print(f"  {len(want_refs)} samples total, {len(already)} already tagged, {len(remaining)} to process")

        if not remaining:
            print("  Skipped (all done)")
            continue

        new_tags = tag_split(split, remaining, dataset_name=args.dataset)
        split_tags = {**(existing.get(split) or {}), **new_tags}
        all_tags[split] = split_tags

        # Save after each split so progress isn't lost on interruption
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_tags, f, indent=2, ensure_ascii=False)

        counts: dict[str, int] = {}
        for v in split_tags.values():
            lv = v["noise_level"]
            counts[lv] = counts.get(lv, 0) + 1
        print(f"  Distribution: {counts}")

    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
