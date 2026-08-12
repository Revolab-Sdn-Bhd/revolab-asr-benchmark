#!/usr/bin/env python3
"""
Noise-robustness analysis: bucket each model's WER by audio noise level.

Uses the per-sample SNR tags written by scripts/tag_noise.py (noise_tags.json,
keyed by reference text -> {snr_db, noise_level}, buckets clean/moderate/noisy)
to slice each model's results manifest. The interesting question is the
clean -> noisy WER *degradation*: a robust model holds up as audio gets harder,
a fragile one falls off a cliff.

Usage:
    uv run python scripts/noise_analysis.py --results-dir results/public
    uv run python scripts/noise_analysis.py --results-dir results/public --model-id qwen3-asr-0.6b-malay-aug-s12000-L0
    uv run python scripts/noise_analysis.py --results-dir results/public --json-out results/public/noise_analysis.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from asr_benchmark.utils.manifest import read_manifest

BUCKETS = ["clean", "moderate", "noisy"]


def _load_tags(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # flatten {split: {ref: {...}}} -> {ref: {...}} (refs are globally unique enough)
    flat: dict[str, dict] = {}
    for split_tags in raw.values():
        if isinstance(split_tags, dict):
            flat.update(split_tags)
    return flat


def _model_rows(results_dir: Path, model_id: str | None) -> dict[str, list[dict]]:
    """model_id -> list of rows, across all manifests in results_dir."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("*.jsonl")):
        for r in read_manifest(path):
            if model_id is None or r.get("model_id") == model_id:
                by_model[r.get("model_id", "?")].append(r)
    return by_model


def _bucket(rows: list[dict], tags: dict[str, dict]) -> dict[str, dict]:
    """Aggregate per-noise-bucket word-edit stats + empty-prediction counts."""
    acc = {b: {"n": 0, "subs": 0, "ins": 0, "dels": 0, "ref_words": 0,
               "empties": 0, "snr": []} for b in BUCKETS + ["unknown"]}
    for r in rows:
        tag = tags.get(r.get("reference", ""))
        level = tag["noise_level"] if tag else "unknown"
        a = acc[level]
        a["n"] += 1
        a["subs"] += r.get("word_substitutions", 0)
        a["ins"] += r.get("word_insertions", 0)
        a["dels"] += r.get("word_deletions", 0)
        a["ref_words"] += r.get("num_ref_words", 0)
        if tag:
            a["snr"].append(tag["snr_db"])
        if not (r.get("prediction") or "").strip() and r.get("num_ref_words", 0) > 0:
            a["empties"] += 1
    return acc


def _wer(b: dict) -> float:
    return 100 * (b["subs"] + b["ins"] + b["dels"]) / b["ref_words"] if b["ref_words"] else 0.0


def build_report(model_id: str, rows: list[dict], tags: dict[str, dict]) -> dict:
    acc = _bucket(rows, tags)
    per = {}
    for b in BUCKETS + ["unknown"]:
        bb = acc[b]
        per[b] = {
            "n": bb["n"],
            "wer": round(_wer(bb), 2),
            "substitutions": bb["subs"], "insertions": bb["ins"], "deletions": bb["dels"],
            "ref_words": bb["ref_words"], "empties": bb["empties"],
            "mean_snr_db": round(sum(bb["snr"]) / len(bb["snr"]), 1) if bb["snr"] else None,
        }
    degradation = per["noisy"]["wer"] - per["clean"]["wer"] if (per["clean"]["n"] and per["noisy"]["n"]) else None
    return {"model_id": model_id, "num_rows": len(rows), "by_noise": per,
            "degradation_pp": round(degradation, 2) if degradation is not None else None}


def print_table(reports: list[dict], sort_by: str) -> None:
    key = {"noisy": lambda r: r["by_noise"]["noisy"]["wer"],
           "clean": lambda r: r["by_noise"]["clean"]["wer"],
           "degradation": lambda r: r["degradation_pp"] if r["degradation_pp"] is not None else 9e9}[sort_by]
    reports = sorted(reports, key=key)
    hdr = f"{'model':<46} {'clean':>10} {'moderate':>10} {'noisy':>10} {'clean->noisy':>12}"
    print(hdr); print("-" * len(hdr))
    for r in reports:
        b = r["by_noise"]
        def cell(x): return f"{x['wer']:5.2f}%/{x['n']:<3d}"
        deg = f"+{r['degradation_pp']:.2f}pp" if r["degradation_pp"] is not None else "  -"
        print(f"{r['model_id']:<46} {cell(b['clean']):>10} {cell(b['moderate']):>10} {cell(b['noisy']):>10} {deg:>12}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/public")
    ap.add_argument("--noise-tags", default=None, help="default <results-dir>/noise_tags.json")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--sort", choices=["noisy", "clean", "degradation"], default="noisy")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    tags_path = Path(args.noise_tags) if args.noise_tags else results_dir / "noise_tags.json"
    tags = _load_tags(tags_path)
    by_model = _model_rows(results_dir, args.model_id)

    reports = [build_report(mid, rows, tags) for mid, rows in by_model.items()]
    print_table(reports, args.sort)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {len(reports)} model report(s) to {args.json_out}")


if __name__ == "__main__":
    main()
