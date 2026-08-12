#!/usr/bin/env python3
"""
Read all JSONL manifest files in a results directory and print a leaderboard.

Per-row logic:
  - Both references (ref_ours, ref_dataset) are normalized (strip tags/punct/case)
  - WER computed against both; lower WER ref wins for that row
  - Corpus WER = aggregate of all per-row winning stats

Usage:
    python scripts/summarize_results.py --results-dir results/malay-benchmark/
    python scripts/summarize_results.py --results-dir results/ --csv results/leaderboard.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from asr_benchmark.utils.manifest import read_manifest
from asr_benchmark.utils.metrics import (
    build_freq_map,
    compute_all_metrics,
    compute_rare_wer,
    make_common_words,
)

TOP_N_COMMON = 500  # top-N most-frequent words treated as "common" for rare-WER


def _normalise_records(records: list[dict]) -> None:
    """Back-fill reference_normalized_final for old-format manifests (in-place)."""
    if records and "reference_normalized_final" not in records[0]:
        key = (
            "reference_normalized_ours"
            if "reference_normalized_ours" in records[0]
            else "reference_normalized"
        )
        for r in records:
            r["reference_normalized_final"] = r.get(key, "")
            r.setdefault("ref_source", "ours")


def _compute(records: list[dict], common_words: frozenset | None = None) -> dict:
    """Aggregate metrics from per-row winning references."""
    refs  = [r["reference_normalized_final"] for r in records]
    preds = [r["prediction_normalized"] for r in records]
    m = compute_all_metrics(refs, preds)

    total_audio = sum(r["audio_length_s"] for r in records)
    total_time  = sum(r["transcription_time_s"] for r in records)
    n_dataset   = sum(1 for r in records if r.get("ref_source") == "dataset")

    m["num_samples"] = len(records)
    m["rtfx"] = round(total_audio / total_time, 2) if total_time > 0 else float("inf")
    m["rows_dataset_ref"] = n_dataset
    m["rows_ours_ref"]    = len(records) - n_dataset

    if common_words is not None:
        rare = compute_rare_wer(refs, preds, common_words)
        m.update(rare)

    return m


def summarize_manifest(
    path: Path, common_words: frozenset | None = None
) -> dict | None:
    records = read_manifest(path)
    if not records:
        return None

    _normalise_records(records)

    first = records[0]
    overall = _compute(records, common_words)

    by_category: dict[str, dict] = {}
    if "category" in first:
        grouped: dict[str, list] = defaultdict(list)
        for r in records:
            grouped[r["category"]].append(r)
        for cat, cat_records in sorted(grouped.items()):
            by_category[cat] = _compute(cat_records, common_words)

    return {
        "model_id": first.get("model_id", "unknown"),
        "dataset": first.get("dataset", "unknown"),
        "subset": first.get("subset", ""),
        "split": first.get("split", "test"),
        "overall": overall,
        "by_category": by_category,
    }


def _row(label: str, m: dict, col: int = 40) -> str:
    rare_wer_str = f" {m['rare_wer']:>8.2f}%" if "rare_wer" in m else ""
    return (
        f"  {label:<{col}}"
        f" {m['num_samples']:>6}"
        f" {m['wer']:>7.2f}%"
        f" {m['cer']:>7.2f}%"
        f" {m['substitution_rate']:>6.2f}%"
        f" {m['insertion_rate']:>6.2f}%"
        f" {m['deletion_rate']:>6.2f}%"
        f" {m['rtfx']:>7.2f}"
        f"{rare_wer_str}"
    )


def print_leaderboard(summaries: list[dict]) -> None:
    summaries.sort(key=lambda s: s["overall"]["wer"])

    has_rare = any("rare_wer" in s["overall"] for s in summaries)
    COL = 40
    header = (
        f"  {'Model / Category':<{COL}}"
        f" {'N':>6} {'WER':>8} {'CER':>8}"
        f" {'Sub%':>7} {'Ins%':>7} {'Del%':>7} {'RTFx':>7}"
        + (f" {'RareWER':>9}" if has_rare else "")
    )
    sep  = "=" * len(header)
    thin = "-" * len(header)

    for s in summaries:
        ds = f"{s['dataset']}/{s['subset']}" if s["subset"] else s["dataset"]
        o  = s["overall"]
        print(f"\n{sep}")
        print(f"  {s['model_id']}  ·  {ds}  ·  split={s['split']}")
        print(f"  ref source: {o['rows_dataset_ref']} rows via dataset_ref, {o['rows_ours_ref']} rows via ours")
        print(sep)
        print(header)
        print(thin)
        print(_row("OVERALL", o, COL))

        if s["by_category"]:
            print(thin)
            for cat, m in s["by_category"].items():
                print(_row(cat, m, COL))

    print(f"\n{sep}")

    # Quick-view table
    print("\n── QUICK VIEW (sorted by WER) ──")
    rare_col = f" {'RareWER':>9}" if has_rare else ""
    qh = (
        f"  {'Model':<45} {'Split':<14}"
        f" {'WER':>8} {'CER':>8} {'Sub%':>7} {'Ins%':>7} {'Del%':>7} {'RTFx':>7}"
        + rare_col
    )
    print(qh)
    print("-" * len(qh))
    for s in summaries:
        o = s["overall"]
        rare_str = f" {o['rare_wer']:>8.2f}%" if "rare_wer" in o else ""
        print(
            f"  {s['model_id']:<45} {s['split']:<14}"
            f" {o['wer']:>7.2f}%"
            f" {o['cer']:>7.2f}%"
            f" {o['substitution_rate']:>6.2f}%"
            f" {o['insertion_rate']:>6.2f}%"
            f" {o['deletion_rate']:>6.2f}%"
            f" {o['rtfx']:>7.2f}"
            f"{rare_str}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", help="Directory containing JSONL manifests.")
    parser.add_argument("--csv", default=None, help="Optional path to write CSV leaderboard.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    manifests   = sorted(results_dir.glob("*.jsonl"))

    if not manifests:
        print(f"No JSONL files found in {results_dir}")
        return

    # Build cross-corpus frequency map for rare-WER (all models share the same refs)
    all_refs: list[str] = []
    manifest_records: dict[Path, list[dict]] = {}
    for m in manifests:
        records = read_manifest(m)
        if not records:
            continue
        _normalise_records(records)
        manifest_records[m] = records
        all_refs.extend(r["reference_normalized_final"] for r in records)

    common_words: frozenset | None = None
    if all_refs:
        freq_map = build_freq_map(all_refs)
        common_words = make_common_words(freq_map, TOP_N_COMMON)
        unique = len(freq_map)
        print(
            f"Rare-WER: top {TOP_N_COMMON} of {unique} unique words are 'common'; "
            f"evaluating tail ({unique - TOP_N_COMMON} words)."
        )

    summaries = []
    for m in manifests:
        if m not in manifest_records:
            continue
        records = manifest_records[m]
        first = records[0]
        overall = _compute(records, common_words)
        by_category: dict[str, dict] = {}
        if "category" in first:
            grouped: dict[str, list] = defaultdict(list)
            for r in records:
                grouped[r["category"]].append(r)
            for cat, cat_records in sorted(grouped.items()):
                by_category[cat] = _compute(cat_records, common_words)
        summaries.append({
            "model_id": first.get("model_id", "unknown"),
            "dataset":  first.get("dataset", "unknown"),
            "subset":   first.get("subset", ""),
            "split":    first.get("split", "test"),
            "overall":  overall,
            "by_category": by_category,
        })

    if not summaries:
        return

    print_leaderboard(summaries)

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for s in summaries:
            row = {
                "model_id": s["model_id"],
                "dataset":  s["dataset"],
                "subset":   s["subset"],
                "split":    s["split"],
            }
            for k, v in s["overall"].items():
                row[f"overall_{k}"] = v
            for cat, m in s["by_category"].items():
                for k, v in m.items():
                    row[f"{cat}_{k}"] = v
            rows.append(row)

        all_keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written to {out}")


if __name__ == "__main__":
    main()
