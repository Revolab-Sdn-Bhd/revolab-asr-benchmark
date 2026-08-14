#!/usr/bin/env python3
"""
Compute word-level substitution confusion pairs per model.

For each record, aligns reference vs prediction words (Levenshtein traceback)
and tallies which reference word was substituted with which predicted word.

Output: results/malay-benchmark/word_confusion.json
  {model_id: [{ref, hyp, count}, ...sorted by count desc]}

Usage:
    python scripts/compute_confusion.py
    python scripts/compute_confusion.py --top 20 --min-count 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asr_benchmark.utils.manifest import read_manifest


# Additional display-only canonical pairs not in MalayTextNormalizer.
# Applied before word_align so these don't show up as substitution errors.
# (bahawa/bahwa and saspek/suspek are handled by MalayTextNormalizer in the
# manifests themselves — no need to repeat them here.)
_CANONICAL_PAIRS: list[tuple[str, str]] = [
    ("takde",  "tiada"),
    ("takda",  "tiada"),
    ("takdak", "tiada"),
    ("gak",    "juga"),
    ("jer",    "je"),
]

CANONICAL: dict[str, str] = {}
for a, b in _CANONICAL_PAIRS:
    CANONICAL[a] = b
    CANONICAL[b] = b


# Filler / discourse particles — excluded from substitution pairs display.
FILLERS: set[str] = {
    "ah", "aa", "aah", "um", "umm", "hmm", "hm", "hmmm",
    "er", "err", "eh", "uh", "uhh", "ha", "ok", "okay",
    "lah", "la", "lah", "lor", "lor", "mah", "wah", "weh",
}


def canonicalize(words: list[str]) -> list[str]:
    return [CANONICAL.get(w, w) for w in words]


def word_align(ref: list[str], hyp: list[str]) -> list[tuple[str, str, str]]:
    """Levenshtein DP + traceback → list of (op, ref_word, hyp_word)."""
    n, m = len(ref), len(hyp)
    INF = n + m + 1

    # DP table
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # substitution
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                )

    # Traceback
    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            ops.append(("hit", ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1

    return ops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/malay-benchmark")
    parser.add_argument("--output", default=None)
    parser.add_argument("--top", type=int, default=25, help="Top N pairs to keep per model")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum substitution count to include")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "word_confusion.json"

    # {model_id: {(ref_word, hyp_word): count}}
    sub_counts: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
    # {model_id: {ref_word: count}}
    del_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for path in sorted(results_dir.glob("*.jsonl")):
        records = read_manifest(path)
        if not records:
            continue
        model_id = records[0]["model_id"]
        print(f"  {model_id}  ({len(records)} records from {path.name})")

        for r in records:
            ref_text  = r.get("reference_normalized_final", "")
            pred_text = r.get("prediction_normalized", "")
            if not ref_text:
                continue

            ref_words  = canonicalize(ref_text.split())
            pred_words = canonicalize(pred_text.split())

            for op, rw, hw in word_align(ref_words, pred_words):
                if op == "sub" and rw and hw and rw != hw:
                    sub_counts[model_id][(rw, hw)] += 1
                elif op == "del" and rw:
                    del_counts[model_id][rw] += 1

    # ── Substitution confusion output ──
    output: dict[str, list[dict]] = {}
    for model_id, pair_counts in sub_counts.items():
        ranked = sorted(pair_counts.items(), key=lambda x: -x[1])
        top = [
            {"ref": rw, "hyp": hw, "count": c}
            for (rw, hw), c in ranked
            if c >= args.min_count and rw not in FILLERS and hw not in FILLERS
        ][:args.top]
        output[model_id] = top
        print(f"  {model_id}: {len(top)} substitution pairs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved substitutions → {output_path}")

    # ── Deletion output ──
    del_output_path = output_path.parent / "word_deletions.json"

    # Per-model top deletions
    per_model: dict[str, list[dict]] = {}
    for model_id, word_counts in del_counts.items():
        ranked = sorted(word_counts.items(), key=lambda x: -x[1])
        per_model[model_id] = [
            {"word": w, "count": c} for w, c in ranked if c >= args.min_count
        ][:args.top]
        print(f"  {model_id}: {len(per_model[model_id])} deletion words")

    # Cross-model: words deleted by all models (intersection), sorted by total count
    all_model_ids = list(del_counts.keys())
    if all_model_ids:
        word_total: dict[str, int] = defaultdict(int)
        word_model_count: dict[str, int] = defaultdict(int)
        for model_id, word_counts in del_counts.items():
            for w, c in word_counts.items():
                if c >= args.min_count:
                    word_total[w] += c
                    word_model_count[w] += 1

        n_models = len(all_model_ids)
        common = [
            {"word": w, "total_count": word_total[w], "n_models": word_model_count[w]}
            for w in word_total
            if word_model_count[w] >= max(2, n_models - 1)  # deleted by most models
        ]
        common.sort(key=lambda x: -x["total_count"])
    else:
        common = []

    del_result = {"per_model": per_model, "common": common[:args.top]}
    with open(del_output_path, "w", encoding="utf-8") as f:
        json.dump(del_result, f, indent=2, ensure_ascii=False)
    print(f"Saved deletions → {del_output_path}")
    print(f"\nTop common deletions (deleted by most models):")
    for d in common[:10]:
        print(f"  '{d['word']}' — {d['total_count']} times across {d['n_models']} models")


if __name__ == "__main__":
    print("Computing word confusion pairs…\n")
    main()
