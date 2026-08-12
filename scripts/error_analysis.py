#!/usr/bin/env python3
"""
Qualitative error analysis for one or all models' results manifest(s).

Goes beyond the aggregate WER/CER numbers in summarize_results.py to surface
*why* a model is losing points: which categories are worst, whether errors are
concentrated in a few catastrophic rows vs. spread evenly, and whether there's
a systematic pattern (script-mismatch hallucination, filler-word deletion,
reduplication merging, word-repetition hallucination) worth fixing.

Usage:
    python scripts/error_analysis.py --model-id ilmu-asr-v4.2
    python scripts/error_analysis.py --model-id ilmu-asr-v4.2 --category short-inputs
    python scripts/error_analysis.py --model-id ilmu-asr-v4.2 --json

    # Every model in --results-dir, written as one JSON file for the UI to consume:
    python scripts/error_analysis.py --all-models --json-out results/public/error_analysis.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from asr_benchmark.utils.manifest import read_manifest
from asr_benchmark.utils.metrics import _align_words

FILLER_WORDS = {"ah", "um", "erm", "uh", "hmm", "eh"}

# Unicode script blocks that should never appear in this benchmark's
# Malay/English references — their presence in a prediction is a strong
# signal of language-ID / decoding collapse rather than a normal ASR miss.
NON_LATIN_RE = re.compile(
    r"[一-鿿"  # CJK
    r"぀-ヿ"  # Hiragana/Katakana
    r"가-힯"  # Hangul
    r"؀-ۿ"  # Arabic
    r"Ѐ-ӿ"  # Cyrillic
    r"฀-๿"  # Thai
    r"ऀ-ॿ"  # Devanagari
    r"]"
)


def _all_model_ids(results_dirs: list[Path]) -> list[str]:
    ids = []
    for results_dir in results_dirs:
        for path in sorted(results_dir.glob("*.jsonl")):
            records = read_manifest(path)
            if records:
                mid = records[0].get("model_id")
                if mid and mid not in ids:
                    ids.append(mid)
    return ids


def _find_manifests(results_dirs: list[Path], model_id: str) -> list[Path]:
    matches = []
    for results_dir in results_dirs:
        for path in sorted(results_dir.glob("*.jsonl")):
            records = read_manifest(path)
            if records and records[0].get("model_id") == model_id:
                matches.append(path)
    return matches


def _wer_final(r: dict) -> float:
    if "wer_ours" in r and "wer_dataset" in r:
        return min(r["wer_ours"], r["wer_dataset"])
    return r.get("wer_ours", r.get("wer_dataset", 0.0))


def category_stats(rows: list[dict]) -> list[dict]:
    stats = defaultdict(Counter)
    for r in rows:
        c = r.get("category", "unknown")
        stats[c]["subs"] += r["word_substitutions"]
        stats[c]["ins"] += r["word_insertions"]
        stats[c]["dels"] += r["word_deletions"]
        stats[c]["ref_words"] += r["num_ref_words"]
        stats[c]["n"] += 1

    ranked = sorted(
        stats.items(),
        key=lambda kv: -(kv[1]["subs"] + kv[1]["ins"] + kv[1]["dels"]) / max(kv[1]["ref_words"], 1),
    )
    out = []
    for cat, s in ranked:
        wer = 100 * (s["subs"] + s["ins"] + s["dels"]) / s["ref_words"] if s["ref_words"] else 0
        out.append({
            "category": cat, "n": s["n"], "wer": round(wer, 2),
            "substitutions": s["subs"], "insertions": s["ins"], "deletions": s["dels"],
            "ref_words": s["ref_words"],
        })
    return out


def print_category_table(stats: list[dict], overall: dict) -> None:
    print(f"{'category':17s} {'n':>4s} {'WER%':>7s} {'sub':>5s} {'ins':>5s} {'del':>5s} {'ref_words':>9s}")
    print("-" * 62)
    for s in stats:
        print(f"{s['category']:17s} {s['n']:4d} {s['wer']:7.2f} {s['substitutions']:5d} {s['insertions']:5d} {s['deletions']:5d} {s['ref_words']:9d}")
    print("-" * 62)
    print(f"{'OVERALL':17s} {overall['n']:4d} {overall['wer']:7.2f} {overall['substitutions']:5d} {overall['insertions']:5d} {overall['deletions']:5d} {overall['ref_words']:9d}")


def _non_latin_fraction(text: str) -> float:
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 0.0
    return sum(1 for c in non_space if NON_LATIN_RE.match(c)) / len(non_space)


def script_mismatch_rows(rows: list[dict], min_fraction: float = 0.03) -> list[dict]:
    """Real script collapse (e.g. Chinese words on Malay audio), not an isolated
    stray non-Latin character amid an otherwise-correct, long transcript — the
    latter is noise (e.g. one misplaced Arabic glyph in an otherwise-perfect
    322-character sentence) rather than a language-ID failure. A fraction (not
    raw count) correctly catches full single-word wipeouts on short clips too
    (e.g. "ah" -> "啊" is 100% non-Latin despite being only 1 character).
    """
    hits = [r for r in rows if _non_latin_fraction(r["prediction"]) >= min_fraction]
    return sorted(hits, key=lambda r: r["audio_length_s"])


def empty_prediction_rows(rows: list[dict]) -> list[dict]:
    """Model produced no output at all despite a non-trivial reference."""
    hits = [r for r in rows if not r["prediction"].strip() and r["num_ref_words"] > 0]
    return sorted(hits, key=lambda r: -r["audio_length_s"])


def filler_deletions(rows: list[dict]) -> tuple[int, Counter, list[dict]]:
    total = 0
    by_cat = Counter()
    examples = []
    for r in rows:
        ref = r["reference_normalized_final"].split()
        hyp = r["prediction_normalized"].split()
        hit = False
        for op, rw, hw in _align_words(ref, hyp):
            if op == "del" and rw in FILLER_WORDS:
                total += 1
                by_cat[r.get("category", "unknown")] += 1
                hit = True
        if hit:
            examples.append({"category": r.get("category", "unknown"), **_evidence(r)})
    return total, by_cat, examples


def _evidence(r: dict) -> dict:
    """Fields needed by the explorer UI to jump from a finding to its audio sample."""
    return {
        "reference": r.get("reference", ""),
        "split": r.get("split", ""),
        "audio_length_s": r.get("audio_length_s"),
        "wer": round(_wer_final(r), 2),
    }


def reduplication_merges(rows: list[dict]) -> list[dict]:
    """Two identical consecutive ref words collapsed into one hyp token.

    The DP word-alignment renders this as an adjacent (sub, del) or (del, sub)
    pair rather than two 'sub' ops, since only one hyp token is consumed for
    both ref words.
    """
    hits = []
    for r in rows:
        ref = r["reference_normalized_final"].split()
        hyp = r["prediction_normalized"].split()
        ops = _align_words(ref, hyp)
        for i in range(len(ops) - 1):
            a, b = ops[i], ops[i + 1]
            match = None
            if a[0] == "sub" and b[0] == "del" and a[1] == b[1] and a[2] == a[1] + b[1]:
                match = a[2]
            elif a[0] == "del" and b[0] == "sub" and a[1] == b[1] and b[2] == b[1] + a[1]:
                match = b[2]
            if match:
                hits.append({"category": r.get("category", "unknown"), "word": a[1], "merged": match, **_evidence(r)})
    return hits


def repetition_hallucinations(rows: list[dict]) -> list[dict]:
    """Hypothesis repeats a word 2+ times where the reference doesn't."""
    hits = []
    for r in rows:
        ref = r["reference_normalized_final"].split()
        hyp = r["prediction_normalized"].split()
        ops = _align_words(ref, hyp)
        run_hyp, run_ref = [], []
        for op, rw, hw in ops:
            if op in ("ins", "sub") and hw is not None:
                run_hyp.append(hw)
                if rw is not None:
                    run_ref.append(rw)
            else:
                if len(run_hyp) >= 2 and len(set(run_hyp)) == 1:
                    hits.append({"category": r.get("category", "unknown"), "ref": run_ref, "hyp": run_hyp, **_evidence(r)})
                run_hyp, run_ref = [], []
        if len(run_hyp) >= 2 and len(set(run_hyp)) == 1:
            hits.append({"category": r.get("category", "unknown"), "ref": run_ref, "hyp": run_hyp, **_evidence(r)})
    return hits


def top_substitution_pairs(rows: list[dict], top_n: int) -> list[dict]:
    """Group consecutive 'sub' ops into phrase-level substitution pairs."""
    counter = Counter()
    for r in rows:
        ref = r["reference_normalized_final"].split()
        hyp = r["prediction_normalized"].split()
        ops = _align_words(ref, hyp)
        buf_ref, buf_hyp = [], []
        for op, rw, hw in ops + [("hit", None, None)]:
            if op == "sub":
                buf_ref.append(rw)
                buf_hyp.append(hw)
            else:
                if buf_ref:
                    counter[(" ".join(buf_ref), " ".join(buf_hyp))] += 1
                buf_ref, buf_hyp = [], []
    return [{"ref": r_, "hyp": h_, "count": c} for (r_, h_), c in counter.most_common(top_n)]


def worst_rows(rows: list[dict], top_n: int) -> list[dict]:
    top = sorted(rows, key=_wer_final, reverse=True)[:top_n]
    return [
        {
            "category": r.get("category"),
            "reference_normalized": r["reference_normalized_final"],
            "prediction_normalized": r["prediction_normalized"],
            **_evidence(r),
        }
        for r in top
    ]


def _counted(hits: list[dict], limit: int = 10) -> dict:
    return {"count": len(hits), "examples": hits[:limit]}


def build_report(model_id: str, rows: list[dict], top_n_worst: int, top_n_subs: int) -> dict:
    stats = category_stats(rows)
    overall = Counter()
    for r in rows:
        overall["subs"] += r["word_substitutions"]
        overall["ins"] += r["word_insertions"]
        overall["dels"] += r["word_deletions"]
        overall["ref_words"] += r["num_ref_words"]
    overall_wer = 100 * (overall["subs"] + overall["ins"] + overall["dels"]) / overall["ref_words"] if overall["ref_words"] else 0
    overall_dict = {
        "n": len(rows), "wer": round(overall_wer, 2),
        "substitutions": overall["subs"], "insertions": overall["ins"], "deletions": overall["dels"],
        "ref_words": overall["ref_words"],
    }

    mismatch = script_mismatch_rows(rows)
    blank = empty_prediction_rows(rows)
    filler_total, filler_by_cat, filler_examples = filler_deletions(rows)
    total_dels = overall["dels"]

    blank_by_cat = Counter(r.get("category", "unknown") for r in blank)

    return {
        "model_id": model_id,
        "num_rows": len(rows),
        "overall": overall_dict,
        "by_category": stats,
        "script_mismatch": {
            "count": len(mismatch),
            "examples": [
                {
                    "category": r.get("category"), "audio_length_s": r["audio_length_s"],
                    "reference_normalized": r["reference_normalized_final"], "prediction_normalized": r["prediction_normalized"],
                    **_evidence(r),
                }
                for r in mismatch[:20]
            ],
        },
        "empty_predictions": {
            "count": len(blank),
            "pct_of_rows": round(100 * len(blank) / len(rows), 1) if rows else 0,
            "by_category": dict(blank_by_cat.most_common()),
            "examples": [
                {"category": r.get("category"), "reference_normalized": r["reference_normalized_final"], **_evidence(r)}
                for r in blank[:10]
            ],
        },
        "filler_word_deletions": {
            "count": filler_total,
            "pct_of_total_deletions": round(100 * filler_total / total_dels, 1) if total_dels else 0,
            "by_category": dict(filler_by_cat.most_common()),
            "examples": filler_examples[:10],
        },
        "reduplication_merges": _counted(reduplication_merges(rows)),
        "repetition_hallucinations": _counted(repetition_hallucinations(rows)),
        "top_substitutions": top_substitution_pairs(rows, top_n_subs),
        "worst_rows": worst_rows(rows, top_n_worst),
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def build_narrative(model_id: str, report: dict, all_reports: dict[str, dict]) -> dict:
    """Deterministic, template-driven write-up — same reasoning as a manual read of
    error_analysis output, just automated. No LLM in the loop, so no spin/variance."""
    peers = {mid: r for mid, r in all_reports.items() if mid != model_id}
    peer_wers = {mid: r["overall"]["wer"] for mid, r in all_reports.items()}
    ranked = sorted(peer_wers.items(), key=lambda kv: kv[1])
    rank = next(i for i, (mid, _) in enumerate(ranked, 1) for _ in [None] if mid == model_id)
    n_models = len(ranked)

    o = report["overall"]
    summary = f"{o['wer']:.2f}% WER — ranks {rank} of {n_models} models tested."

    # ── Peer baselines, to judge whether a count is actually unusual ──────────
    peer_blank_pct   = [r["empty_predictions"]["pct_of_rows"] for r in peers.values()]
    peer_mismatch    = [r["script_mismatch"]["count"] for r in peers.values()]
    peer_redup       = [r["reduplication_merges"]["count"] for r in peers.values()]
    peer_rep         = [r["repetition_hallucinations"]["count"] for r in peers.values()]

    bugs: list[tuple[float, str]] = []  # (severity_score, sentence) — sorted desc, worst leads

    ep = report["empty_predictions"]
    if ep["count"] >= 3 and ep["pct_of_rows"] >= 0.5:
        med = _median(peer_blank_pct)
        top_cats = ", ".join(f"{c} ({n})" for c, n in list(ep["by_category"].items())[:3])
        severity = ep["pct_of_rows"]
        if ep["pct_of_rows"] > max(med * 3, 2.0):
            bugs.append((severity + 100, (
                f"The dominant issue is a hard failure, not a transcription mistake: it returns a completely blank "
                f"prediction on {ep['count']} of {report['num_rows']} clips ({ep['pct_of_rows']}%) — far above the "
                f"peer median of {med:.1f}%. Worst in {top_cats}. Every blank scores 100% WER regardless of "
                f"reference length, so this alone likely accounts for a large share of the overall error."
            )))
        else:
            bugs.append((severity, (
                f"Returns a blank prediction on {ep['count']} clips ({ep['pct_of_rows']}% of rows), concentrated in {top_cats}."
            )))

    sm = report["script_mismatch"]
    if sm["count"]:
        ex = sm["examples"][0]
        lens = [e["audio_length_s"] for e in sm["examples"]]
        length_clause = (
            f"usually on very short clips (median {_median(lens):.1f}s)" if _median(lens) <= 2.0
            else f"not limited to short clips (median {_median(lens):.1f}s)"
        )
        bugs.append((sm["count"] + 50, (
            f"Script-mismatch hallucination: outputs non-Latin script on {sm['count']} rows despite Malay/English "
            f"audio (e.g. \"{ex['reference_normalized']}\" -> \"{ex['prediction_normalized']}\"), {length_clause} "
            f"— a language-ID/decoding collapse rather than a normal ASR miss."
        )))

    redup = report["reduplication_merges"]
    if redup["count"] >= 3:
        med = _median(peer_redup)
        ex = redup["examples"][0]
        if redup["count"] > max(med * 3, 20):
            bugs.append((redup["count"], (
                f"Has an unusually large reduplication-merge problem: {redup['count']} instances of reduplicated "
                f"Malay words collapsed into one token (e.g. \"{ex['word']} {ex['word']}\" -> \"{ex['merged']}\") — "
                f"far more than peers (median {med:.0f})."
            )))
        else:
            bugs.append((redup["count"] / 5, (
                f"Occasionally merges reduplicated Malay words (\"{ex['word']} {ex['word']}\" -> \"{ex['merged']}\"), "
                f"{redup['count']} times."
            )))

    rep = report["repetition_hallucinations"]
    if rep["count"] >= 8:
        med = _median(peer_rep)
        bugs.append((rep["count"] / 2, (
            f"Repeats words the reference doesn't {rep['count']} times (peer median {med:.0f}) — a mild "
            f"hallucination pattern, mostly on short/noisy clips."
        )))

    fd = report["filler_word_deletions"]
    filler_note = (
        f"{fd['pct_of_total_deletions']}% of its deletion errors are dropped filler words (ah/um) — "
        f"this is a benchmark-wide pattern (most models sit around 30-39%), not model-specific."
        if fd["count"] else "Does not drop filler words the way most peers do."
    )

    bugs.sort(key=lambda b: -b[0])
    bug_sentences = [s for _, s in bugs]

    # ── Category strengths/weaknesses ──────────────────────────────────────────
    cats = report["by_category"]
    best_cat, worst_cat = cats[-1], cats[0]  # by_category is sorted worst-first

    def _cat_reason(c: dict) -> str:
        total = max(c["substitutions"] + c["insertions"] + c["deletions"], 1)
        if c["deletions"] / total > 0.5:
            return "mostly drops words"
        if c["insertions"] / total > 0.3:
            return "mostly hallucinates extra words"
        return "mostly mishears words"

    strengths = [f"Best category: {best_cat['category']} at {best_cat['wer']:.1f}% WER."]
    if not bug_sentences:
        strengths.append("No systematic bugs detected (no script-mismatch, blank output, or reduplication issues).")
    weaknesses = bug_sentences + [
        f"Worst category: {worst_cat['category']} at {worst_cat['wer']:.1f}% WER ({_cat_reason(worst_cat)})."
    ]

    if bug_sentences:
        verdict = "The score is driven more by the systematic issue(s) above than by broad mishearing of Malay/English speech."
    elif worst_cat["wer"] > 2 * o["wer"]:
        verdict = f"No systematic bugs found — the gap is concentrated in {worst_cat['category']}, not spread evenly."
    else:
        verdict = "No systematic bugs found and errors are fairly evenly spread across categories — a broad accuracy gap, not a specific fixable issue."

    return {
        "summary": summary,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "filler_note": filler_note,
        "verdict": verdict,
    }


def print_report(report: dict) -> None:
    print(f"=== {report['model_id']} — {report['num_rows']} rows ===\n")

    if "narrative" in report:
        nar = report["narrative"]
        print(f"── Narrative ──")
        print(nar["summary"])
        for s in nar["strengths"]:
            print(f"  + {s}")
        for w in nar["weaknesses"]:
            print(f"  - {w}")
        print(f"  {nar['filler_note']}")
        print(f"  Verdict: {nar['verdict']}")
        print()

    print("── Per-category WER (worst first) ──")
    print_category_table(report["by_category"], report["overall"])

    sm = report["script_mismatch"]
    print(f"\n── Script-mismatch hallucinations (non-Latin output on Malay/English audio): {sm['count']} rows ──")
    for r in sm["examples"]:
        print(f"  [{r.get('category'):15s}] len={r['audio_length_s']:.2f}s  REF: {r['reference_normalized']!r}  PRED: {r['prediction_normalized']!r}")

    ep = report["empty_predictions"]
    print(f"\n── Blank predictions (model returned nothing for non-trivial audio): {ep['count']} rows ({ep['pct_of_rows']}% of all rows) ──")
    for cat, n in ep["by_category"].items():
        print(f"  {cat:17s} {n}")

    fd = report["filler_word_deletions"]
    print(f"\n── Filler-word ({'/'.join(sorted(FILLER_WORDS))}) deletions: {fd['count']} of {report['overall']['deletions']} total deletions ({fd['pct_of_total_deletions']:.1f}%) ──")
    for cat, n in fd["by_category"].items():
        print(f"  {cat:17s} {n}")

    redup = report["reduplication_merges"]
    print(f"\n── Reduplication-merge errors (e.g. 'kanak kanak' -> 'kanakkanak'): {redup['count']} ──")
    for h in redup["examples"]:
        print(f"  [{h['category']:15s}] '{h['word']} {h['word']}' -> '{h['merged']}'")

    reps = report["repetition_hallucinations"]
    print(f"\n── Word-repetition hallucinations (hyp repeats a word the ref doesn't): {reps['count']} ──")
    for h in reps["examples"]:
        print(f"  [{h['category']:15s}] ref={h['ref']} -> hyp={h['hyp']}")

    print(f"\n── Top substitution pairs ──")
    for s in report["top_substitutions"]:
        print(f"  {s['count']:3d}  {s['ref']!r} -> {s['hyp']!r}")

    print(f"\n── Worst rows by WER ──")
    for r in report["worst_rows"]:
        print(f"  WER={r['wer']:6.1f}%  len={r['audio_length_s']:.2f}s  [{r['category']}]")
        print(f"    REF : {r['reference_normalized']}")
        print(f"    PRED: {r['prediction_normalized']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=None, help="model_id as it appears in the manifest / leaderboard")
    parser.add_argument("--all-models", action="store_true", help="Run for every model_id found in --results-dir")
    parser.add_argument("--results-dir", nargs="+", default=["results/public", "results/private"],
                        help="One or more directories containing JSONL manifests (combined per model_id)")
    parser.add_argument("--category", default=None, help="Restrict analysis to one category")
    parser.add_argument("--top-n-worst", type=int, default=15)
    parser.add_argument("--top-n-subs", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text (single model only)")
    parser.add_argument("--json-out", default=None, help="Write {model_id: report} JSON for all models to this path")
    args = parser.parse_args()

    if not args.model_id and not args.all_models:
        parser.error("pass --model-id or --all-models")

    results_dirs = [Path(d) for d in args.results_dir]

    model_ids = _all_model_ids(results_dirs) if args.all_models else [args.model_id]

    reports: dict[str, dict] = {}
    for model_id in model_ids:
        manifests = _find_manifests(results_dirs, model_id)
        if not manifests:
            print(f"No manifests found for model_id={model_id!r} in {results_dirs}")
            continue
        rows: list[dict] = []
        for m in manifests:
            rows.extend(read_manifest(m))
        if args.category:
            rows = [r for r in rows if r.get("category") == args.category]
        if not rows:
            print(f"No rows match the given filters for {model_id!r}.")
            continue
        reports[model_id] = build_report(model_id, rows, args.top_n_worst, args.top_n_subs)

    if len(reports) > 1:
        for model_id, report in reports.items():
            report["narrative"] = build_narrative(model_id, report, reports)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(reports, indent=2, ensure_ascii=False))
        print(f"Wrote {len(reports)} model report(s) to {args.json_out}")
        return

    for i, (model_id, report) in enumerate(reports.items()):
        if i > 0:
            print("\n" + "=" * 70 + "\n")
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print_report(report)


if __name__ == "__main__":
    main()
