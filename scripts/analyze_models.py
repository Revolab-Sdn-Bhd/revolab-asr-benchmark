#!/usr/bin/env python3
"""
Analyze model behavior from benchmark results using an LLM.

Produces model_analysis.json with strong/weak bullet points per model,
each backed by a specific sample from the benchmark as evidence.

Usage:
    set -a && source .env && set +a
    python scripts/analyze_models.py
    python scripts/analyze_models.py --provider claude --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_benchmark.utils.manifest import read_manifest
from error_analysis import build_report
from dotenv import load_dotenv
load_dotenv()

# ── Data loading ───────────────────────────────────────────────────────────────

def load_model_data(dirs: list[Path]) -> dict[str, list[dict]]:
    model_data: dict[str, list[dict]] = defaultdict(list)
    for d in dirs:
        if not d.exists():
            continue
        for path in sorted(d.glob("*.jsonl")):
            for r in read_manifest(path):
                model_data[r["model_id"]].append(r)
    return dict(model_data)


# ── Stats ──────────────────────────────────────────────────────────────────────

def compute_overall_stats(records: list[dict]) -> dict:
    total_err = total_ref = total_sub = total_ins = total_del = 0
    for r in records:
        s = r.get("word_substitutions", 0)
        i = r.get("word_insertions", 0)
        d = r.get("word_deletions", 0)
        n = r.get("num_ref_words", 0)
        total_err += s + i + d
        total_ref += n
        total_sub += s
        total_ins += i
        total_del += d

    wer = total_err / total_ref * 100 if total_ref else 0
    return {
        "wer":      round(wer, 2),
        "sub_rate": round(total_sub / total_ref * 100, 2) if total_ref else 0,
        "ins_rate": round(total_ins / total_ref * 100, 2) if total_ref else 0,
        "del_rate": round(total_del / total_ref * 100, 2) if total_ref else 0,
        "n":        len(records),
    }


def compute_category_stats(records: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, dict] = defaultdict(lambda: {
        "errors": 0, "ref": 0, "sub": 0, "ins": 0, "del": 0, "samples": []
    })

    for r in records:
        cat = r.get("category") or "unknown"
        s = r.get("word_substitutions", 0)
        i = r.get("word_insertions", 0)
        d = r.get("word_deletions", 0)
        n = r.get("num_ref_words", 0)
        wer = (s + i + d) / max(n, 1) * 100

        by_cat[cat]["errors"] += s + i + d
        by_cat[cat]["ref"]    += n
        by_cat[cat]["sub"]    += s
        by_cat[cat]["ins"]    += i
        by_cat[cat]["del"]    += d
        by_cat[cat]["samples"].append({
            "ref":        r["reference"],
            "pred":       r.get("prediction", ""),
            "ref_norm":   r.get("reference_normalized_final", ""),
            "pred_norm":  r.get("prediction_normalized", ""),
            "split":      r.get("split", ""),
            "wer":        round(wer, 2),
        })

    stats: dict[str, dict] = {}
    for cat, data in by_cat.items():
        if data["ref"] < 10:
            continue
        corpus_wer = data["errors"] / data["ref"] * 100
        sorted_s = sorted(data["samples"], key=lambda x: x["wer"])
        # pick a representative worst sample — not top-1 (often pathological), take p90
        p90_idx = int(len(sorted_s) * 0.9)
        stats[cat] = {
            "wer":          round(corpus_wer, 2),
            "sub_rate":     round(data["sub"] / data["ref"] * 100, 2) if data["ref"] else 0,
            "ins_rate":     round(data["ins"] / data["ref"] * 100, 2) if data["ref"] else 0,
            "del_rate":     round(data["del"] / data["ref"] * 100, 2) if data["ref"] else 0,
            "n_samples":    len(data["samples"]),
            "samples":      sorted_s,
            "best_sample":  sorted_s[0],
            "worst_sample": sorted_s[p90_idx],
        }
    return stats


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM = (
    "You are an expert in speech recognition and Malaysian/Malay language processing. "
    "Your job is to write sharp, evidence-backed analysis of ASR model accuracy for a "
    "product team deciding which model to deploy. Audience ranges from engineers to "
    "non-technical stakeholders, so keep language clear and concrete — no jargon without "
    "a plain-English explanation. Every claim must be grounded in the numbers provided. "
    "Return only valid JSON — no markdown fences, no commentary outside the JSON."
)


def _sanitize(text: str, maxlen: int = 80) -> str:
    """Strip characters that would break JSON strings in the LLM output."""
    return text[:maxlen].replace('"', "'").replace('\n', ' ').replace('\\', '')


def _sample_lines(samples: list[dict], n: int = 3, worst: bool = False) -> str:
    pool = sorted(samples, key=lambda x: x["wer"], reverse=worst)
    chosen = pool[:n]
    lines = []
    for s in chosen:
        ref  = _sanitize(s["ref_norm"])
        pred = _sanitize(s["pred_norm"])
        lines.append(f'  REF:  {ref}\n  HYP:  {pred}\n  WER: {s["wer"]:.0f}%')
    return "\n\n".join(lines)


def _pattern_section(pattern_report: dict) -> str:
    """Render systematic-bug findings from error_analysis.py for the prompt."""
    lines = []

    sm = pattern_report["script_mismatch"]
    if sm["count"]:
        lines.append(f"SCRIPT-MISMATCH HALLUCINATION: {sm['count']} rows output non-Latin script "
                      f"(Chinese/Arabic/etc.) despite Malay/English audio — a language-ID/decoding collapse, "
                      f"not a normal ASR miss.")
        for ex in sm["examples"][:3]:
            lines.append(f'  len={ex["audio_length_s"]:.2f}s  REF: "{ex["reference_normalized"]}"  PRED: "{ex["prediction_normalized"]}"')

    ep = pattern_report["empty_predictions"]
    if ep["count"]:
        top_cats = ", ".join(f"{c} ({n})" for c, n in list(ep["by_category"].items())[:3])
        lines.append(f"BLANK OUTPUT: {ep['count']} rows ({ep['pct_of_rows']}% of all samples) got NO prediction at "
                      f"all despite non-trivial reference audio — a hard failure (timeout/silence/refusal), not a "
                      f"transcription mistake. Worst in: {top_cats}.")

    fd = pattern_report["filler_word_deletions"]
    if fd["count"]:
        lines.append(f"FILLER-WORD DELETION (counts against WER, always a weakness — the benchmark references keep "
                      f"filler words, so dropping them is scored as a miss, not a cleanup): {fd['count']} of "
                      f"{pattern_report['overall']['deletions']} total deletions ({fd['pct_of_total_deletions']}%) "
                      f"are dropped filler words (ah/um/eh/erm).")

    redup = pattern_report["reduplication_merges"]
    if redup["count"]:
        examples_str = ", ".join(f'"{e["word"]} {e["word"]}"->"{e["merged"]}"' for e in redup["examples"][:3])
        lines.append(f"REDUPLICATION MERGING: {redup['count']} instances of reduplicated Malay words "
                      f"collapsed into one token (e.g. {examples_str}).")

    rep = pattern_report["repetition_hallucinations"]
    if rep["count"]:
        lines.append(f"REPETITION HALLUCINATION: {rep['count']} instances of the model repeating a word "
                      f"2+ times that wasn't repeated in the reference.")

    if not lines:
        return "None detected — no script-mismatch hallucination, no unusual reduplication/repetition patterns."
    return "\n".join(lines)


def build_prompt(model_id: str, overall: dict, cat_stats: dict[str, dict],
                 all_model_wers: dict[str, float], pattern_report: dict) -> str:
    ranked = sorted(cat_stats.items(), key=lambda x: x[1]["wer"])
    best_cat  = ranked[0][0]
    worst_cat = ranked[-1][0]

    cat_table = "\n".join(
        f"  {cat:25s} WER={d['wer']:5.1f}%  Sub={d['sub_rate']:.1f}%  Ins={d['ins_rate']:.1f}%  Del={d['del_rate']:.1f}%  (n={d['n_samples']})"
        for cat, d in ranked
    )

    # Peer comparison — rank this model among others
    sorted_peers = sorted(all_model_wers.items(), key=lambda x: x[1])
    rank = next((i+1 for i, (m, _) in enumerate(sorted_peers) if m == model_id), None)
    peer_lines = "\n".join(
        f"  {'>>> ' if m == model_id else '    '}{m}: {w:.1f}%"
        for m, w in sorted_peers
    )

    best_examples  = _sample_lines(cat_stats[best_cat]["samples"],  n=2, worst=False)
    worst_examples = _sample_lines(cat_stats[worst_cat]["samples"], n=2, worst=True)
    pattern_section = _pattern_section(pattern_report)

    return f"""You are analyzing ASR model "{model_id}" evaluated on the Malaysian Speech Benchmark — a multi-domain Malay speech dataset covering {len(cat_stats)} real-world categories (e.g. customer service calls, medical dictation, news, casual conversation, etc.).

Error types explained for reference:
- WER (Word Error Rate): % of words that are wrong — lower is better.
- Sub (substitution): model said a different word (e.g. "beli" → "bela").
- Ins (insertion): model hallucinated extra words that weren't spoken.
- Del (deletion): model missed words that were spoken.

═══ OVERALL ACCURACY — {overall['n']} samples ═══
WER: {overall['wer']:.1f}%  |  Sub: {overall['sub_rate']:.1f}%  Ins: {overall['ins_rate']:.1f}%  Del: {overall['del_rate']:.1f}%

═══ PEER RANKING (overall WER, lower = better) ═══
{peer_lines}
→ This model ranks {rank} out of {len(sorted_peers)}

═══ ACCURACY BY CATEGORY (sorted best → worst) ═══
{cat_table}

═══ BEST CATEGORY SAMPLES: {best_cat} ═══
{best_examples}

═══ WORST CATEGORY SAMPLES: {worst_cat} ═══
{worst_examples}

═══ SYSTEMATIC PATTERNS DETECTED (automated error analysis, not category WER) ═══
{pattern_section}

Return a JSON object with this exact structure:
{{
  "strong": [
    {{"text": "<phrase>", "category": "<exact category name>", "pattern": "<optional, see below>"}},
    ...
  ],
  "weak": [
    {{"text": "<phrase>", "category": "<exact category name>", "pattern": "<optional, see below>"}},
    ...
  ]
}}

"pattern" is optional — set it to one of "script_mismatch", "empty_predictions", "filler_word_deletions",
"reduplication_merges", "repetition_hallucinations" ONLY when the point is directly about
that finding from the SYSTEMATIC PATTERNS section above (omit the field entirely otherwise).

Hard rules — violating any of these makes the output useless:
- 3–5 strong, 3–5 weak points
- "category" must exactly match one of the category names in the table above
- If SYSTEMATIC PATTERNS lists anything with count > 0, it MUST produce at least one weak
  point (with the matching "pattern" field) — these are more informative than generic
  category WER call-outs and take priority over them. Every listed pattern (script-mismatch,
  filler-word deletion, reduplication merging, repetition hallucination) is BY DEFINITION a
  weakness that costs WER — never reframe one as a strength. If patterns section says "None
  detected", say so as a strong point instead (e.g. "no script-mismatch hallucination unlike some peers").
- Each "text" is ≤ 12 words. Count them. If you exceed 12 words, cut.
- Lead with the number, then the one-word verdict. No filler.

Good examples (≤12 words each):
  "#1 overall at 7.2% WER — beats all 9 peers"         ← 11 words ✓
  "read-speech: 1.7% WER — near-perfect"                ← 6 words ✓
  "singing: 16.5% WER, 11.7% sub — mishears everything" ← 8 words ✓
  "insertion spike on singing (2.0%) — hallucinates"    ← 7 words ✓
  "drops 1 in 5 words on telephony (19.2% del)"         ← 10 words ✓

Banned phrases (add zero information — remove them):
  "indicating", "suggesting", "demonstrating", "showcasing",
  "high accuracy for", "performs well", "struggles significantly",
  "suitable for", "showing strong", "showing robustness"

Error type shorthand for the number → implication mapping:
  sub rate high → confuses similar-sounding words
  ins rate high → hallucinates (invents words not spoken)
  del rate high → drops words (misses speech)
"""


# ── LLM callers ────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    return text


def call_llm(prompt: str, provider: str, model: str) -> dict:
    if provider == "gemini":
        return _call_gemini(prompt, model)
    return _call_claude(prompt, model)


def _call_gemini(prompt: str, model: str) -> dict:
    import os
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY env var")

    client = genai.Client(api_key=api_key)
    for attempt in range(3):
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                max_output_tokens=8192,
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        try:
            return json.loads(_strip_fences(response.text))
        except json.JSONDecodeError as exc:
            if attempt == 2:
                raise
            print(f"    retry {attempt+1}/3 — bad JSON: {exc}")


def _call_claude(prompt: str, model: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(_strip_fences(msg.content[0].text))


# ── Evidence attachment ────────────────────────────────────────────────────────

def attach_evidence(points: list[dict], cat_stats: dict[str, dict],
                     pattern_report: dict, use_best: bool) -> None:
    for point in points:
        pattern = point.get("pattern")
        if pattern and pattern in pattern_report:
            examples = pattern_report[pattern].get("examples", [])
            if examples:
                ex = examples[0]
                point["evidence"] = {
                    "ref":   ex["reference"],
                    "split": ex["split"],
                    "wer":   ex["wer"],
                }
                continue
        cat = point.get("category", "")
        if cat not in cat_stats:
            continue
        sample = cat_stats[cat]["best_sample"] if use_best else cat_stats[cat]["worst_sample"]
        point["evidence"] = {
            "ref":   sample["ref"],
            "split": sample["split"],
            "wer":   sample["wer"],
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze model behaviors using an LLM")
    parser.add_argument("--results-dirs", nargs="+", default=["results/public", "results/private"],
                        help="Directories containing .jsonl manifests (default: results/public + results/private)")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "claude"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None,
                        help="Output path (default: first results-dir/model_analysis.json)")
    args = parser.parse_args()

    if args.model is None:
        args.model = "gemini-2.5-flash" if args.provider == "gemini" else "claude-sonnet-4-6"

    root = Path(__file__).resolve().parent.parent
    dirs = [root / d for d in args.results_dirs]
    output_path = Path(args.output) if args.output else dirs[0] / "model_analysis.json"

    print(f"Provider: {args.provider}  Model: {args.model}")
    print(f"Reading from: {', '.join(str(d) for d in dirs)}")

    model_data = load_model_data(dirs)
    print(f"Found {len(model_data)} models: {', '.join(sorted(model_data))}\n")

    # Pre-compute overall WER per model for peer comparison
    all_model_wers = {
        mid: compute_overall_stats(recs)["wer"]
        for mid, recs in model_data.items()
    }

    analysis: dict[str, dict] = {}

    for model_id, records in sorted(model_data.items()):
        print(f"Analyzing {model_id}  ({len(records)} samples) …")
        overall   = compute_overall_stats(records)
        cat_stats = compute_category_stats(records)
        pattern_report = build_report(model_id, records, top_n_worst=15, top_n_subs=20)

        print(f"  Overall WER={overall['wer']:.1f}%  Sub={overall['sub_rate']:.1f}%  "
              f"Ins={overall['ins_rate']:.1f}%  Del={overall['del_rate']:.1f}%  "
              f"categories={len(cat_stats)}")

        if len(cat_stats) < 2:
            print("  Skipped — not enough category variety")
            continue

        try:
            result = call_llm(
                build_prompt(model_id, overall, cat_stats, all_model_wers, pattern_report),
                provider=args.provider,
                model=args.model,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        attach_evidence(result.get("strong", []), cat_stats, pattern_report, use_best=True)
        attach_evidence(result.get("weak",   []), cat_stats, pattern_report, use_best=False)

        analysis[model_id] = {
            "model_id":       model_id,
            "category_stats": {k: {"wer": v["wer"], "n": v["n_samples"]} for k, v in cat_stats.items()},
            "overall":        overall,
            "strong":         result.get("strong", []),
            "weak":           result.get("weak",   []),
        }
        print(f"  ✓  {len(result.get('strong', []))} strong, {len(result.get('weak', []))} weak")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
