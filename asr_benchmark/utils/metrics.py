"""WER, CER, and per-operation (substitution, insertion, deletion) metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class EditStats:
    """Raw counts from aligning a reference to a hypothesis sequence."""
    hits: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def ref_length(self) -> int:
        return self.hits + self.substitutions + self.deletions

    def __add__(self, other: "EditStats") -> "EditStats":
        return EditStats(
            hits=self.hits + other.hits,
            substitutions=self.substitutions + other.substitutions,
            insertions=self.insertions + other.insertions,
            deletions=self.deletions + other.deletions,
        )


def _align(ref: list[str], hyp: list[str]) -> EditStats:
    """
    Compute edit distance with backtracking to count S/I/D/H operations.

    Uses the standard DP table; backtrack path determines operation type.
    """
    r, h = len(ref), len(hyp)

    # dp[i][j] = min edit distance between ref[:i] and hyp[:j]
    dp = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        dp[i][0] = i
    for j in range(h + 1):
        dp[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack
    stats = EditStats()
    i, j = r, h
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            stats.hits += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            stats.substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            stats.insertions += 1
            j -= 1
        else:
            stats.deletions += 1
            i -= 1

    return stats


def _corpus_word_stats(references: list[str], hypotheses: list[str]) -> EditStats:
    total = EditStats()
    for ref, hyp in zip(references, hypotheses):
        total = total + _align(ref.split(), hyp.split())
    return total


def _corpus_char_stats(references: list[str], hypotheses: list[str]) -> EditStats:
    total = EditStats()
    for ref, hyp in zip(references, hypotheses):
        total = total + _align(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level Word Error Rate as a percentage."""
    stats = _corpus_word_stats(references, hypotheses)
    if stats.ref_length == 0:
        return 0.0
    return round(100.0 * stats.errors / stats.ref_length, 2)


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level Character Error Rate as a percentage."""
    stats = _corpus_char_stats(references, hypotheses)
    if stats.ref_length == 0:
        return 0.0
    return round(100.0 * stats.errors / stats.ref_length, 2)


def _align_words(
    ref: list[str], hyp: list[str]
) -> list[tuple[str, str | None, str | None]]:
    """
    Word-level alignment; returns one tuple per operation: (op, ref_word, hyp_word).

    op values: 'hit' | 'sub' | 'ins' | 'del'
    ref_word is None for insertions; hyp_word is None for deletions.
    Same DP as _align() — duplicated to avoid coupling the hot aggregate path
    with the slower word-list return path.
    """
    r, h = len(ref), len(hyp)
    dp = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        dp[i][0] = i
    for j in range(h + 1):
        dp[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    ops: list[tuple] = []
    i, j = r, h
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            ops.append(("hit", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
        else:
            ops.append(("del", ref[i - 1], None))
            i -= 1
    return ops


# ---------------------------------------------------------------------------
# Rare-word WER helpers
# ---------------------------------------------------------------------------

def build_freq_map(references: list[str]) -> Counter:
    """Count word frequencies across all reference strings."""
    freq: Counter = Counter()
    for ref in references:
        freq.update(ref.split())
    return freq


def make_common_words(freq_map: Counter, top_n: int) -> frozenset:
    """Return the top_n most-frequent words as a frozen set."""
    return frozenset(w for w, _ in freq_map.most_common(top_n))


def compute_rare_wer(
    refs: list[str],
    hyps: list[str],
    common_words: frozenset,
) -> dict:
    """
    WER restricted to rare words — reference words *not* in common_words.

    Rare WER   = (rare_S + rare_D) / rare_ref_words * 100
    Insertions are excluded: they have no reference word to anchor rarity.

    The companion metric rare_sub_rate isolates substitution errors, which is
    what the blog post by Kaldi/Vosk author describes as the key signal for
    end-to-end model weakness on tail vocabulary.
    """
    rare_ref = rare_hits = rare_subs = rare_dels = 0
    for ref, hyp in zip(refs, hyps):
        for op, rw, _hw in _align_words(ref.split(), hyp.split()):
            if rw is None or rw in common_words:
                continue
            rare_ref += 1
            if op == "hit":
                rare_hits += 1
            elif op == "sub":
                rare_subs += 1
            elif op == "del":
                rare_dels += 1

    def pct(n: int, d: int) -> float:
        return round(100.0 * n / d, 2) if d > 0 else 0.0

    return {
        "rare_wer":           pct(rare_subs + rare_dels, rare_ref),
        "rare_sub_rate":      pct(rare_subs,             rare_ref),
        "rare_del_rate":      pct(rare_dels,             rare_ref),
        "rare_ref_words":     rare_ref,
        "rare_substitutions": rare_subs,
        "rare_deletions":     rare_dels,
    }


def compute_all_metrics(references: list[str], hypotheses: list[str]) -> dict:
    """
    Return a full metrics dict with WER, CER, and word-level S/I/D counts and rates.

    All rates are percentages relative to the total reference word count.
    """
    w = _corpus_word_stats(references, hypotheses)
    c = _corpus_char_stats(references, hypotheses)

    ref_words = w.ref_length
    ref_chars = c.ref_length

    def pct(n: int, denom: int) -> float:
        return round(100.0 * n / denom, 2) if denom > 0 else 0.0

    return {
        # Word-level
        "wer": pct(w.errors, ref_words),
        "word_substitutions": w.substitutions,
        "word_insertions": w.insertions,
        "word_deletions": w.deletions,
        "word_hits": w.hits,
        "num_ref_words": ref_words,
        "substitution_rate": pct(w.substitutions, ref_words),
        "insertion_rate": pct(w.insertions, ref_words),
        "deletion_rate": pct(w.deletions, ref_words),
        # Character-level
        "cer": pct(c.errors, ref_chars),
        "char_substitutions": c.substitutions,
        "char_insertions": c.insertions,
        "char_deletions": c.deletions,
        "char_hits": c.hits,
        "num_ref_chars": ref_chars,
    }
