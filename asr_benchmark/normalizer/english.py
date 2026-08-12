"""English text normalizer adapted from the Whisper evaluation normalizer."""

from __future__ import annotations

import re


# Mapping of common English contractions and abbreviations
_SUBSTITUTIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(won't)\b", re.I), "will not"),
    (re.compile(r"\b(can't)\b", re.I), "cannot"),
    (re.compile(r"\b(n't)\b", re.I), " not"),
    (re.compile(r"\b('re)\b", re.I), " are"),
    (re.compile(r"\b('s)\b", re.I), " is"),
    (re.compile(r"\b('d)\b", re.I), " would"),
    (re.compile(r"\b('ll)\b", re.I), " will"),
    (re.compile(r"\b('ve)\b", re.I), " have"),
    (re.compile(r"\b('m)\b", re.I), " am"),
]

_NUMBER_WORDS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}


class EnglishTextNormalizer:
    """
    English-specific normalization pipeline:
      1. Lowercase
      2. Expand contractions
      3. Strip punctuation except apostrophes in possessives
      4. Remove filler words (um, uh, etc.)
      5. Collapse whitespace
    """

    FILLER_WORDS = re.compile(
        r"\b(um|uh|hmm|mm|mhm|huh|ah|er|like|you know|i mean)\b", re.I
    )

    def __call__(self, text: str) -> str:
        text = text.lower()
        for pattern, replacement in _SUBSTITUTIONS:
            text = pattern.sub(replacement, text)
        text = self.FILLER_WORDS.sub("", text)
        # Remove punctuation except apostrophes within words
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
