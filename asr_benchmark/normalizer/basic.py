"""Basic multilingual text normalizer — punctuation, tags, case folding."""

from __future__ import annotations

import re
import unicodedata


class BasicTextNormalizer:
    """
    Normalizer for raw transcription text.

    Steps:
      1. Strip spoken-noise/event tags: <noise>, [laughter], (inaudible), etc.
      2. Lowercase
      3. Remove punctuation (keep word chars + whitespace)
      4. Collapse whitespace
    """

    # Matches <tag>, [tag], (tag) — event/noise markers common in ASR refs
    _TAG_RE = re.compile(r"<[^>]+>|\[[^\]]+\]|\([^)]+\)")

    def __init__(self, remove_diacritics: bool = False) -> None:
        self.remove_diacritics = remove_diacritics

    def __call__(self, text: str) -> str:
        text = self._TAG_RE.sub(" ", text)
        text = text.lower()
        # Replace hyphens/dashes with space so "kesatuan-kesatuan" → "kesatuan kesatuan"
        text = re.sub(r"[-–—]", " ", text)
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
        if self.remove_diacritics:
            text = "".join(
                c for c in unicodedata.normalize("NFD", text)
                if unicodedata.category(c) != "Mn"
            )
        text = re.sub(r"\s+", " ", text).strip()
        return text
