"""JSONL manifest read/write — one record per audio sample."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_manifest(
    path: str | Path,
    records: list[dict[str, Any]],
    *,
    append: bool = False,
) -> None:
    """
    Write evaluation records to a JSONL file.

    Each record should contain at minimum:
        reference, prediction, audio_length_s, transcription_time_s

    Args:
        path: Output file path (created with parent dirs).
        records: List of dicts to serialize.
        append: If True, append to existing file; else overwrite.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL manifest file and return a list of dicts."""
    path = Path(path)
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
