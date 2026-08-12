"""Abstract base class all ASR model runners must implement."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptionResult:
    prediction: str
    audio_length_s: float
    transcription_time_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False

    @property
    def rtfx(self) -> float:
        """Real-time factor inverse: audio_length / transcription_time."""
        if self.transcription_time_s == 0:
            return float("inf")
        return self.audio_length_s / self.transcription_time_s


class BaseASRModel(ABC):
    """
    All model runners inherit this class and implement `transcribe_batch`.

    Args:
        model_id: Canonical model identifier (used in result manifests).
        language: BCP-47 language tag, e.g. "en". None = auto-detect.
        **kwargs: Model-specific keyword arguments.
    """

    model_id: str

    def __init__(self, model_id: str, language: str | None = "en", **kwargs: Any) -> None:
        self.model_id = model_id
        self.language = language
        self.kwargs = kwargs
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Instantiate model weights / API client. Called once at init."""

    @abstractmethod
    def transcribe_batch(
        self,
        audio_arrays: list[Any],
        sample_rates: list[int],
        audio_lengths_s: list[float],
    ) -> list[TranscriptionResult]:
        """
        Transcribe a batch of audio samples.

        Args:
            audio_arrays: List of numpy arrays (float32, mono).
            sample_rates: Corresponding sample rates in Hz.
            audio_lengths_s: Audio duration in seconds for each sample.

        Returns:
            List of TranscriptionResult, one per input.
        """

    def transcribe_single(
        self,
        audio_array: Any,
        sample_rate: int,
        audio_length_s: float,
    ) -> TranscriptionResult:
        """Convenience wrapper for a single audio sample."""
        results = self.transcribe_batch([audio_array], [sample_rate], [audio_length_s])
        return results[0]

    # ------------------------------------------------------------------
    # Helper: timing context
    # ------------------------------------------------------------------

    @staticmethod
    def _timed_call(fn, *args, **kwargs) -> tuple[Any, float]:
        """Run fn(*args, **kwargs) and return (result, elapsed_seconds)."""
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        return result, time.perf_counter() - t0
