"""Deepgram Speech-to-Text runner."""

from __future__ import annotations

import io
import os
import time

import numpy as np

from .base import BaseASRModel, TranscriptionResult


class DeepgramModel(BaseASRModel):
    """
    Deepgram Speech-to-Text API.

    Requires:
        pip install deepgram-sdk
        DEEPGRAM_API_KEY env variable (or pass api_key kwarg)

    model_id is passed as the `model` parameter to Deepgram.
    Common options: "nova-3", "nova-2", "nova-2-general", "enhanced", "base"
    """

    def _load_model(self) -> None:
        try:
            from deepgram import DeepgramClient
        except ImportError as e:
            raise ImportError(
                "deepgram-sdk required. Install: pip install deepgram-sdk"
            ) from e

        api_key = self.kwargs.get("api_key") or os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError(
                "Deepgram API key required. Set DEEPGRAM_API_KEY env var or pass api_key=..."
            )
        # DeepgramClient reads DEEPGRAM_API_KEY from the environment; set it
        # explicitly so callers can pass api_key= without touching os.environ.
        os.environ.setdefault("DEEPGRAM_API_KEY", api_key)
        self._client = DeepgramClient()
        self._model_name = self.model_id  # e.g. "nova-3"

    def transcribe_batch(
        self,
        audio_arrays: list[np.ndarray],
        sample_rates: list[int],
        audio_lengths_s: list[float],
    ) -> list[TranscriptionResult]:
        results = []
        for audio, sr, dur in zip(audio_arrays, sample_rates, audio_lengths_s):
            wav_bytes = self._to_wav_bytes(audio, sr)

            t0 = time.perf_counter()
            response = self._client.listen.v1.media.transcribe_file(
                request=wav_bytes,
                model=self._model_name,
                language=self.language or "ms",
                smart_format=False,
                punctuate=False,
            )
            elapsed = time.perf_counter() - t0

            try:
                prediction = (
                    response.results.channels[0].alternatives[0].transcript.strip()
                )
            except (AttributeError, IndexError):
                prediction = ""

            results.append(
                TranscriptionResult(
                    prediction=prediction,
                    audio_length_s=dur,
                    transcription_time_s=elapsed,
                )
            )
        return results

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
