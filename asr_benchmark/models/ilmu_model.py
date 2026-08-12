"""Ilmu ASR runner via OpenAI-compatible transcription API."""

from __future__ import annotations

import io
import os
import time

import numpy as np

from .base import BaseASRModel, TranscriptionResult

ILMU_BASE_URL = "https://api.ilmu.ai/v1"


class IlmuModel(BaseASRModel):
    """
    Ilmu ASR via OpenAI-compatible audio transcription endpoint.

    Requires:
        pip install openai
        ILMU_API_KEY env variable

    model_id: "ilmu-asr-v4.2" (or any ilmu model name)

    kwargs:
        api_key: override ILMU_API_KEY env var
        base_url: override API base URL (default: https://api.ilmu.ai/v1)
        timeout: request timeout in seconds (default: 120)
    """

    def _load_model(self) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError("openai SDK required — run: pip install openai") from e

        api_key = self.kwargs.get("api_key") or os.environ.get("ILMU_API_KEY")
        if not api_key:
            raise ValueError("Ilmu API key required. Set ILMU_API_KEY in .env or pass api_key=...")

        base_url = self.kwargs.get("base_url") or ILMU_BASE_URL
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._timeout = self.kwargs.get("timeout", 120)

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
            try:
                response = self._client.audio.transcriptions.create(
                    model=self.model_id,
                    file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
                    timeout=self._timeout,
                )
                prediction = (response.text or "").strip()
            except Exception as e:
                prediction = ""
            elapsed = time.perf_counter() - t0

            results.append(TranscriptionResult(
                prediction=prediction,
                audio_length_s=dur,
                transcription_time_s=elapsed,
            ))
        return results

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
