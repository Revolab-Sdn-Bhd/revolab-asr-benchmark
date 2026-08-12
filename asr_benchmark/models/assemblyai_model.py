"""AssemblyAI Speech-to-Text runner."""

from __future__ import annotations

import io
import os
import time

import numpy as np

from .base import BaseASRModel, TranscriptionResult

# AssemblyAI language codes use BCP-47 — "ms" works directly for Malay
_LANG_MAP = {
    "ms": "ms",
    "en": "en",
    "zh": "zh",
    "id": "id",
}


class AssemblyAIModel(BaseASRModel):
    """
    AssemblyAI Speech-to-Text API.

    Requires:
        pip install "assemblyai>=1.0.0"
        ASSEMBLYAI_API_KEY env variable

    model_id: "universal-2", "universal-3-pro", "universal-3-5-pro", "best", "nano"
    """

    def _load_model(self) -> None:
        try:
            import assemblyai as aai
        except ImportError as e:
            raise ImportError(
                "assemblyai SDK required — run: pip install -r requirements/assemblyai.txt"
            ) from e

        api_key = self.kwargs.get("api_key") or os.environ.get("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise ValueError(
                "AssemblyAI API key required. Set ASSEMBLYAI_API_KEY in .env or pass api_key=..."
            )
        aai.settings.api_key = api_key
        self._aai = aai

        lang = _LANG_MAP.get(self.language or "ms", self.language)

        # Map model_id to speech model
        model_to_speech_model = {
            "universal-2": ["universal-2"],
            "universal-3": ["universal-3", "universal-2"],
            "universal-3-pro": ["universal-3-pro", "universal-2"],
            "universal-3-5-pro": ["universal-3-5-pro", "universal-3-pro", "universal-2"],
            "best": ["best"],
            "nano": ["nano"],
        }

        speech_models = model_to_speech_model.get(
            self.model_id,
            ["universal-3-5-pro", "universal-3-pro", "universal-2"]  # fallback for unknown model_id
        )

        self._config = aai.TranscriptionConfig(
            speech_models=speech_models,
            language_code=lang,
        )

    def transcribe_batch(
        self,
        audio_arrays: list[np.ndarray],
        sample_rates: list[int],
        audio_lengths_s: list[float],
    ) -> list[TranscriptionResult]:
        results = []
        transcriber = self._aai.Transcriber()

        for audio, sr, dur in zip(audio_arrays, sample_rates, audio_lengths_s):
            wav_bytes = self._to_wav_bytes(audio, sr)

            t0 = time.perf_counter()
            transcript = transcriber.transcribe(
                io.BytesIO(wav_bytes),
                config=self._config,
            )
            elapsed = time.perf_counter() - t0

            if transcript.status == self._aai.TranscriptStatus.error:
                prediction = ""
            else:
                prediction = (transcript.text or "").strip()

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
