"""Google Gemini ASR runner via google-genai SDK."""

from __future__ import annotations

import base64
import io
import logging
import os
import time

import numpy as np

from .base import BaseASRModel, TranscriptionResult

logger = logging.getLogger(__name__)

_TRANSCRIBE_PROMPT = (
    "Generate a transcript of the speech."
    # "Return only the transcription text with no commentary, timestamps, or formatting."
)

_LANG_HINTS: dict[str, str] = {
    "ms": "Malay (ms-MY)",
    "en": "English (en-US)",
    "id": "Indonesian (id-ID)",
    "zh": "Chinese (zh-CN)",
}


class GeminiModel(BaseASRModel):
    """
    Google Gemini multimodal model used as ASR via the google-genai SDK.

    Requires:
        pip install -r requirements/gemini.txt
        GEMINI_API_KEY env variable (or pass api_key kwarg)

    Recommended model_ids:
        gemini-2.5-flash
        gemini-2.5-pro
    """

    def _load_model(self) -> None:
        try:
            from google import genai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "google-genai required — run: pip install -r requirements/gemini.txt"
            ) from e

        from google import genai as _genai

        api_key = self.kwargs.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key=..."
            )
        self._client = _genai.Client(api_key=api_key)
        lang = self.language or "ms"
        self._lang_hint = _LANG_HINTS.get(lang, lang)

    def transcribe_batch(
        self,
        audio_arrays: list[np.ndarray],
        sample_rates: list[int],
        audio_lengths_s: list[float],
    ) -> list[TranscriptionResult]:
        results = []
        for audio, sr, dur in zip(audio_arrays, sample_rates, audio_lengths_s):
            wav_bytes = self._to_wav_bytes(audio, sr)
            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

            prompt = _TRANSCRIBE_PROMPT
            # if self._lang_hint:
            #     prompt += f" The language spoken is {self._lang_hint}."

            t0 = time.perf_counter()
            try:
                interaction = self._client.interactions.create(
                    model=self.model_id,
                    input=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "audio",
                            "data": audio_b64,
                            "mime_type": "audio/wav",
                        },
                    ],
                )
                prediction = (interaction.output_text or "").strip()
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.warning("Gemini blocked sample (%.1fs), skipping row: %s", dur, e)
                results.append(TranscriptionResult(
                    prediction="",
                    audio_length_s=dur,
                    transcription_time_s=elapsed,
                    skipped=True,
                ))
                continue
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
