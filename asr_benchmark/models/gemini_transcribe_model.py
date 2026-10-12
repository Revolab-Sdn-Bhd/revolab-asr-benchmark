"""Google Gemini dedicated transcription models (e.g. gemini-3.5-transcribe)."""

from __future__ import annotations

import io
import logging
import os
import time

import numpy as np

from .base import BaseASRModel, TranscriptionResult

logger = logging.getLogger(__name__)


class GeminiTranscribeModel(BaseASRModel):
    """
    Gemini transcription-only models via generate_content + AudioTranscriptionConfig.

    Unlike GeminiModel (general multimodal LLM prompted to transcribe), these
    models take audio only and are configured through AudioTranscriptionConfig.
    Word timestamps and diarization are disabled so the output is plain text.

    Requires:
        pip install -r requirements/gemini.txt
        GEMINI_API_KEY env variable (or pass api_key kwarg)

    Env overrides:
        GEMINI_TRANSCRIBE_LANGS  comma-separated BCP-47 hints ("" = auto-detect)
        GEMINI_TRANSCRIBE_MODE   VERBATIM (default) or SMART

    Recommended model_ids:
        gemini-3.5-transcribe
    """

    def _load_model(self) -> None:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai required — run: pip install -r requirements/gemini.txt"
            ) from e

        api_key = self.kwargs.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key=..."
            )
        self._client = genai.Client(api_key=api_key)

        from google.genai import types

        langs_env = os.environ.get("GEMINI_TRANSCRIBE_LANGS")
        if langs_env is None:
            langs = [self.language] if self.language else []
        else:
            langs = [l.strip() for l in langs_env.split(",") if l.strip()]
        mode = os.environ.get("GEMINI_TRANSCRIBE_MODE", "VERBATIM")

        self._config = types.GenerateContentConfig(
            audio_transcription_config=types.AudioTranscriptionConfig(
                language_codes=langs or None,
                word_timestamp=False,
                diarization=False,
                mode=mode,
            ),
        )
        logger.info("Gemini transcribe config: langs=%s mode=%s", langs or "auto", mode)

    def transcribe_batch(
        self,
        audio_arrays: list[np.ndarray],
        sample_rates: list[int],
        audio_lengths_s: list[float],
    ) -> list[TranscriptionResult]:
        from google.genai import types

        results = []
        for audio, sr, dur in zip(audio_arrays, sample_rates, audio_lengths_s):
            wav_bytes = self._to_wav_bytes(audio, sr)
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")],
                )
            ]

            t0 = time.perf_counter()
            try:
                response = self._generate_with_retry(contents)
                prediction = self._extract_text(response)
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.warning("Gemini transcribe failed (%.1fs), skipping row: %s", dur, e)
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

    def _generate_with_retry(self, contents, max_attempts: int = 5):
        # Retry transient server errors (429 / 5xx) with exponential backoff
        from google.genai import errors

        for attempt in range(max_attempts):
            try:
                return self._client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=self._config,
                )
            except errors.APIError as e:
                transient = e.code == 429 or (e.code or 0) >= 500
                if not transient or attempt == max_attempts - 1:
                    raise
                wait = 2 ** attempt
                logger.info("Gemini %s, retrying in %ds (%d/%d)", e.code, wait, attempt + 1, max_attempts)
                time.sleep(wait)

    @staticmethod
    def _extract_text(response) -> str:
        # Transcript lives in parts[].audio_transcription.text, not response.text
        # (response.text is None unless word timestamps/diarization are on).
        segments = []
        for cand in response.candidates or []:
            for part in (cand.content.parts if cand.content else None) or []:
                at = getattr(part, "audio_transcription", None)
                text = (at.text if at else None) or part.text
                if text:
                    segments.append(text.strip())
            break  # first candidate only
        return " ".join(segments).strip()

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
