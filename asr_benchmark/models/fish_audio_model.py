"""Fish Audio Speech-to-Text runner (REST API)."""

from __future__ import annotations

import io
import logging
import os
import re
import time

import numpy as np

from .base import BaseASRModel, TranscriptionResult

logger = logging.getLogger(__name__)

_ASR_URL = "https://api.fish.audio/v1/asr"

# Strip inline <|speaker:N|> markers and event cues in case any slip through
_MARKER_RE = re.compile(r"<\|[^|]*\|>")


class FishAudioModel(BaseASRModel):
    """
    Fish Audio Speech-to-Text API.

    Requires:
        FISH_API_KEY env variable (or pass api_key kwarg)

    model_id: "transcribe-1-pro" (recommended) or "transcribe-1".
    The model is selected via a `model` HTTP header; the API silently falls back
    to transcribe-1 if the header is missing or misspelled.

    Timestamps, diarization and audio-event tags are disabled so the transcript
    is plain text. The language hint is dropped automatically if the API
    rejects it (Malay support is not documented).
    """

    def _load_model(self) -> None:
        import requests

        api_key = self.kwargs.get("api_key") or os.environ.get("FISH_API_KEY")
        if not api_key:
            raise ValueError(
                "Fish Audio API key required. Set FISH_API_KEY in .env or pass api_key=..."
            )
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "model": self.model_id,
        })
        self._lang = self.language or None
        self._is_pro = self.model_id == "transcribe-1-pro"

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
                data = self._post_with_retry(wav_bytes)
                prediction = _MARKER_RE.sub(" ", data.get("text") or "")
                prediction = re.sub(r"\s+", " ", prediction).strip()
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.warning("Fish Audio failed (%.1fs), skipping row: %s", dur, e)
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

    def _form_fields(self) -> dict[str, str]:
        fields = {"ignore_timestamps": "true"}
        if self._lang:
            fields["language"] = self._lang
        if self._is_pro:
            fields["tag_audio_events"] = "false"
            fields["diarize"] = "false"
        return fields

    def _post_with_retry(self, wav_bytes: bytes, max_attempts: int = 5) -> dict:
        # Retry transient server errors (429 / 5xx) with exponential backoff
        for attempt in range(max_attempts):
            resp = self._session.post(
                _ASR_URL,
                files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
                data=self._form_fields(),
                timeout=300,
            )
            if resp.status_code == 400 and self._lang and "language" in resp.text.lower():
                logger.warning("Fish Audio rejected language=%r, falling back to auto-detect: %s",
                               self._lang, resp.text[:200])
                self._lang = None
                continue
            transient = resp.status_code == 429 or resp.status_code >= 500
            if transient and attempt < max_attempts - 1:
                wait = 2 ** attempt
                logger.info("Fish Audio %s, retrying in %ds (%d/%d)",
                            resp.status_code, wait, attempt + 1, max_attempts)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
