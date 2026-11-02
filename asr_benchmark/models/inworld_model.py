"""Inworld Speech-to-Text runner (sync REST API)."""

from __future__ import annotations

import base64
import io
import logging
import os
import time

import numpy as np

from .base import BaseASRModel, TranscriptionResult

logger = logging.getLogger(__name__)

_TRANSCRIBE_URL = "https://api.inworld.ai/stt/v1/transcribe"
_TARGET_SR = 16000


class InworldModel(BaseASRModel):
    """
    Inworld STT sync endpoint (/stt/v1/transcribe).

    Requires:
        INWORLD_API_KEY env variable (the Base64 credential from the Inworld
        portal, sent as `Authorization: Basic <key>`)

    model_id: "inworld/inworld-stt-1"
    Audio is sent as 16 kHz mono 16-bit WAV (LINEAR16). `language` is only a
    hint; the model still auto-detects (INWORLD_LANG="" drops it). Voice
    profiling is left off.
    """

    def _load_model(self) -> None:
        import requests

        api_key = self.kwargs.get("api_key") or os.environ.get("INWORLD_API_KEY")
        if not api_key:
            raise ValueError(
                "Inworld API key required. Set INWORLD_API_KEY in .env or pass api_key=..."
            )
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json",
        })
        # INWORLD_LANG overrides the hint; set it to "" to let the model auto-detect
        self._lang = os.environ.get("INWORLD_LANG", self.language) or None

    def transcribe_batch(
        self,
        audio_arrays: list[np.ndarray],
        sample_rates: list[int],
        audio_lengths_s: list[float],
    ) -> list[TranscriptionResult]:
        results = []
        for audio, sr, dur in zip(audio_arrays, sample_rates, audio_lengths_s):
            payload = self._payload(self._to_wav_bytes(audio, sr))

            t0 = time.perf_counter()
            try:
                data = self._post_with_retry(payload)
                prediction = ((data.get("transcription") or {}).get("transcript") or "").strip()
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.warning("Inworld failed (%.1fs), skipping row: %s", dur, e)
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

    def _payload(self, wav_bytes: bytes) -> dict:
        config = {
            "modelId": self.model_id,
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": _TARGET_SR,
        }
        if self._lang:
            config["language"] = self._lang
        return {
            "transcribeConfig": config,
            "audioData": {"content": base64.b64encode(wav_bytes).decode("ascii")},
        }

    def _post_with_retry(self, payload: dict, max_attempts: int = 5) -> dict:
        # Retry transient server errors (429 / 5xx) with exponential backoff
        for attempt in range(max_attempts):
            resp = self._session.post(_TRANSCRIBE_URL, json=payload, timeout=300)
            transient = resp.status_code == 429 or resp.status_code >= 500
            if transient and attempt < max_attempts - 1:
                wait = 2 ** attempt
                logger.info("Inworld %s, retrying in %ds (%d/%d)",
                            resp.status_code, wait, attempt + 1, max_attempts)
                time.sleep(wait)
                continue
            if not resp.ok:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        raise RuntimeError("unreachable")

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
        import soundfile as sf
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != _TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)
        buf = io.BytesIO()
        sf.write(buf, audio, _TARGET_SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()
