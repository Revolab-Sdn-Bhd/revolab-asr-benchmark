"""Revolab STT API runner (aisyah-1.0-flash / aisyah-1.0-pro).

Two modes:
  Cloud (multipart + Bearer):  REVOLAB_API_KEY set, no REVOLAB_LOCAL_URL
  Local (raw binary, no auth): REVOLAB_LOCAL_URL set (overrides cloud)
"""

from __future__ import annotations

import io
import os
import time

import numpy as np
import requests

from .base import BaseASRModel, TranscriptionResult

CLOUD_STT_URL = "https://api-revovoice-local.revocall-staging.com/v1/stt"


class RevolabAPIModel(BaseASRModel):
    """
    Cloud mode — multipart POST with Bearer auth:
        curl -X POST https://api-revovoice-local.revocall-staging.com/v1/stt \\
             -H "Authorization: Bearer $REVOLAB_API_KEY" \\
             -F "file=@audio.wav" -F "model=aisyah-1.0-flash" -F "language=ms"

    Local mode — raw binary POST (no auth), set REVOLAB_LOCAL_URL:
        curl -X POST http://100.88.39.80:6009/recognize \\
             -H "Content-Type: audio/wav" --data-binary "@audio.wav"

    model_id: "aisyah-1.0-flash" or "aisyah-1.0-pro" or "aisyah-1.0-turbo"
    """

    def _load_model(self) -> None:
        local_url = self.kwargs.get("local_url") or os.environ.get("REVOLAB_LOCAL_URL")
        if local_url:
            self._url = local_url
            self._mode = "local"
            self._headers = {"Content-Type": "audio/wav"}
        else:
            api_key = self.kwargs.get("api_key") or os.environ.get("REVOLAB_API_KEY")
            if not api_key:
                raise ValueError(
                    "Set REVOLAB_API_KEY (cloud) or REVOLAB_LOCAL_URL (local) in .env"
                )
            self._url = self.kwargs.get("base_url") or CLOUD_STT_URL
            self._mode = "cloud"
            self._headers = {"Authorization": f"Bearer {api_key}"}

        self._timeout = self.kwargs.get("timeout", 120)
        self._session = requests.Session()

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
                if self._mode == "local":
                    response = self._session.post(
                        self._url,
                        headers=self._headers,
                        data=wav_bytes,
                        timeout=self._timeout,
                    )
                else:
                    response = self._session.post(
                        self._url,
                        headers=self._headers,
                        files={"file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")},
                        data={"model": self.model_id, "language": self.language or "ms"},
                        timeout=self._timeout,
                    )
                if response.status_code == 429:
                    try:
                        msg = response.json().get("error", {}).get("message", response.text)
                    except Exception:
                        msg = response.text
                    raise RuntimeError(f"Revolab API quota/rate limit: {msg}")
                response.raise_for_status()
                payload = response.json()
                prediction = (
                    payload.get("text")
                    or payload.get("transcript")
                    or payload.get("transcription")
                    or ""
                ).strip()
            except (requests.RequestException, ValueError) as e:
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
