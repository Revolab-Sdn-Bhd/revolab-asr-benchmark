"""Qwen Audio ASR runner via DashScope multimodal-generation API."""

from __future__ import annotations

import base64
import io
import os
import time

import numpy as np
import requests

from .base import BaseASRModel, TranscriptionResult

DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


class QwenAudioModel(BaseASRModel):
    """
    Qwen Audio ASR via DashScope multimodal-generation API.

    Requires:
        DASHSCOPE_API_KEY env variable
        pip install requests soundfile numpy

    model_id: "qwen-audio-3.0-asr-flash" (or any DashScope audio model)

    kwargs:
        api_key: override DASHSCOPE_API_KEY env var
        base_url: override API URL (default: DashScope intl endpoint)
        timeout: request timeout in seconds (default: 120)
        sample_rate: output sample rate hint (default: "16000")
        format: output format hint (default: "wav")
    """

    def _load_model(self) -> None:
        api_key = self.kwargs.get("api_key") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DashScope API key required. Set DASHSCOPE_API_KEY in .env "
                "or pass api_key=..."
            )

        self._url = self.kwargs.get("base_url") or DASHSCOPE_URL
        self._timeout = self.kwargs.get("timeout", 120)
        self._sample_rate = self.kwargs.get("sample_rate", "16000")
        self._format = self.kwargs.get("format", "wav")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "disable",
        })

    def transcribe_batch(
        self,
        audio_arrays: list[np.ndarray],
        sample_rates: list[int],
        audio_lengths_s: list[float],
    ) -> list[TranscriptionResult]:
        results = []
        for audio, sr, dur in zip(audio_arrays, sample_rates, audio_lengths_s):
            data_uri = self._to_wav_data_uri(audio, sr)

            payload = {
                "model": self.model_id,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {"data": data_uri},
                                }
                            ],
                        }
                    ]
                },
                "parameters": {
                    "format": self._format,
                    "sample_rate": self._sample_rate,
                },
            }

            t0 = time.perf_counter()
            try:
                response = self._session.post(
                    self._url,
                    json=payload,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                data = response.json()

                # DashScope ASR response: {"output":{"text": "...", "sentence": {...}}}
                # "ASR_RESPONSE_HAVE_NO_WORDS" (silence/no speech) yields an empty transcript.
                prediction = (data.get("output", {}).get("text") or "").strip()
            except (requests.RequestException, ValueError, KeyError):
                prediction = ""
            elapsed = time.perf_counter() - t0

            results.append(TranscriptionResult(
                prediction=prediction,
                audio_length_s=dur,
                transcription_time_s=elapsed,
            ))
        return results

    @staticmethod
    def _to_wav_data_uri(audio: np.ndarray, sr: int) -> str:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        wav_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:audio/wav;base64,{wav_b64}"
