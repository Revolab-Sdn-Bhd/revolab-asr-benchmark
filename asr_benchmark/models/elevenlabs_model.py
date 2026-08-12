"""ElevenLabs Speech-to-Text runner."""

from __future__ import annotations

import io
import os
import re
import time

import numpy as np

from .base import BaseASRModel, TranscriptionResult

# ElevenLabs uses ISO 639-3 codes; map common 2-letter codes
_LANG_MAP = {
    "ms": "msa",  # Malay
    "en": "eng",
    "zh": "zho",
    "id": "ind",
}


def _map_language(lang: str | None) -> str:
    if not lang:
        return "msa"
    return _LANG_MAP.get(lang, lang)


# Strip audio event tags like <laughter>, [music], (applause) from transcript
_EVENT_RE = re.compile(r"[\[\(<][^\]\)>]{1,40}[\]\)>]")


class ElevenLabsModel(BaseASRModel):
    """
    ElevenLabs Speech-to-Text API (scribe_v2).

    Requires:
        pip install elevenlabs
        ELEVENLABS_API_KEY env variable

    model_id: "scribe_v2" (default) or "scribe_v1"
    """

    def _load_model(self) -> None:
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as e:
            raise ImportError(
                "elevenlabs SDK required — run: pip install -r requirements/elevenlabs.txt"
            ) from e

        api_key = self.kwargs.get("api_key") or os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError(
                "ElevenLabs API key required. Set ELEVENLABS_API_KEY in .env or pass api_key=..."
            )
        self._client = ElevenLabs(api_key=api_key)
        self._lang = _map_language(self.language)

    def transcribe_batch(
        self,
        audio_arrays: list[np.ndarray],
        sample_rates: list[int],
        audio_lengths_s: list[float],
    ) -> list[TranscriptionResult]:
        results = []
        for audio, sr, dur in zip(audio_arrays, sample_rates, audio_lengths_s):
            buf = io.BytesIO(self._to_wav_bytes(audio, sr))
            buf.name = "audio.wav"

            t0 = time.perf_counter()
            response = self._client.speech_to_text.convert(
                file=buf,
                model_id=self.model_id,
                language_code=self._lang,
                tag_audio_events=False,  # keep transcript clean for WER
                diarize=False,
            )
            elapsed = time.perf_counter() - t0

            raw = response.text.strip() if hasattr(response, "text") else ""
            # Remove any residual event tags even with tag_audio_events=False
            prediction = _EVENT_RE.sub("", raw).strip()

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
