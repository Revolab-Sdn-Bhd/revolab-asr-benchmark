"""Qwen3-ASR runner via the qwen_asr package."""

from __future__ import annotations

import numpy as np

from .base import BaseASRModel, TranscriptionResult

# Default language passed to model.transcribe() — Qwen3-ASR uses full language names
DEFAULT_LANGUAGE = "Malay"

# Map BCP-47 / ISO codes → Qwen3-ASR full language names
_LANG_MAP: dict[str, str] = {
    "ms": "Malay",
    "en": "English",
    "zh": "Chinese",
    "id": "Indonesian",
    "ta": "Tamil",
    "ar": "Arabic",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
}


class QwenASRModel(BaseASRModel):
    """
    Qwen3-ASR (1.7B / 0.6B) and any fine-tuned variant published on HuggingFace.

    Requires: pip install qwen-asr

    model_ids:
        Qwen/Qwen3-ASR-1.7B
        Qwen/Qwen3-ASR-0.6B
        Revolab/Malaysian-Qwen3-ASR-1.7B   (or any merged/LoRA checkpoint path)

    language kwarg accepts full language name strings e.g. "Malay", "English".
    """

    def _load_model(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        use_bf16 = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] >= 8
        )
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        device_map = "auto" if torch.cuda.is_available() else "cpu"

        self._model = Qwen3ASRModel.from_pretrained(
            self.model_id,
            # revision='staging',
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=self.kwargs.get("max_inference_batch_size", 4),
            max_new_tokens=self.kwargs.get("max_new_tokens", 256),
        )
        # Resolve BCP-47 code or full name → Qwen3-ASR full language name
        lang = self.language or "ms"
        self._language = _LANG_MAP.get(lang.lower(), lang)

    def transcribe_batch(
        self,
        audio_arrays: list[np.ndarray],
        sample_rates: list[int],
        audio_lengths_s: list[float],
    ) -> list[TranscriptionResult]:
        # qwen_asr.transcribe() accepts a list — the library handles internal
        # batching via max_inference_batch_size set at load time
        audio_inputs = [(a.astype(np.float32), sr) for a, sr in zip(audio_arrays, sample_rates)]

        out, elapsed = self._timed_call(
            self._model.transcribe,
            audio=audio_inputs,
            language=self._language,
            return_time_stamps=False,
        )

        # distribute elapsed proportionally by audio length for per-sample RTFx
        total_dur = sum(audio_lengths_s) or 1.0
        results = []
        for transcription, dur in zip(out, audio_lengths_s):
            results.append(
                TranscriptionResult(
                    prediction=transcription.text.strip() if transcription else "",
                    audio_length_s=dur,
                    transcription_time_s=elapsed * (dur / total_dur),
                    metadata={"detected_language": getattr(transcription, "language", None)},
                )
            )
        return results
