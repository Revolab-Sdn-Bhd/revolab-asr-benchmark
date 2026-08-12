"""Whisper runner via HuggingFace transformers (supports any openai/whisper-* checkpoint)."""

from __future__ import annotations

import numpy as np

from .base import BaseASRModel, TranscriptionResult


class WhisperModel(BaseASRModel):
    """
    Example model_ids:
        openai/whisper-large-v3
        openai/whisper-medium.en
        openai/whisper-small
    """

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)

        generate_kwargs={
        "language": "malay",
        }

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
            generate_kwargs=generate_kwargs,
        )

    def transcribe_batch(
        self,
        audio_arrays: list[np.ndarray],
        sample_rates: list[int],
        audio_lengths_s: list[float],
    ) -> list[TranscriptionResult]:
        inputs = [{"array": a, "sampling_rate": sr} for a, sr in zip(audio_arrays, sample_rates)]

        short_idx = [i for i, d in enumerate(audio_lengths_s) if d <= 30.0]
        long_idx = [i for i, d in enumerate(audio_lengths_s) if d > 30.0]

        results: list[tuple[int, str, float]] = []  # (original_idx, text, elapsed)

        if short_idx:
            short_inputs = [inputs[i] for i in short_idx]
            out, elapsed = self._timed_call(self._pipe, short_inputs, batch_size=len(short_inputs))
            per = elapsed / len(short_inputs)
            for i, o in zip(short_idx, out):
                results.append((i, o["text"].strip(), per))

        if long_idx:
            long_inputs = [inputs[i] for i in long_idx]
            out, elapsed = self._timed_call(
                self._pipe, long_inputs, batch_size=len(long_inputs),
                chunk_length_s=30,
            )
            per = elapsed / len(long_inputs)
            for i, o in zip(long_idx, out):
                results.append((i, o["text"].strip(), per))

        results.sort(key=lambda x: x[0])
        return [
            TranscriptionResult(
                prediction=text,
                audio_length_s=audio_lengths_s[i],
                transcription_time_s=elapsed,
            )
            for i, text, elapsed in results
        ]
