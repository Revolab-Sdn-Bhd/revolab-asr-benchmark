"""Zipformer runner via sherpa-onnx (CPU/GPU, no k2 build required)."""

from __future__ import annotations

import numpy as np

from .base import BaseASRModel, TranscriptionResult

# Registered HuggingFace checkpoints.
# Each entry maps logical name → HF repo + subfolder + file stems.
# Files are downloaded via hf_hub_download (respects HF_TOKEN / huggingface-cli login).
CHECKPOINT_MAP: dict[str, dict] = {
    # English LibriSpeech — online (streaming) model
    "zipformer-en-2023-06-21": {
        "mode":      "online",
        "repo_id":   "csukuangfj/sherpa-onnx-zipformer-en-2023-06-21",
        "subfolder": "",
        "encoder": "encoder-epoch-99-avg-1.onnx",
        "decoder": "decoder-epoch-99-avg-1.onnx",
        "joiner":  "joiner-epoch-99-avg-1.onnx",
        "tokens":  "tokens.txt",
    },
    # Revolab Malaysian telephony — offline (pruned_transducer_stateless7)
    "revolab-zipformer-ms-telephony": {
        "mode":      "offline",
        "repo_id":   "Revolab/malaysian-pruned_transducer_stateless7",
        "subfolder": "telephony",
        "encoder": "encoder-epoch-15-avg-3.onnx",
        "decoder": "decoder-epoch-15-avg-3.onnx",
        "joiner":  "joiner-epoch-15-avg-3.onnx",
        "tokens":  "tokens.txt",
    },
    # Same model, int8-quantised (faster, lower memory)
    "revolab-zipformer-ms-telephony-int8": {
        "mode":      "offline",
        "repo_id":   "Revolab/malaysian-pruned_transducer_stateless7",
        "subfolder": "telephony",
        "encoder": "encoder-epoch-15-avg-3.int8.onnx",
        "decoder": "decoder-epoch-15-avg-3.int8.onnx",
        "joiner":  "joiner-epoch-15-avg-3.int8.onnx",
        "tokens":  "tokens.txt",
    },
}


class ZipformerModel(BaseASRModel):
    """
    Zipformer transducer via sherpa-onnx.

    model_id options:
      - Key in CHECKPOINT_MAP (auto-downloaded via hf_hub_download):
          revolab-zipformer-ms-telephony
          revolab-zipformer-ms-telephony-int8
          zipformer-en-2023-06-21
      - Local directory path containing encoder.onnx / decoder.onnx /
        joiner.onnx / tokens.txt

    Install: pip install sherpa-onnx huggingface_hub
    """

    def _load_model(self) -> None:
        try:
            import sherpa_onnx
        except ImportError as e:
            raise ImportError(
                "sherpa-onnx is required for ZipformerModel. "
                "Install via: pip install sherpa-onnx"
            ) from e

        import os

        if self.model_id in CHECKPOINT_MAP:
            paths = self._download_checkpoint(self.model_id)
            mode = CHECKPOINT_MAP[self.model_id]["mode"]
        elif os.path.isdir(self.model_id):
            paths = {
                "encoder": os.path.join(self.model_id, "encoder.onnx"),
                "decoder": os.path.join(self.model_id, "decoder.onnx"),
                "joiner":  os.path.join(self.model_id, "joiner.onnx"),
                "tokens":  os.path.join(self.model_id, "tokens.txt"),
            }
            mode = self.kwargs.get("mode", "offline")
        else:
            raise ValueError(
                f"Unknown model_id '{self.model_id}'. "
                f"Use one of {list(CHECKPOINT_MAP)} or a local directory."
            )

        self._mode = mode
        num_threads   = self.kwargs.get("num_threads", 4)
        decoding_method = self.kwargs.get("decoding_method", "greedy_search")

        if mode == "offline":
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=paths["encoder"],
                decoder=paths["decoder"],
                joiner=paths["joiner"],
                tokens=paths["tokens"],
                num_threads=num_threads,
                decoding_method=decoding_method,
            )
        else:
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=paths["encoder"],
                decoder=paths["decoder"],
                joiner=paths["joiner"],
                tokens=paths["tokens"],
                num_threads=num_threads,
                sample_rate=16000,
                feature_dim=80,
                decoding_method=decoding_method,
            )

    def _download_checkpoint(self, name: str) -> dict[str, str]:
        """Download ONNX + tokens via hf_hub_download into HF cache."""
        from huggingface_hub import hf_hub_download

        info = CHECKPOINT_MAP[name]
        repo    = info["repo_id"]
        subdir  = info["subfolder"]
        paths: dict[str, str] = {}

        for key in ("encoder", "decoder", "joiner", "tokens"):
            filename = f"{subdir}/{info[key]}" if subdir else info[key]
            local = hf_hub_download(repo_id=repo, filename=filename)
            paths[key] = local

        return paths

    def transcribe_batch(
        self,
        audio_arrays: list[np.ndarray],
        sample_rates: list[int],
        audio_lengths_s: list[float],
    ) -> list[TranscriptionResult]:
        import time

        results = []
        for audio, sr, dur in zip(audio_arrays, sample_rates, audio_lengths_s):
            if sr != 16000:
                audio = self._resample(audio, sr, 16000)

            audio = audio.astype(np.float32)
            t0 = time.perf_counter()

            if self._mode == "offline":
                stream = self._recognizer.create_stream()
                stream.accept_waveform(sample_rate=16000, waveform=audio)
                self._recognizer.decode_stream(stream)
                prediction = stream.result.text.strip()
            else:
                stream = self._recognizer.create_stream()
                stream.accept_waveform(16000, audio)
                tail_paddings = np.zeros(int(0.3 * 16000), dtype=np.float32)
                stream.accept_waveform(16000, tail_paddings)
                stream.input_finished()
                while self._recognizer.is_ready(stream):
                    self._recognizer.decode_stream(stream)
                prediction = self._recognizer.get_result(stream).text.strip()

            elapsed = time.perf_counter() - t0
            results.append(
                TranscriptionResult(
                    prediction=prediction,
                    audio_length_s=dur,
                    transcription_time_s=elapsed,
                )
            )
        return results

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        try:
            import resampy
            return resampy.resample(audio, orig_sr, target_sr)
        except ImportError:
            n = int(len(audio) * target_sr / orig_sr)
            return np.interp(
                np.linspace(0, len(audio) - 1, n),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
