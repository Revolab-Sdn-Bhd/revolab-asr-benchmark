"""HuggingFace dataset loader with audio preprocessing."""

from __future__ import annotations

from typing import Generator

import numpy as np


def _decode_audio(audio_data, target_sr: int) -> tuple[np.ndarray, int]:
    """
    Decode an audio field from a HuggingFace dataset row.

    Handles two formats:
    - Standard dict: {"array": np.ndarray, "sampling_rate": int}
    - torchcodec AudioDecoder: used by newer HF datasets (e.g. Revolab/ASR-Benchmark-test)
    """
    if isinstance(audio_data, dict):
        array = audio_data["array"].astype(np.float32)
        sr = audio_data["sampling_rate"]
        if sr != target_sr:
            array = _resample(array, sr, target_sr)
        return array, target_sr

    # torchcodec AudioDecoder
    samples = audio_data.get_all_samples()
    data = samples.data
    sr = int(samples.sample_rate)
    try:
        array = data.numpy()
    except Exception:
        array = data.cpu().numpy()
    if array.ndim == 2:
        array = array.mean(axis=0)
    array = array.astype(np.float32)
    if sr != target_sr:
        array = _resample(array, sr, target_sr)
    return array, target_sr


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    try:
        import resampy
        return resampy.resample(audio, orig_sr, target_sr)
    except ImportError:
        pass
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        pass
    n = int(len(audio) * target_sr / orig_sr)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def load_dataset_hf(
    dataset_name: str,
    split: str = "test",
    subset: str | None = None,
    audio_column: str = "audio",
    text_column: str = "sentence",
    metadata_columns: list[str] | None = None,
    target_sample_rate: int = 16000,
    max_samples: int | None = None,
    streaming: bool = True,
) -> Generator[tuple[np.ndarray, int, float, str, dict], None, None]:
    """
    Yield (audio_array, sample_rate, duration_s, reference_text, metadata) tuples.

    Args:
        metadata_columns: Extra columns to carry through (e.g. ["category"]).
                          Values are returned in the metadata dict per sample.

    Yields:
        (audio_array float32, sample_rate, duration_s, reference_text, metadata_dict)
    """
    from datasets import load_dataset

    meta_cols = metadata_columns or []
    load_kwargs: dict = dict(split=split, streaming=streaming)
    if subset:
        ds = load_dataset(dataset_name, subset, **load_kwargs)
    else:
        ds = load_dataset(dataset_name, **load_kwargs)

    first_row = next(iter(ds))
    audio_sample = first_row[audio_column]
    is_standard_audio = isinstance(audio_sample, dict)

    if is_standard_audio:
        from datasets import Audio
        ds = ds.cast_column(audio_column, Audio(sampling_rate=target_sample_rate))

    def _extract(row: dict) -> tuple[np.ndarray, int, float, str, dict]:
        array, sr = _decode_audio(row[audio_column], target_sample_rate)
        duration_s = len(array) / sr
        meta = {col: row[col] for col in meta_cols if col in row}
        return array, sr, duration_s, row[text_column], meta

    count = 0
    started = False
    for row in ds:
        if not started and not is_standard_audio:
            started = True
            yield _extract(first_row)
            count += 1
            if max_samples is not None and count >= max_samples:
                return
            continue

        if not started:
            started = True

        if max_samples is not None and count >= max_samples:
            break

        yield _extract(row)
        count += 1
