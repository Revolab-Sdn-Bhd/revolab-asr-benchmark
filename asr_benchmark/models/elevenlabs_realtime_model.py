"""ElevenLabs realtime (streaming) Speech-to-Text runner over WebSocket."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time

import numpy as np

from .base import BaseASRModel, TranscriptionResult

logger = logging.getLogger(__name__)

_WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
_TARGET_SR = 16000
_CHUNK_S = 0.1  # 100 ms per audio chunk

# Strip audio event tags like <laughter>, [music], (applause) from transcript
_EVENT_RE = re.compile(r"[\[\(<][^\]\)>]{1,40}[\]\)>]")

_ERROR_TYPES = {
    "error", "auth_error", "quota_exceeded", "unaccepted_terms", "rate_limited",
    "queue_overflow", "resource_exhausted", "session_time_limit_exceeded",
    "input_error", "invalid_request", "chunk_size_exceeded", "transcriber_error",
}
_RETRYABLE = {"rate_limited", "queue_overflow", "resource_exhausted", "commit_throttled"}
_FATAL = ("auth_error", "quota_exceeded", "unaccepted_terms")


class ElevenLabsRealtimeModel(BaseASRModel):
    """
    ElevenLabs realtime STT (scribe_v2_realtime) via WebSocket.

    Requires:
        pip install websockets
        ELEVENLABS_API_KEY env variable

    Each clip is resampled to 16 kHz PCM and streamed in 100 ms chunks, paced
    like a live microphone (ELEVENLABS_RT_SPEED=1.0; higher = faster than
    real time). The server segments with VAD; the final chunk forces a commit
    and the committed segments are joined into the prediction. All clips in a
    batch stream concurrently, so use --batch-size to control parallelism.

    Per-row metadata:
        finalization_latency_s  last audio chunk sent -> final committed text
        first_partial_latency_s stream start -> first partial transcript
    """

    def _load_model(self) -> None:
        try:
            import websockets  # noqa: F401
        except ImportError as e:
            raise ImportError("websockets required — run: pip install websockets") from e

        api_key = self.kwargs.get("api_key") or os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError(
                "ElevenLabs API key required. Set ELEVENLABS_API_KEY in .env or pass api_key=..."
            )
        self._api_key = api_key
        self._lang = os.environ.get("ELEVENLABS_RT_LANG", self.language or "ms")
        self._speed = float(os.environ.get("ELEVENLABS_RT_SPEED", "1.0"))
        self._commit_strategy = os.environ.get("ELEVENLABS_RT_COMMIT", "vad")

    def transcribe_batch(
        self,
        audio_arrays: list[np.ndarray],
        sample_rates: list[int],
        audio_lengths_s: list[float],
    ) -> list[TranscriptionResult]:
        async def run_all():
            return await asyncio.gather(*[
                self._transcribe_with_retry(a, sr, d)
                for a, sr, d in zip(audio_arrays, sample_rates, audio_lengths_s)
            ])
        return list(asyncio.run(run_all()))

    async def _transcribe_with_retry(self, audio, sr, dur, max_attempts: int = 6):
        pcm = self._to_pcm16(audio, sr)
        for attempt in range(max_attempts):
            t0 = time.perf_counter()
            try:
                text, meta = await self._stream(pcm)
            except Exception as e:
                # Rate limits, handshake timeouts and dropped connections are transient;
                # account-level errors (bad key, no quota) are not
                fatal = str(e).startswith(_FATAL)
                if not fatal and attempt < max_attempts - 1:
                    wait = min(2 ** (attempt + 1), 30)
                    logger.info("ElevenLabs realtime %s, retrying in %ds", e, wait)
                    await asyncio.sleep(wait)
                    continue
                logger.warning("ElevenLabs realtime failed (%.1fs), skipping row: %s", dur, e)
            else:
                return TranscriptionResult(
                    prediction=text,
                    audio_length_s=dur,
                    transcription_time_s=time.perf_counter() - t0,
                    metadata=meta,
                )
            break
        return TranscriptionResult(prediction="", audio_length_s=dur,
                                   transcription_time_s=0.0, skipped=True)

    async def _stream(self, pcm: bytes) -> tuple[str, dict]:
        import websockets

        params = (f"model_id={self.model_id}&audio_format=pcm_{_TARGET_SR}"
                  f"&commit_strategy={self._commit_strategy}")
        if self._lang:
            params += f"&language_code={self._lang}"
        chunk_bytes = int(_TARGET_SR * _CHUNK_S) * 2
        chunks = [pcm[i:i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)] or [b""]

        committed: list[str] = []
        meta: dict = {}
        final_sent_at: float | None = None
        start = time.perf_counter()

        async with websockets.connect(f"{_WS_URL}?{params}",
                                      additional_headers={"xi-api-key": self._api_key},
                                      max_size=None) as ws:
            got_final = asyncio.Event()

            async def receiver():
                async for raw in ws:
                    msg = json.loads(raw)
                    mtype = msg.get("message_type")
                    if mtype == "partial_transcript" and msg.get("text"):
                        meta.setdefault("first_partial_latency_s",
                                        round(time.perf_counter() - start, 3))
                    elif mtype == "committed_transcript":
                        if msg.get("text", "").strip():
                            committed.append(msg["text"].strip())
                        if final_sent_at is not None:
                            meta["finalization_latency_s"] = round(
                                time.perf_counter() - final_sent_at, 3)
                            got_final.set()
                    elif mtype == "insufficient_audio_activity" and final_sent_at is not None:
                        got_final.set()  # nothing left to commit
                    elif mtype in _RETRYABLE:
                        raise _Retryable(mtype)
                    elif mtype in _ERROR_TYPES:
                        raise RuntimeError(f"{mtype}: {msg}")

            recv_task = asyncio.create_task(receiver())
            try:
                for i, chunk in enumerate(chunks):
                    is_last = i == len(chunks) - 1
                    await ws.send(json.dumps({
                        "message_type": "input_audio_chunk",
                        "audio_base_64": base64.b64encode(chunk).decode(),
                        "commit": is_last,
                        "sample_rate": _TARGET_SR,
                    }))
                    if is_last:
                        final_sent_at = time.perf_counter()
                    else:
                        # Pace like a live stream: chunk i+1 is due at (i+1)*chunk / speed
                        due = start + (i + 1) * _CHUNK_S / self._speed
                        await asyncio.sleep(max(0.0, due - time.perf_counter()))
                    if recv_task.done():
                        recv_task.result()  # surface server errors early
                done, _ = await asyncio.wait(
                    {recv_task, asyncio.create_task(got_final.wait())},
                    timeout=20, return_when=asyncio.FIRST_COMPLETED)
                if recv_task in done:
                    recv_task.result()
                if not got_final.is_set():
                    meta["final_commit_timeout"] = True
            finally:
                recv_task.cancel()

        text = _EVENT_RE.sub("", " ".join(committed))
        return re.sub(r"\s+", " ", text).strip(), meta

    @staticmethod
    def _to_pcm16(audio: np.ndarray, sr: int) -> bytes:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != _TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)
        return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class _Retryable(Exception):
    pass
