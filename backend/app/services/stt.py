"""Streaming speech-to-text. Deepgram realtime WebSocket, OpenAI Whisper fallback."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import wave
from typing import Any, Awaitable, Callable

import httpx
import websockets

from ..config import get_settings

log = logging.getLogger(__name__)

TranscriptCallback = Callable[[str, bool], Awaitable[None]]
SpeechStartedCallback = Callable[[], Awaitable[None]]


class DeepgramStream:
    """Full-duplex Deepgram connection fed raw µ-law 8 kHz frames from Twilio."""

    def __init__(
        self,
        on_transcript: TranscriptCallback,
        on_speech_started: SpeechStartedCallback | None = None,
    ) -> None:
        self._settings = get_settings()
        self._on_transcript = on_transcript
        self._on_speech_started = on_speech_started
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._reader: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._closed

    async def connect(self) -> None:
        s = self._settings
        if not s.deepgram_api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not configured")
        params = {
            "model": s.deepgram_model,
            "language": s.deepgram_language,
            "encoding": "mulaw",
            "sample_rate": "8000",
            "channels": "1",
            "punctuate": "true",
            "smart_format": "true",
            "interim_results": "true",
            "endpointing": "300",
            "vad_events": "true",
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"wss://api.deepgram.com/v1/listen?{query}"
        self._ws = await websockets.connect(
            url,
            additional_headers={"Authorization": f"Token {s.deepgram_api_key}"},
            ping_interval=5,
            ping_timeout=20,
            max_size=None,
        )
        self._reader = asyncio.create_task(self._read_loop())
        log.info("deepgram stream connected model=%s", s.deepgram_model)

    async def send_audio(self, mulaw: bytes) -> None:
        if self._ws and not self._closed:
            try:
                await self._ws.send(mulaw)
            except Exception:  # noqa: BLE001
                log.warning("deepgram send failed", exc_info=True)

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                msg: dict[str, Any] = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "SpeechStarted" and self._on_speech_started:
                    await self._on_speech_started()
                    continue
                if mtype != "Results":
                    continue
                alts = msg.get("channel", {}).get("alternatives", [])
                if not alts:
                    continue
                text = (alts[0].get("transcript") or "").strip()
                if not text:
                    continue
                await self._on_transcript(text, bool(msg.get("is_final")))
        except websockets.ConnectionClosed:
            log.info("deepgram connection closed")
        except Exception:  # noqa: BLE001
            log.exception("deepgram read loop error")

    async def finish(self) -> None:
        if self._ws and not self._closed:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:  # noqa: BLE001
                pass

    async def close(self) -> None:
        self._closed = True
        if self._reader:
            self._reader.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
        self._ws = None


async def transcribe_pcm_openai(pcm16_8k: bytes) -> str:
    """Whisper fallback for buffered utterances when Deepgram is unavailable."""
    s = get_settings()
    if not s.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured for STT fallback")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(pcm16_8k)
    buf.seek(0)
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {s.openai_api_key}"},
            files={"file": ("audio.wav", buf.getvalue(), "audio/wav")},
            data={"model": "whisper-1", "response_format": "text"},
        )
    resp.raise_for_status()
    return resp.text.strip()


def decode_twilio_payload(payload: str) -> bytes:
    return base64.b64decode(payload)
