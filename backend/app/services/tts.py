"""Streaming text-to-speech returning PCM16 chunks ready for µ-law conversion."""

from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx

from ..config import get_settings

log = logging.getLogger(__name__)

async def stream_tts(text: str, voice: str | None = None) -> AsyncIterator[bytes]:
    s = get_settings()
    if s.tts_provider == "elevenlabs" and s.elevenlabs_api_key:
        async for chunk in _elevenlabs(text, voice):
            yield chunk
        return
    async for chunk in _openai(text, voice):
        yield chunk


async def _elevenlabs(text: str, voice: str | None) -> AsyncIterator[bytes]:
    s = get_settings()
    voice_id = voice or s.elevenlabs_voice_id
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        f"?output_format=pcm_16000&optimize_streaming_latency=3"
    )
    payload = {
        "text": text,
        "model_id": s.elevenlabs_model,
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "speed": 1.0},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            url,
            headers={
                "xi-api-key": s.elevenlabs_api_key or "",
                "accept": "audio/pcm",
                "content-type": "application/json",
            },
            json=payload,
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "ignore")
                raise RuntimeError(f"ElevenLabs error {resp.status_code}: {body[:300]}")
            async for chunk in resp.aiter_bytes(chunk_size=3200):
                if chunk:
                    yield chunk


async def _openai(text: str, voice: str | None) -> AsyncIterator[bytes]:
    s = get_settings()
    if not s.openai_api_key:
        raise RuntimeError("No TTS provider configured (set ELEVENLABS or OPENAI key)")
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {s.openai_api_key}"},
            json={
                "model": s.openai_tts_model,
                "voice": voice or s.openai_tts_voice,
                "input": text,
                "response_format": "pcm",  # 24 kHz PCM16
            },
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "ignore")
                raise RuntimeError(f"OpenAI TTS error {resp.status_code}: {body[:300]}")
            async for chunk in resp.aiter_bytes(chunk_size=9600):
                if chunk:
                    yield chunk


def source_sample_rate() -> int:
    s = get_settings()
    if s.tts_provider == "elevenlabs" and s.elevenlabs_api_key:
        return 16000
    return 24000
