"""G.711 µ-law <-> PCM helpers for Twilio Media Streams (8 kHz mono µ-law)."""

from __future__ import annotations

import audioop
import base64

TWILIO_SAMPLE_RATE = 8000
FRAME_BYTES = 160  # 20 ms of 8 kHz µ-law


class TwilioAudioEncoder:
    """Stateful PCM16 resampler/framer for uninterrupted Media Stream audio."""

    def __init__(self, src_rate: int) -> None:
        self.src_rate = src_rate
        self._rate_state = None
        self._source_pending = b""
        self._pending = b""

    def encode(self, pcm: bytes) -> list[str]:
        pcm = self._source_pending + pcm
        aligned = len(pcm) - (len(pcm) % 2)
        self._source_pending = pcm[aligned:]
        if aligned == 0:
            return []
        converted, self._rate_state = resample_pcm16(
            pcm[:aligned], self.src_rate, TWILIO_SAMPLE_RATE, self._rate_state
        )
        self._pending += pcm16_to_mulaw(converted)
        frames: list[str] = []
        while len(self._pending) >= FRAME_BYTES:
            frame, self._pending = self._pending[:FRAME_BYTES], self._pending[FRAME_BYTES:]
            frames.append(base64.b64encode(frame).decode("ascii"))
        return frames


def mulaw_to_pcm16(mulaw: bytes) -> bytes:
    return audioop.ulaw2lin(mulaw, 2)


def pcm16_to_mulaw(pcm: bytes) -> bytes:
    return audioop.lin2ulaw(pcm, 2)


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int, state=None):
    if src_rate == dst_rate:
        return pcm, state
    return audioop.ratecv(pcm, 2, 1, src_rate, dst_rate, state)


def pcm16_to_twilio_payloads(pcm: bytes, src_rate: int) -> list[str]:
    """Convert arbitrary-rate PCM16 mono into base64 20 ms µ-law frames."""
    converted, _ = resample_pcm16(pcm, src_rate, TWILIO_SAMPLE_RATE)
    mulaw = pcm16_to_mulaw(converted)
    return [
        base64.b64encode(mulaw[i : i + FRAME_BYTES]).decode("ascii")
        for i in range(0, len(mulaw), FRAME_BYTES)
        if mulaw[i : i + FRAME_BYTES]
    ]


def rms_level(pcm: bytes) -> int:
    if not pcm:
        return 0
    return audioop.rms(pcm, 2)
