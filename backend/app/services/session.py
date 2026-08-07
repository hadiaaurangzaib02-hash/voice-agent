"""Live call session orchestration: STT -> LLM(+RAG) -> TTS with barge-in."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from ..config import get_settings
from . import audio, calls, escalation, llm, memory, stt, tts

log = logging.getLogger(__name__)


@dataclass
class CallSession:
    ws: WebSocket
    call_id: str
    org_id: str
    agent: dict[str, Any]
    stream_sid: str | None = None
    external_call_id: str | None = None
    resume_escalation_id: str | None = None

    speaking: bool = False
    generation: int = 0
    closed: bool = False
    started_at: float = field(default_factory=time.monotonic)
    last_voice_at: float = field(default_factory=time.monotonic)
    _pending_final: str = ""
    _turn_task: asyncio.Task[None] | None = None
    _stt: stt.DeepgramStream | None = None
    _muted: bool = False
    _memory: str = ""
    _profile: dict[str, Any] | None = None
    _opening_task: asyncio.Task[None] | None = None
    _finalize_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ setup
    async def start(self) -> None:
        await self.event("session_started", {"agent": self.agent.get("name")})
        await self._load_memory()
        await calls.update_status(self.call_id, "ai_answering")
        self._stt = stt.DeepgramStream(
            on_transcript=self._on_transcript, on_speech_started=self._on_speech_started
        )
        try:
            await self._stt.connect()
        except Exception as exc:  # noqa: BLE001
            await self.event("error", {"stage": "stt_connect", "message": str(exc)})
            log.exception("stt connect failed")
            raise
        opening = await self._opening_line()
        # Keep receiving caller audio while the greeting is synthesized so
        # Deepgram VAD can interrupt the agent immediately.
        self._opening_task = asyncio.create_task(self._speak_opening(opening))

    async def _speak_opening(self, opening: str) -> None:
        await self.speak(opening, persist=True)
        if not self.closed:
            await calls.update_status(self.call_id, "listening")

    async def _load_memory(self) -> None:
        call = await calls.get_call(self.call_id)
        if not call:
            return
        customer = (
            call.get("caller_e164")
            if call.get("direction") == "inbound"
            else call.get("callee_e164")
        )
        try:
            self._memory = await memory.caller_history(
                org_id=self.org_id, e164=customer, exclude_call_id=self.call_id
            )
            self._profile = await memory.caller_profile(org_id=self.org_id, e164=customer)
        except Exception:  # noqa: BLE001
            log.warning("memory load failed", exc_info=True)

    async def _opening_line(self) -> str:
        """Resume an escalated conversation, or greet normally."""
        if self.resume_escalation_id:
            row = await escalation.get_escalation(self.resume_escalation_id)
            if row and (row.get("operator_answer") or "").strip():
                await escalation.mark_resolved(self.resume_escalation_id, "resolved")
                await self.event(
                    "escalation",
                    {"escalation_id": self.resume_escalation_id, "status": "resumed"},
                )
                return await escalation.resume_text(row)
        return self.agent.get("greeting") or "Hello, how can I help you today?"

    # ------------------------------------------------------------- media in
    async def on_media(self, payload: str) -> None:
        if self._muted:
            return
        raw = base64.b64decode(payload)
        pcm = audio.mulaw_to_pcm16(raw)
        if audio.rms_level(pcm) > 350:
            self.last_voice_at = time.monotonic()
        if self._stt:
            await self._stt.send_audio(raw)

    async def _on_speech_started(self) -> None:
        """Barge-in: caller started talking while the agent was speaking."""
        if self.speaking and get_settings().barge_in_enabled:
            await self.interrupt("barge_in")

    async def _on_transcript(self, text: str, is_final: bool) -> None:
        if not is_final:
            if self.speaking and get_settings().barge_in_enabled and len(text) > 2:
                await self.interrupt("barge_in")
            return
        self._pending_final = f"{self._pending_final} {text}".strip()
        if self._finalize_task and not self._finalize_task.done():
            self._finalize_task.cancel()
        self._finalize_task = asyncio.create_task(self._finalize_utterance())

    async def _finalize_utterance(self) -> None:
        # Debounce without blocking Deepgram's receive loop.
        await asyncio.sleep(0.3)
        utterance, self._pending_final = self._pending_final, ""
        if not utterance:
            return
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        if self._opening_task and not self._opening_task.done():
            self._opening_task.cancel()
        self._turn_task = asyncio.create_task(self._handle_turn(utterance))

    # --------------------------------------------------------------- turn
    async def _handle_turn(self, utterance: str) -> None:
        t0 = time.monotonic()
        try:
            await calls.add_message(
                call_id=self.call_id, org_id=self.org_id, role="user", content=utterance
            )
            await self.event("transcript", {"role": "user", "text": utterance})
            await calls.update_status(self.call_id, "thinking")

            if escalation.detects_do_not_call(utterance):
                await self._handle_do_not_call()
                return

            context = await llm.retrieve_context(
                self.org_id, self.agent.get("id"), utterance
            )
            history = await calls.transcript(self.call_id)
            messages = llm.build_messages(
                agent=self.agent,
                history=history,
                context=context,
                memory=self._memory,
                profile=self._profile,
            )

            self.generation += 1
            gen = self.generation
            spoken: list[str] = []
            async for sentence in llm.stream_completion(messages):
                if gen != self.generation or self.closed:
                    return
                if llm.ESCALATE_TOKEN in sentence:
                    await self._escalate(utterance)
                    return
                spoken.append(sentence)
                await self.speak(sentence, persist=False, generation=gen)

            reply = " ".join(spoken).strip()
            if reply:
                latency = int((time.monotonic() - t0) * 1000)
                await calls.add_message(
                    call_id=self.call_id,
                    org_id=self.org_id,
                    role="assistant",
                    content=reply,
                    latency_ms=latency,
                )
                await self.event(
                    "transcript",
                    {"role": "assistant", "text": reply, "latency_ms": latency},
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("turn failed")
            await self.event("error", {"stage": "turn", "message": str(exc)})
            await self.speak(
                "Sorry, I ran into a problem on my side. Could you repeat that?",
                persist=True,
            )
        finally:
            if not self.closed:
                await calls.update_status(self.call_id, "listening")

    # ----------------------------------------------------------- escalation
    async def _escalate(self, question: str) -> None:
        """AI does not know: tell the caller, notify a supervisor, hold the line."""
        await self.speak(escalation.HOLD_MESSAGE, persist=True)
        call = await calls.get_call(self.call_id)
        customer = None
        if call:
            customer = (
                call.get("caller_e164")
                if call.get("direction") == "inbound"
                else call.get("callee_e164")
            )
        row = await escalation.create_escalation(
            call_id=self.call_id,
            org_id=self.org_id,
            agent_id=self.agent.get("id"),
            question=question,
            customer_e164=customer,
        )
        await calls.update_status(self.call_id, "on_hold")
        await self._redirect_to_hold(str(row["id"]))

    async def _redirect_to_hold(self, escalation_id: str) -> None:
        from ..config import get_settings as _s
        from . import twilio_client

        call = await calls.get_call(self.call_id)
        if not call or not call.get("external_call_id"):
            return
        number = (
            await calls.get_number_by_id(str(call["phone_number_id"]))
            if call.get("phone_number_id")
            else None
        )
        base = _s().public_base_url.rstrip("/")
        url = f"{base}/webhooks/twilio/hold?call_id={self.call_id}&escalation_id={escalation_id}"
        try:
            creds = twilio_client.credentials_for(number)
            await twilio_client.redirect_call(creds, str(call["external_call_id"]), url)
        except Exception as exc:  # noqa: BLE001
            log.exception("hold redirect failed")
            await self.event("error", {"stage": "hold", "message": str(exc)})

    async def _handle_do_not_call(self) -> None:
        call = await calls.get_call(self.call_id)
        customer = None
        if call:
            customer = call.get("callee_e164") or call.get("caller_e164")
        await escalation.add_do_not_call(
            org_id=self.org_id,
            e164=customer,
            reason="Customer asked not to be called again",
            call_id=self.call_id,
        )
        await self.event("crm_action", {"action": "do_not_call", "e164": customer})
        await self.speak(
            "Understood. I've added your number to our do-not-call list and you won't "
            "hear from us again. Thank you for your time.",
            persist=True,
        )
        await self.close("completed", "do_not_call")

    # -------------------------------------------------------------- speaking
    async def speak(
        self, text: str, *, persist: bool, generation: int | None = None
    ) -> None:
        if self.closed or not self.stream_sid:
            return
        gen = generation if generation is not None else self.generation
        self.speaking = True
        await calls.update_status(self.call_id, "speaking")
        rate = tts.source_sample_rate()
        encoder = audio.TwilioAudioEncoder(rate)
        try:
            async for pcm in tts.stream_tts(text, self.agent.get("voice")):
                if self.closed or gen != self.generation:
                    return
                for frame in encoder.encode(pcm):
                    if self.closed or gen != self.generation:
                        return
                    await self.ws.send_text(
                        json.dumps(
                            {
                                "event": "media",
                                "streamSid": self.stream_sid,
                                "media": {"payload": frame},
                            }
                        )
                    )
                    await asyncio.sleep(0.018)  # pace ~20 ms frames
        except Exception as exc:  # noqa: BLE001
            log.exception("tts failed")
            await self.event("error", {"stage": "tts", "message": str(exc)})
        finally:
            self.speaking = False
        if persist and text.strip():
            await calls.add_message(
                call_id=self.call_id,
                org_id=self.org_id,
                role="assistant",
                content=text.strip(),
            )
            await self.event("transcript", {"role": "assistant", "text": text.strip()})

    async def interrupt(self, reason: str) -> None:
        self.generation += 1
        self.speaking = False
        if self.stream_sid:
            try:
                await self.ws.send_text(
                    json.dumps({"event": "clear", "streamSid": self.stream_sid})
                )
            except Exception:  # noqa: BLE001
                pass
        await self.event("interruption", {"reason": reason})
        await calls.update_status(self.call_id, "listening")

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    # ---------------------------------------------------------------- events
    async def event(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        await calls.log_event(
            call_id=self.call_id,
            org_id=self.org_id,
            agent_id=self.agent.get("id"),
            kind=kind,
            payload=payload or {},
        )

    async def close(self, status: str = "completed", reason: str = "stop") -> None:
        if self.closed:
            return
        self.closed = True
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        if self._opening_task and not self._opening_task.done():
            self._opening_task.cancel()
        if self._finalize_task and not self._finalize_task.done():
            self._finalize_task.cancel()
        if self._stt:
            await self._stt.close()
        await self.event("ended", {"reason": reason})
        await calls.end_call(self.call_id, status)
        if reason != "supervisor_takeover":
            from . import postcall

            asyncio.create_task(postcall.finalize(self.call_id))


class SessionRegistry:
    """Tracks every live media session so control APIs can act on them."""

    def __init__(self) -> None:
        self._by_call: dict[str, CallSession] = {}

    def add(self, session: CallSession) -> None:
        self._by_call[session.call_id] = session

    def remove(self, call_id: str) -> None:
        self._by_call.pop(call_id, None)

    def get(self, call_id: str) -> CallSession | None:
        return self._by_call.get(call_id)

    def all(self) -> list[CallSession]:
        return list(self._by_call.values())


registry = SessionRegistry()
