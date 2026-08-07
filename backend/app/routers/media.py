"""Twilio Media Streams WebSocket endpoint (bidirectional audio)."""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import get_settings
from ..security import verify_media_stream_token
from ..services import calls
from ..services.session import CallSession, registry

log = logging.getLogger(__name__)
router = APIRouter(tags=["media"])


@router.websocket("/ws/media-stream")
async def media_stream(ws: WebSocket) -> None:
    requested_call_id = ws.query_params.get("call_id") or ""
    if not requested_call_id or not verify_media_stream_token(
        requested_call_id, ws.query_params.get("token")
    ):
        await ws.close(code=1008, reason="Invalid media stream token")
        return
    await ws.accept()
    session: CallSession | None = None
    watchdog: asyncio.Task[None] | None = None
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                continue

            if event == "start":
                start = msg.get("start", {})
                params = start.get("customParameters", {}) or {}
                call_id = params.get("callId") or requested_call_id
                if not call_id:
                    log.error("media stream start without call_id")
                    await ws.close(code=1008)
                    return
                if call_id != requested_call_id:
                    log.error("media stream call_id mismatch")
                    await ws.close(code=1008)
                    return
                call = await calls.get_call(call_id)
                if not call:
                    log.error("media stream for unknown call %s", call_id)
                    await ws.close(code=1008)
                    return
                agent = await calls.get_agent(str(call["agent_id"])) if call["agent_id"] else None
                if not agent:
                    await calls.log_event(
                        call_id=call_id,
                        org_id=str(call["org_id"]),
                        kind="error",
                        payload={"stage": "session", "message": "Agent not found"},
                    )
                    await ws.close(code=1011)
                    return
                session = CallSession(
                    ws=ws,
                    call_id=call_id,
                    org_id=str(call["org_id"]),
                    agent=agent,
                    stream_sid=start.get("streamSid") or msg.get("streamSid"),
                    external_call_id=start.get("callSid"),
                    resume_escalation_id=params.get("resumeEscalationId")
                    or ws.query_params.get("resume_escalation"),
                )
                registry.add(session)
                await session.start()
                watchdog = asyncio.create_task(_watchdog(session))
                continue

            if event == "media" and session:
                await session.on_media(msg["media"]["payload"])
                continue

            if event == "mark":
                continue

            if event == "stop":
                break
    except WebSocketDisconnect:
        log.info("media stream disconnected")
    except Exception:  # noqa: BLE001
        log.exception("media stream error")
        if session:
            await session.event("error", {"stage": "media_stream"})
    finally:
        if watchdog:
            watchdog.cancel()
        if session:
            registry.remove(session.call_id)
            await session.close("completed", "stream_stopped")
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


async def _watchdog(session: CallSession) -> None:
    """Enforces max call length and silence hangup."""
    s = get_settings()
    while not session.closed:
        await asyncio.sleep(2)
        now = time.monotonic()
        if now - session.started_at > s.max_call_seconds:
            await session.event("status", {"status": "max_duration_reached"})
            await session.close("completed", "max_duration")
            try:
                await session.ws.close()
            except Exception:  # noqa: BLE001
                pass
            return
        if (
            not session.speaking
            and now - session.last_voice_at > s.silence_hangup_seconds
        ):
            await session.event("silence", {"seconds": s.silence_hangup_seconds})
            await session.speak(
                "I haven't heard anything, so I'll end the call here. Thanks for calling.",
                persist=True,
            )
            await session.close("completed", "silence_timeout")
            try:
                await session.ws.close()
            except Exception:  # noqa: BLE001
                pass
            return
