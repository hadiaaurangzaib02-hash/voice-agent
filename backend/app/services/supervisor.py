"""Supervisor live-call control: listen, whisper, join, take over, hold, resume.

Implemented with real Twilio Conferences: the customer leg is redirected into a
per-call conference, then the supervisor is dialled into the same conference either
muted (listen), as a coach (whisper) or as a full participant (join / take over).
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import get_settings
from ..security import media_stream_token, ws_url_for
from . import calls as call_svc
from . import twilio_client
from .session import registry

log = logging.getLogger(__name__)


def conference_name(call_id: str) -> str:
    return f"call-{call_id}"


async def _creds_for_call(call: dict[str, Any]) -> twilio_client.TwilioCredentials:
    number = (
        await call_svc.get_number_by_id(str(call["phone_number_id"]))
        if call.get("phone_number_id")
        else None
    )
    return twilio_client.credentials_for(number)


async def ensure_conference(call: dict[str, Any], *, end_ai: bool) -> str:
    """Move the customer leg into a conference so supervisors can be bridged in."""
    call_id = str(call["id"])
    name = conference_name(call_id)
    base = get_settings().public_base_url.rstrip("/")
    creds = await _creds_for_call(call)

    if end_ai:
        session = registry.get(call_id)
        if session:
            await session.close("active", "supervisor_takeover")

    await twilio_client.redirect_call(
        creds,
        str(call["external_call_id"]),
        f"{base}/webhooks/twilio/conference?name={name}&call_id={call_id}",
    )
    return name


async def bridge_supervisor(
    *, call: dict[str, Any], supervisor_e164: str, mode: str
) -> dict[str, Any]:
    """mode: listen | whisper | join | takeover."""
    call_id = str(call["id"])
    if not call.get("external_call_id"):
        raise ValueError("Call has no active provider leg")

    name = await ensure_conference(call, end_ai=mode == "takeover")
    base = get_settings().public_base_url.rstrip("/")
    creds = await _creds_for_call(call)

    answer_url = (
        f"{base}/webhooks/twilio/supervisor?name={name}&mode={mode}"
        f"&coach={call['external_call_id']}"
    )
    from_number = call.get("caller_e164") if call.get("direction") == "outbound" else call.get("callee_e164")
    result = await twilio_client.create_call(
        creds,
        to=supervisor_e164,
        from_=str(from_number or ""),
        answer_url=answer_url,
        status_callback_url=f"{base}/webhooks/twilio/status?call_id={call_id}",
        machine_detection=False,
    )

    await call_svc.update_status(
        call_id, "human_transfer" if mode == "takeover" else "active"
    )
    await call_svc.log_event(
        call_id=call_id,
        org_id=str(call["org_id"]),
        kind="escalation",
        payload={"supervisor": supervisor_e164, "mode": mode, "conference": name},
    )
    return {"conference": name, "supervisor_call_sid": result.get("sid"), "mode": mode}


async def hold(call: dict[str, Any]) -> None:
    call_id = str(call["id"])
    base = get_settings().public_base_url.rstrip("/")
    creds = await _creds_for_call(call)
    await twilio_client.redirect_call(
        creds,
        str(call["external_call_id"]),
        f"{base}/webhooks/twilio/hold?call_id={call_id}&escalation_id=",
    )
    await call_svc.update_status(call_id, "on_hold")
    await call_svc.log_event(
        call_id=call_id, org_id=str(call["org_id"]), kind="status",
        payload={"status": "on_hold", "by": "supervisor"},
    )


async def resume(call: dict[str, Any]) -> None:
    """Take the caller off hold and hand the conversation back to the AI agent."""
    call_id = str(call["id"])
    base = get_settings().public_base_url.rstrip("/")
    creds = await _creds_for_call(call)
    stream_url = ws_url_for(
        f"/ws/media-stream?call_id={call_id}&token={media_stream_token(call_id)}"
    )
    status_url = f"{base}/webhooks/twilio/status?call_id={call_id}"
    xml_url = f"{base}/webhooks/twilio/answer?call_id={call_id}&number_id={call.get('phone_number_id')}"
    log.info("resuming %s to %s (stream %s)", call_id, xml_url, stream_url)
    await twilio_client.redirect_call(creds, str(call["external_call_id"]), xml_url)
    await call_svc.update_status(call_id, "connecting")
    await call_svc.log_event(
        call_id=call_id, org_id=str(call["org_id"]), kind="status",
        payload={"status": "resumed", "by": "supervisor", "status_url": status_url},
    )
