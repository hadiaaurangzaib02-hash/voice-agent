"""Control APIs consumed by the dashboard: outbound, hangup, transfer, live, history."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ..config import get_settings
from ..models import (
    CallActionRequest,
    CampaignRequest,
    CampaignResponse,
    MuteRequest,
    OutboundCallRequest,
    OutboundCallResponse,
    RecordRequest,
    SupervisorRequest,
    TransferRequest,
)
from ..security import require_api_key
from ..services import calls as call_svc
from ..services import dialer, supervisor, twilio_client
from ..services.session import registry

log = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/calls", tags=["calls"], dependencies=[Depends(require_api_key)]
)


async def _call_and_creds(call_id: str) -> tuple[dict[str, Any], Any]:
    call = await call_svc.get_call(call_id)
    if not call:
        raise HTTPException(404, "Call not found")
    number = (
        await call_svc.get_number_by_id(str(call["phone_number_id"]))
        if call.get("phone_number_id")
        else None
    )
    return call, twilio_client.credentials_for(number)


@router.post("/outbound", response_model=OutboundCallResponse)
async def outbound(body: OutboundCallRequest) -> OutboundCallResponse:
    if body.scheduled_at:
        asyncio.create_task(
            dialer.schedule_call(
                run_at=body.scheduled_at,
                number_id=body.number_id,
                to=body.to,
                agent_id=body.agent_id,
                max_retries=body.max_retries,
            )
        )
        return OutboundCallResponse(call_id="", external_call_id=None, status="scheduled")
    try:
        result = await dialer.place_call(
            number_id=body.number_id,
            to=body.to,
            agent_id=body.agent_id,
            max_retries=body.max_retries,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Provider rejected the call: {exc}") from exc
    return OutboundCallResponse(**result)


@router.post("/hangup")
async def hangup(body: CallActionRequest) -> dict[str, str]:
    call, creds = await _call_and_creds(body.call_id)
    if call.get("external_call_id"):
        await twilio_client.hangup(creds, str(call["external_call_id"]))
    session = registry.get(body.call_id)
    if session:
        await session.close("completed", "manual_hangup")
    else:
        await call_svc.end_call(body.call_id, "completed")
    await call_svc.log_event(
        call_id=body.call_id, org_id=str(call["org_id"]), kind="status",
        payload={"status": "hangup", "by": "dashboard"},
    )
    return {"status": "ok"}


@router.post("/transfer")
async def transfer(body: TransferRequest) -> dict[str, str]:
    call, creds = await _call_and_creds(body.call_id)
    if not call.get("external_call_id"):
        raise HTTPException(409, "Call has no active provider leg")
    base = get_settings().public_base_url.rstrip("/")
    params = f"to={body.to}&caller_id={call.get('caller_e164') or ''}"
    if body.announce:
        params += f"&announce={body.announce}"
    await twilio_client.redirect_call(
        creds, str(call["external_call_id"]), f"{base}/webhooks/twilio/transfer?{params}"
    )
    await call_svc.update_status(body.call_id, "human_transfer")
    await call_svc.log_event(
        call_id=body.call_id, org_id=str(call["org_id"]), kind="transfer",
        payload={"to": body.to},
    )
    return {"status": "transferring"}


@router.post("/mute")
async def mute(body: MuteRequest) -> dict[str, str]:
    session = registry.get(body.call_id)
    if not session:
        raise HTTPException(404, "No live media session for this call")
    session.set_muted(True)
    await session.event("status", {"status": "muted"})
    return {"status": "muted"}


@router.post("/unmute")
async def unmute(body: MuteRequest) -> dict[str, str]:
    session = registry.get(body.call_id)
    if not session:
        raise HTTPException(404, "No live media session for this call")
    session.set_muted(False)
    await session.event("status", {"status": "unmuted"})
    return {"status": "unmuted"}


@router.post("/record")
async def record(body: RecordRequest) -> dict[str, Any]:
    call, creds = await _call_and_creds(body.call_id)
    sid = str(call.get("external_call_id") or "")
    if not sid:
        raise HTTPException(409, "Call has no active provider leg")
    if body.action == "start":
        base = get_settings().public_base_url.rstrip("/")
        result = await twilio_client.start_recording(
            creds,
            sid,
            callback_url=f"{base}/webhooks/twilio/recording?call_id={body.call_id}",
        )
    else:
        if not body.recording_sid:
            raise HTTPException(400, "recording_sid is required to stop a recording")
        result = await twilio_client.stop_recording(creds, sid, body.recording_sid)
    await call_svc.log_event(
        call_id=body.call_id, org_id=str(call["org_id"]), kind="recording",
        payload={"action": body.action, "recording_sid": result.get("sid")},
    )
    return {"status": body.action, "recording_sid": result.get("sid")}


@router.post("/supervisor")
async def supervisor_bridge(body: SupervisorRequest) -> dict[str, Any]:
    """Listen, whisper, join or take over a live call (real Twilio conference)."""
    call = await call_svc.get_call(body.call_id)
    if not call:
        raise HTTPException(404, "Call not found")
    try:
        return await supervisor.bridge_supervisor(
            call=call, supervisor_e164=body.supervisor_e164, mode=body.mode
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Provider rejected the bridge: {exc}") from exc


@router.post("/hold")
async def hold_call(body: CallActionRequest) -> dict[str, str]:
    call = await call_svc.get_call(body.call_id)
    if not call or not call.get("external_call_id"):
        raise HTTPException(409, "Call has no active provider leg")
    await supervisor.hold(call)
    return {"status": "on_hold"}


@router.post("/resume")
async def resume_call(body: CallActionRequest) -> dict[str, str]:
    call = await call_svc.get_call(body.call_id)
    if not call or not call.get("external_call_id"):
        raise HTTPException(409, "Call has no active provider leg")
    await supervisor.resume(call)
    return {"status": "resuming"}


@router.get("/live")
async def live(org_id: str | None = Query(default=None)) -> dict[str, Any]:
    rows = await call_svc.live_calls(org_id)
    sessions = {s.call_id: s for s in registry.all()}
    for row in rows:
        s = sessions.get(str(row["id"]))
        row["media_session"] = bool(s)
        row["agent_speaking"] = bool(s and s.speaking)
    return {
        "calls": rows,
        "counts": {
            "total": len(rows),
            "inbound": sum(1 for r in rows if r.get("direction") == "inbound"),
            "outbound": sum(1 for r in rows if r.get("direction") == "outbound"),
            "media_sessions": len(sessions),
        },
    }


@router.get("/history")
async def history(
    org_id: str, limit: int = Query(default=50, le=200), offset: int = 0
) -> dict[str, Any]:
    return {"calls": await call_svc.history(org_id, limit, offset)}


@router.post("/campaign", response_model=CampaignResponse)
async def campaign(body: CampaignRequest) -> CampaignResponse:
    if body.scheduled_at:
        asyncio.create_task(_scheduled_campaign(body))
        return CampaignResponse(
            campaign_id="scheduled",
            accepted=len(body.targets),
            scheduled_at=body.scheduled_at,
        )
    campaign_id = await dialer.run_campaign(
        number_id=body.number_id,
        targets=body.targets,
        agent_id=body.agent_id,
        concurrency=body.concurrency,
        max_retries=body.max_retries,
    )
    return CampaignResponse(campaign_id=campaign_id, accepted=len(body.targets))


async def _scheduled_campaign(body: CampaignRequest) -> None:
    from datetime import datetime, timezone

    assert body.scheduled_at is not None
    delay = max(0.0, (body.scheduled_at - datetime.now(timezone.utc)).total_seconds())
    await asyncio.sleep(delay)
    await dialer.run_campaign(
        number_id=body.number_id,
        targets=body.targets,
        agent_id=body.agent_id,
        concurrency=body.concurrency,
        max_retries=body.max_retries,
    )
