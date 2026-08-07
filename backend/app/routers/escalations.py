"""Supervisor escalation API: inbox, answering, callback control, do-not-call."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ..models import (
    DoNotCallRequest,
    EscalationAnswerRequest,
    EscalationAnswerResponse,
)
from ..security import require_api_key
from ..services import calls as call_svc
from ..services import escalation as esc
from ..services import twilio_client
from ..services.session import registry
from .. import db

log = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/escalations", tags=["escalations"], dependencies=[Depends(require_api_key)]
)


@router.get("")
async def list_escalations(
    org_id: str, status: str | None = Query(default=None), limit: int = 50
) -> dict[str, Any]:
    rows = await db.fetch(
        """
        select * from public.escalations
         where org_id = $1::uuid
           and ($2::text is null or status = $2)
         order by created_at desc
         limit $3
        """,
        org_id,
        status,
        limit,
    )
    return {"escalations": [dict(r) for r in rows]}


@router.post("/answer", response_model=EscalationAnswerResponse)
async def answer(body: EscalationAnswerRequest) -> EscalationAnswerResponse:
    """Operator replies. Resume the live call if still held, else call the customer back."""
    escalation = await esc.get_escalation(body.escalation_id)
    if not escalation:
        raise HTTPException(404, "Escalation not found")
    if escalation["status"] in ("resolved", "do_not_call"):
        raise HTTPException(409, "Escalation is already closed")

    escalation = await esc.record_answer(
        escalation_id=body.escalation_id,
        answer=body.answer,
        answered_by=body.answered_by,
    )

    call = (
        await call_svc.get_call(str(escalation["call_id"]))
        if escalation.get("call_id")
        else None
    )
    live = bool(call and call.get("ended_at") is None and call.get("external_call_id"))

    if live:
        # The hold loop picks the answer up on its next poll (within ~30 seconds).
        await call_svc.log_event(
            call_id=str(call["id"]),
            org_id=str(escalation["org_id"]),
            kind="escalation",
            payload={"escalation_id": body.escalation_id, "status": "resuming_live_call"},
        )
        return EscalationAnswerResponse(
            escalation_id=body.escalation_id, delivery="resume_live_call"
        )

    await esc.schedule_callback(escalation)
    return EscalationAnswerResponse(
        escalation_id=body.escalation_id, delivery="callback_scheduled"
    )


@router.post("/resolve")
async def resolve(escalation_id: str) -> dict[str, str]:
    await esc.mark_resolved(escalation_id, "resolved")
    return {"status": "resolved"}


@router.get("/do-not-call")
async def list_dnc(org_id: str) -> dict[str, Any]:
    rows = await db.fetch(
        "select * from public.do_not_call where org_id = $1::uuid order by created_at desc",
        org_id,
    )
    return {"numbers": [dict(r) for r in rows]}


@router.post("/do-not-call")
async def add_dnc(body: DoNotCallRequest) -> dict[str, str]:
    await esc.add_do_not_call(
        org_id=body.org_id,
        e164=body.e164,
        reason=body.reason or "Added from the dashboard",
        call_id=None,
    )
    return {"status": "added"}


@router.delete("/do-not-call")
async def remove_dnc(org_id: str, e164: str) -> dict[str, str]:
    await db.execute(
        "delete from public.do_not_call where org_id = $1::uuid and e164 = $2",
        org_id,
        e164,
    )
    return {"status": "removed"}


@router.post("/hold")
async def manual_hold(call_id: str) -> dict[str, str]:
    """Put a live caller on professional hold music without an AI escalation."""
    call = await call_svc.get_call(call_id)
    if not call or not call.get("external_call_id"):
        raise HTTPException(409, "Call has no active provider leg")
    session = registry.get(call_id)
    if session:
        await session._redirect_to_hold("")  # noqa: SLF001 - same module family
    else:
        from ..config import get_settings

        number = (
            await call_svc.get_number_by_id(str(call["phone_number_id"]))
            if call.get("phone_number_id")
            else None
        )
        base = get_settings().public_base_url.rstrip("/")
        await twilio_client.redirect_call(
            twilio_client.credentials_for(number),
            str(call["external_call_id"]),
            f"{base}/webhooks/twilio/hold?call_id={call_id}&escalation_id=",
        )
    await call_svc.update_status(call_id, "on_hold")
    return {"status": "on_hold"}
