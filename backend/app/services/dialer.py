"""Outbound dialer: single calls, retries, scheduling and bulk campaigns."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from ..config import get_settings
from ..security import ws_url_for
from . import calls, twilio_client

log = logging.getLogger(__name__)


def _urls(
    number_id: str, call_id: str, escalation_id: str | None = None
) -> tuple[str, str, str]:
    base = get_settings().public_base_url.rstrip("/")
    answer = f"{base}/webhooks/twilio/answer?number_id={number_id}&call_id={call_id}"
    if escalation_id:
        answer += f"&resume_escalation={escalation_id}"
    status = f"{base}/webhooks/twilio/status?call_id={call_id}"
    recording = f"{base}/webhooks/twilio/recording?call_id={call_id}"
    return answer, status, recording


async def place_call(
    *,
    number_id: str,
    to: str,
    agent_id: str | None = None,
    attempt: int = 1,
    max_retries: int | None = None,
    escalation_id: str | None = None,
) -> dict[str, Any]:
    from . import escalation as escalation_svc

    s = get_settings()
    number = await calls.get_number_by_id(number_id)
    if not number:
        raise ValueError("Phone number not found")
    if await escalation_svc.is_do_not_call(str(number["org_id"]), to):
        raise ValueError("This number is on the organisation's do-not-call list")
    resolved_agent = agent_id or number.get("agent_id")
    if not resolved_agent:
        raise ValueError("Assign an agent to this number before dialing")

    call = await calls.create_call(
        org_id=str(number["org_id"]),
        agent_id=str(resolved_agent),
        provider_id=str(number["provider_id"]),
        phone_number_id=str(number["id"]),
        direction="outbound",
        external_call_id=None,
        caller_e164=number["e164"],
        callee_e164=to,
        status="dialing",
    )
    call_id = str(call["id"])
    answer_url, status_url, recording_url = _urls(number_id, call_id, escalation_id)
    creds = twilio_client.credentials_for(number)

    try:
        result = await twilio_client.create_call(
            creds,
            to=to,
            from_=number["e164"],
            answer_url=answer_url,
            status_callback_url=status_url,
            recording_callback_url=recording_url if s.automatic_recording else None,
        )
    except Exception as exc:  # noqa: BLE001
        await calls.log_event(
            call_id=call_id,
            org_id=str(number["org_id"]),
            kind="error",
            payload={"stage": "outbound_dial", "message": str(exc), "attempt": attempt},
        )
        await calls.end_call(call_id, "failed")
        retries = s.outbound_retry_max if max_retries is None else max_retries
        if attempt <= retries:
            asyncio.create_task(
                _retry_later(
                    number_id=number_id,
                    to=to,
                    agent_id=resolved_agent,
                    attempt=attempt + 1,
                    max_retries=retries,
                    delay=s.outbound_retry_delay_seconds,
                    escalation_id=escalation_id,
                )
            )
        raise

    sid = result.get("sid")
    if sid:
        await calls.set_external_call_id(call_id, sid)
    await calls.log_event(
        call_id=call_id,
        org_id=str(number["org_id"]),
        agent_id=str(resolved_agent),
        kind="status",
        payload={"status": "ringing", "to": to, "from": number["e164"], "attempt": attempt},
    )
    await calls.update_status(call_id, "ringing")
    return {"call_id": call_id, "external_call_id": sid, "status": "ringing"}


async def _retry_later(
    *,
    number_id: str,
    to: str,
    agent_id: str | None,
    attempt: int,
    max_retries: int,
    delay: int,
    escalation_id: str | None,
) -> None:
    await asyncio.sleep(delay)
    try:
        await place_call(
            number_id=number_id,
            to=to,
            agent_id=agent_id,
            attempt=attempt,
            max_retries=max_retries,
            escalation_id=escalation_id,
        )
    except Exception:  # noqa: BLE001
        log.warning("outbound retry %s failed for %s", attempt, to)


async def schedule_call(*, run_at: datetime, **kwargs: Any) -> None:
    delay = max(0.0, (run_at - datetime.now(timezone.utc)).total_seconds())
    await asyncio.sleep(delay)
    await place_call(**kwargs)


async def run_campaign(
    *,
    number_id: str,
    targets: list[str],
    agent_id: str | None,
    concurrency: int | None,
    max_retries: int | None,
) -> str:
    campaign_id = str(uuid.uuid4())
    limit = concurrency or get_settings().campaign_concurrency
    sem = asyncio.Semaphore(limit)

    async def dial(target: str) -> None:
        async with sem:
            try:
                await place_call(
                    number_id=number_id,
                    to=target,
                    agent_id=agent_id,
                    max_retries=max_retries,
                )
            except Exception:  # noqa: BLE001
                log.warning("campaign %s: dial failed for %s", campaign_id, target)
            await asyncio.sleep(1.0)  # gentle pacing to respect carrier limits

    asyncio.create_task(_gather(campaign_id, [dial(t) for t in targets]))
    return campaign_id


async def _gather(campaign_id: str, tasks: list[Any]) -> None:
    await asyncio.gather(*tasks, return_exceptions=True)
    log.info("campaign %s finished", campaign_id)
