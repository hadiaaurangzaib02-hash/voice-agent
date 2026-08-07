"""Unknown-answer escalation: hold music, Gmail notification, operator reply, callback.

Flow (all real, no simulation):
1. The AI emits [[ESCALATE]] when the knowledge base cannot answer.
2. The caller hears "Please give me a moment while I verify this information."
3. The live call is redirected to hold music while an escalation row is created.
4. An email is sent through Gmail SMTP with number, question, transcript, summary, call id.
5. The dashboard operator is notified through the escalations table (Supabase Realtime).
6. When the operator answers, the caller is either resumed on the live call or called
   back automatically ("Thank you for waiting." + the answer), retrying every 2 hours.
7. Callers who ask not to be contacted again are stored permanently in do_not_call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any

from .. import db
from ..config import get_settings
from . import calls as call_svc

log = logging.getLogger(__name__)

HOLD_MESSAGE = "Please give me a moment while I verify this information."
RESUME_MESSAGE = "Thank you for waiting."

DNC_PHRASES = (
    "do not call me again",
    "don't call me again",
    "dont call me again",
    "stop calling me",
    "never call me again",
    "remove me from your list",
    "take me off your list",
)


def detects_do_not_call(text: str) -> bool:
    lowered = " ".join(text.lower().split())
    return any(phrase in lowered for phrase in DNC_PHRASES)


async def add_do_not_call(
    *, org_id: str, e164: str | None, reason: str, call_id: str | None
) -> None:
    if not e164:
        return
    await db.execute(
        """
        insert into public.do_not_call (org_id, e164, reason, source, call_id)
        values ($1::uuid, $2, $3, 'customer_request', $4::uuid)
        on conflict (org_id, e164) do update set reason = excluded.reason
        """,
        org_id,
        e164,
        reason,
        call_id,
    )


async def is_do_not_call(org_id: str, e164: str) -> bool:
    return bool(
        await db.fetchval(
            "select 1 from public.do_not_call where org_id = $1::uuid and e164 = $2",
            org_id,
            e164,
        )
    )


# ------------------------------------------------------------------ creation


async def summarize(transcript: str) -> str:
    """Short conversation summary using the configured LLM (falls back to a slice)."""
    from . import llm

    try:
        chunks: list[str] = []
        async for part in llm.stream_completion(
            [
                {
                    "role": "system",
                    "content": "Summarize this phone conversation in 2 short sentences for a human supervisor.",
                },
                {"role": "user", "content": transcript[:6000]},
            ]
        ):
            chunks.append(part)
        summary = " ".join(chunks).strip()
        if summary:
            return summary
    except Exception:  # noqa: BLE001
        log.warning("escalation summary failed", exc_info=True)
    return transcript[-600:]


async def create_escalation(
    *,
    call_id: str,
    org_id: str,
    agent_id: str | None,
    question: str,
    customer_e164: str | None,
) -> dict[str, Any]:
    history = await call_svc.transcript(call_id)
    transcript_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history
    )
    summary = await summarize(transcript_text or question)

    assigned_to = await db.fetchval(
        "select user_id from public.org_members where org_id = $1::uuid "
        "order by case role when 'owner' then 0 when 'admin' then 1 else 2 end limit 1",
        org_id,
    )

    row = await db.fetchrow(
        """
        insert into public.escalations
          (org_id, call_id, agent_id, customer_e164, question, transcript, summary,
           status, assigned_to)
        values ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6, $7, 'pending', $8::uuid)
        returning *
        """,
        org_id,
        call_id,
        agent_id,
        customer_e164,
        question,
        transcript_text,
        summary,
        assigned_to,
    )
    escalation = dict(row)

    await call_svc.log_event(
        call_id=call_id,
        org_id=org_id,
        agent_id=agent_id,
        kind="escalation",
        payload={
            "escalation_id": str(escalation["id"]),
            "question": question,
            "status": "pending",
        },
    )

    asyncio.create_task(send_escalation_email(escalation))
    return escalation


# --------------------------------------------------------------------- email


def _send_smtp(message: EmailMessage) -> None:
    s = get_settings()
    with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=30) as server:
        server.starttls()
        server.login(s.smtp_username or "", s.smtp_password or "")
        server.send_message(message)


async def send_escalation_email(escalation: dict[str, Any]) -> None:
    s = get_settings()
    recipient = s.escalation_email_to or s.smtp_username
    if not (s.smtp_username and s.smtp_password and recipient):
        await db.execute(
            "update public.escalations set email_error = $2 where id = $1::uuid",
            str(escalation["id"]),
            "Gmail SMTP is not configured (SMTP_USERNAME / SMTP_PASSWORD / ESCALATION_EMAIL_TO)",
        )
        log.warning("escalation email skipped: SMTP not configured")
        return

    msg = EmailMessage()
    msg["Subject"] = f"[AI Voice Agent] Unanswered question from {escalation.get('customer_e164') or 'caller'}"
    msg["From"] = s.smtp_from or s.smtp_username
    msg["To"] = recipient
    msg.set_content(
        "\n".join(
            [
                "The AI agent could not answer a customer question and placed the caller on hold.",
                "",
                f"Call ID:          {escalation.get('call_id')}",
                f"Escalation ID:    {escalation.get('id')}",
                f"Customer number:  {escalation.get('customer_e164') or 'unknown'}",
                "",
                "QUESTION",
                str(escalation.get("question") or ""),
                "",
                "SUMMARY",
                str(escalation.get("summary") or ""),
                "",
                "FULL TRANSCRIPT",
                str(escalation.get("transcript") or ""),
                "",
                f"Answer it in the dashboard: {s.dashboard_url.rstrip('/')}/escalations",
            ]
        )
    )

    try:
        await asyncio.to_thread(_send_smtp, msg)
        await db.execute(
            "update public.escalations set email_sent_at = now(), email_error = null "
            "where id = $1::uuid",
            str(escalation["id"]),
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("escalation email failed")
        await db.execute(
            "update public.escalations set email_error = $2 where id = $1::uuid",
            str(escalation["id"]),
            str(exc)[:500],
        )


# ------------------------------------------------------------------ answering


async def get_escalation(escalation_id: str) -> dict[str, Any] | None:
    row = await db.fetchrow(
        "select * from public.escalations where id = $1::uuid", escalation_id
    )
    return dict(row) if row else None


async def pending_for_call(call_id: str) -> dict[str, Any] | None:
    row = await db.fetchrow(
        "select * from public.escalations where call_id = $1::uuid "
        "and status in ('pending','answered') order by created_at desc limit 1",
        call_id,
    )
    return dict(row) if row else None


async def record_answer(
    *, escalation_id: str, answer: str, answered_by: str | None
) -> dict[str, Any]:
    row = await db.fetchrow(
        """
        update public.escalations
           set operator_answer = $2,
               answered_by = $3::uuid,
               answered_at = now(),
               status = 'answered',
               next_retry_at = null
         where id = $1::uuid
        returning *
        """,
        escalation_id,
        answer,
        answered_by,
    )
    if not row:
        raise ValueError("Escalation not found")
    escalation = dict(row)
    await call_svc.log_event(
        call_id=str(escalation["call_id"]),
        org_id=str(escalation["org_id"]),
        agent_id=str(escalation["agent_id"]) if escalation.get("agent_id") else None,
        kind="escalation",
        payload={"escalation_id": escalation_id, "status": "answered"},
    )
    return escalation


async def mark_resolved(escalation_id: str, status: str = "resolved") -> None:
    await db.execute(
        "update public.escalations set status = $2, resolved_at = now() where id = $1::uuid",
        escalation_id,
        status,
    )


async def schedule_callback(escalation: dict[str, Any], *, delay_seconds: int = 0) -> None:
    """Call the customer back with the operator's answer; retry every 2 hours."""
    asyncio.create_task(_callback_worker(str(escalation["id"]), delay_seconds))


async def _callback_worker(escalation_id: str, delay_seconds: int) -> None:
    from . import dialer

    s = get_settings()
    if delay_seconds:
        await asyncio.sleep(delay_seconds)

    while True:
        escalation = await get_escalation(escalation_id)
        if not escalation or escalation["status"] not in ("answered", "calling_back"):
            return
        customer = escalation.get("customer_e164")
        org_id = str(escalation["org_id"])
        if not customer:
            await mark_resolved(escalation_id, "resolved")
            return
        if await is_do_not_call(org_id, customer):
            await mark_resolved(escalation_id, "do_not_call")
            return

        attempts = int(escalation.get("callback_attempts") or 0) + 1
        original = await call_svc.get_call(str(escalation["call_id"])) if escalation.get("call_id") else None
        number_id = str(original["phone_number_id"]) if original and original.get("phone_number_id") else None
        if not number_id:
            await mark_resolved(escalation_id, "failed")
            return

        try:
            result = await dialer.place_call(
                number_id=number_id,
                to=customer,
                agent_id=str(escalation["agent_id"]) if escalation.get("agent_id") else None,
                max_retries=0,
                escalation_id=escalation_id,
            )
            await db.execute(
                """
                update public.escalations
                   set status = 'calling_back', callback_attempts = $2,
                       callback_call_id = $3::uuid,
                       next_retry_at = now() + make_interval(hours => $4)
                 where id = $1::uuid
                """,
                escalation_id,
                attempts,
                result["call_id"],
                s.callback_retry_hours,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("callback attempt %s failed: %s", attempts, exc)
            await db.execute(
                "update public.escalations set callback_attempts = $2, "
                "next_retry_at = now() + make_interval(hours => $3) where id = $1::uuid",
                escalation_id,
                attempts,
                s.callback_retry_hours,
            )

        if attempts >= s.callback_max_attempts:
            await mark_resolved(escalation_id, "unreachable")
            return

        await asyncio.sleep(s.callback_retry_hours * 3600)

        current = await get_escalation(escalation_id)
        if not current or current["status"] in ("resolved", "do_not_call", "unreachable"):
            return


async def resume_text(escalation: dict[str, Any]) -> str:
    answer = (escalation.get("operator_answer") or "").strip()
    return f"{RESUME_MESSAGE} {answer}".strip()


async def due_retries() -> list[dict[str, Any]]:
    rows = await db.fetch(
        "select * from public.escalations where status = 'calling_back' "
        "and next_retry_at is not null and next_retry_at <= now() limit 50"
    )
    return [dict(r) for r in rows]


def escalation_payload(escalation: dict[str, Any]) -> str:
    return json.dumps(
        {k: str(v) for k, v in escalation.items() if v is not None}, default=str
    )


def next_retry_at(hours: int | None = None) -> datetime:
    h = hours or get_settings().callback_retry_hours
    return datetime.now(timezone.utc) + timedelta(hours=h)
