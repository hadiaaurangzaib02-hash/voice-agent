"""Call state manager + event logging against the shared dashboard database."""

from __future__ import annotations

import json
import logging
from typing import Any

from .. import db

log = logging.getLogger(__name__)

LIVE_STATUSES = (
    "dialing",
    "ringing",
    "connecting",
    "ai_answering",
    "listening",
    "thinking",
    "speaking",
    "on_hold",
    "escalated",
    "human_transfer",
    "active",
)


async def get_number_by_e164(e164: str) -> dict[str, Any] | None:
    row = await db.fetchrow(
        """
        select pn.id, pn.org_id, pn.provider_id, pn.agent_id, pn.e164,
               pn.inbound_webhook_secret, pn.status, pn.config,
               tp.kind as provider_kind, tp.credentials_ciphertext
        from public.phone_numbers pn
        join public.telephony_providers tp on tp.id = pn.provider_id
        where pn.e164 = $1
        limit 1
        """,
        e164,
    )
    return dict(row) if row else None


async def get_number_by_id(number_id: str) -> dict[str, Any] | None:
    row = await db.fetchrow(
        """
        select pn.id, pn.org_id, pn.provider_id, pn.agent_id, pn.e164,
               pn.inbound_webhook_secret, pn.status, pn.config,
               tp.kind as provider_kind, tp.credentials_ciphertext
        from public.phone_numbers pn
        join public.telephony_providers tp on tp.id = pn.provider_id
        where pn.id = $1::uuid
        """,
        number_id,
    )
    return dict(row) if row else None


async def get_agent(agent_id: str) -> dict[str, Any] | None:
    row = await db.fetchrow(
        """
        select id, org_id, name, system_prompt, greeting, voice, language,
               temperature, response_style, silence_timeout_seconds,
               fallback_prompt, is_active
        from public.agents where id = $1::uuid
        """,
        agent_id,
    )
    return dict(row) if row else None


async def create_call(
    *,
    org_id: str,
    agent_id: str | None,
    provider_id: str,
    phone_number_id: str,
    direction: str,
    external_call_id: str | None,
    caller_e164: str | None,
    callee_e164: str | None,
    status: str = "ringing",
    user_id: str | None = None,
) -> dict[str, Any]:
    owner = user_id or await db.fetchval(
        "select user_id from public.org_members where org_id = $1::uuid "
        "order by case role when 'owner' then 0 else 1 end limit 1",
        org_id,
    )
    row = await db.fetchrow(
        """
        insert into public.calls
          (org_id, agent_id, user_id, provider_id, phone_number_id, direction,
           channel, external_call_id, caller_e164, callee_e164, status, started_at)
        values ($1::uuid, $2::uuid, $3::uuid, $4::uuid, $5::uuid, $6, 'phone',
                $7, $8, $9, $10::public.call_status, now())
        returning id, org_id, agent_id, status, started_at
        """,
        org_id,
        agent_id,
        owner,
        provider_id,
        phone_number_id,
        direction,
        external_call_id,
        caller_e164,
        callee_e164,
        status,
    )
    return dict(row)


async def get_call_by_external_id(external_call_id: str) -> dict[str, Any] | None:
    row = await db.fetchrow(
        "select * from public.calls where external_call_id = $1 "
        "order by started_at desc limit 1",
        external_call_id,
    )
    return dict(row) if row else None


async def get_call(call_id: str) -> dict[str, Any] | None:
    row = await db.fetchrow("select * from public.calls where id = $1::uuid", call_id)
    return dict(row) if row else None


async def set_external_call_id(call_id: str, external_call_id: str) -> None:
    await db.execute(
        "update public.calls set external_call_id = $2, updated_at = now() "
        "where id = $1::uuid",
        call_id,
        external_call_id,
    )


async def set_recording_url(call_id: str, recording_url: str) -> None:
    await db.execute(
        "update public.calls set recording_url = $2, updated_at = now() "
        "where id = $1::uuid",
        call_id,
        recording_url,
    )


async def update_status(call_id: str, status: str, **extra: Any) -> None:
    sets = ["status = $2::public.call_status", "last_event_at = now()", "updated_at = now()"]
    args: list[Any] = [call_id, status]
    for key, value in extra.items():
        args.append(value)
        sets.append(f"{key} = ${len(args)}")
    await db.execute(
        f"update public.calls set {', '.join(sets)} where id = $1::uuid", *args
    )


async def end_call(call_id: str, status: str = "completed") -> None:
    await db.execute(
        """
        update public.calls
           set status = $2::public.call_status,
               ended_at = coalesce(ended_at, now()),
               duration_seconds = coalesce(
                   duration_seconds,
                   greatest(0, extract(epoch from (now() - started_at))::int)),
               last_event_at = now(),
               updated_at = now()
         where id = $1::uuid
        """,
        call_id,
        status,
    )


async def log_event(
    *,
    call_id: str,
    org_id: str,
    kind: str,
    payload: dict[str, Any] | None = None,
    agent_id: str | None = None,
) -> None:
    try:
        await db.execute(
            """
            insert into public.call_events (org_id, call_id, agent_id, kind, payload)
            values ($1::uuid, $2::uuid, $3::uuid, $4, $5::jsonb)
            """,
            org_id,
            call_id,
            agent_id,
            kind,
            json.dumps(payload or {}),
        )
    except Exception:  # noqa: BLE001 - telemetry must never break a live call
        log.exception("failed to write call_event kind=%s call=%s", kind, call_id)


async def add_message(
    *,
    call_id: str,
    org_id: str,
    role: str,
    content: str,
    latency_ms: int | None = None,
) -> None:
    await db.execute(
        """
        insert into public.call_messages (call_id, org_id, role, content, latency_ms)
        values ($1::uuid, $2::uuid, $3, $4, $5)
        """,
        call_id,
        org_id,
        role,
        content,
        latency_ms,
    )


async def transcript(call_id: str, limit: int = 40) -> list[dict[str, Any]]:
    rows = await db.fetch(
        "select role, content from public.call_messages where call_id = $1::uuid "
        "order by created_at asc limit $2",
        call_id,
        limit,
    )
    return db.rows_to_dicts(rows)


async def live_calls(org_id: str | None = None) -> list[dict[str, Any]]:
    statuses = list(LIVE_STATUSES)
    if org_id:
        rows = await db.fetch(
            "select * from public.calls where org_id = $1::uuid "
            "and status::text = any($2::text[]) and ended_at is null "
            "order by started_at desc",
            org_id,
            statuses,
        )
    else:
        rows = await db.fetch(
            "select * from public.calls where status::text = any($1::text[]) "
            "and ended_at is null order by started_at desc",
            statuses,
        )
    return db.rows_to_dicts(rows)


async def history(
    org_id: str, limit: int = 50, offset: int = 0
) -> list[dict[str, Any]]:
    rows = await db.fetch(
        "select * from public.calls where org_id = $1::uuid "
        "order by started_at desc limit $2 offset $3",
        org_id,
        limit,
        offset,
    )
    return db.rows_to_dicts(rows)
