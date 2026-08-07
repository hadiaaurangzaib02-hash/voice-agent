"""Conversation memory that persists across calls for the same phone number."""

from __future__ import annotations

import logging
from typing import Any

from .. import db

log = logging.getLogger(__name__)


async def caller_history(
    *, org_id: str, e164: str | None, exclude_call_id: str, max_calls: int = 3
) -> str:
    """Compact recap of this caller's previous calls, injected into the system prompt."""
    if not e164:
        return ""
    rows = await db.fetch(
        """
        select c.id, c.started_at, c.summary, c.current_intent, c.sentiment
          from public.calls c
         where c.org_id = $1::uuid
           and (c.caller_e164 = $2 or c.callee_e164 = $2)
           and c.id <> $3::uuid
           and c.ended_at is not null
         order by c.started_at desc
         limit $4
        """,
        org_id,
        e164,
        exclude_call_id,
        max_calls,
    )
    if not rows:
        return ""

    lines: list[str] = []
    for row in rows:
        summary = (row["summary"] or "").strip()
        if not summary:
            msgs = await db.fetch(
                "select role, content from public.call_messages where call_id = $1::uuid "
                "order by created_at asc limit 12",
                str(row["id"]),
            )
            summary = " ".join(f"{m['role']}: {m['content']}" for m in msgs)[:400]
        if not summary:
            continue
        stamp = row["started_at"].strftime("%Y-%m-%d") if row["started_at"] else ""
        intent = row["current_intent"] or ""
        lines.append(f"- {stamp} ({intent}): {summary}")

    return "\n".join(lines)


async def caller_profile(*, org_id: str, e164: str | None) -> dict[str, Any] | None:
    if not e164:
        return None
    try:
        row = await db.fetchrow(
            "select * from public.leads where org_id = $1::uuid and phone = $2 limit 1",
            org_id,
            e164,
        )
    except Exception:  # noqa: BLE001
        log.warning("caller profile lookup failed", exc_info=True)
        return None
    return dict(row) if row else None
