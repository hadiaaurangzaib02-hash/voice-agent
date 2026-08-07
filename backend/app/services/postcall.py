"""Post-call CRM persistence: summary, sentiment, lead profile, notes, follow-up tasks."""

from __future__ import annotations

import json
import logging
from typing import Any

from .. import db
from . import calls as call_svc, llm

log = logging.getLogger(__name__)

ANALYSIS_PROMPT = (
    "You analyse finished customer phone calls. Return ONLY compact JSON with keys: "
    'summary (2 sentences), intent (short label), sentiment (one of positive, neutral, '
    'negative), lead_status (one of new, qualified, contacted, won, lost), '
    "caller_name (string or null), follow_up (short next action or null)."
)


async def finalize(call_id: str) -> None:
    """Runs once after a call ends. Never raises — CRM writes must not break telephony."""
    try:
        call = await call_svc.get_call(call_id)
        if not call or call.get("summary"):
            return
        history = await call_svc.transcript(call_id, limit=200)
        if not history:
            return
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)

        chunks: list[str] = []
        async for part in llm.stream_completion(
            [
                {"role": "system", "content": ANALYSIS_PROMPT},
                {"role": "user", "content": transcript[:12000]},
            ]
        ):
            chunks.append(part)
        raw = " ".join(chunks).strip()
        data = _parse_json(raw)
        if not data:
            return

        sentiment = str(data.get("sentiment") or "neutral").lower()
        if sentiment not in ("positive", "neutral", "negative"):
            sentiment = "neutral"

        await db.execute(
            """
            update public.calls
               set summary = $2,
                   intent = coalesce($3, intent),
                   current_intent = coalesce($3, current_intent),
                   sentiment = $4::public.sentiment_label,
                   caller_name = coalesce($5, caller_name),
                   action_items = $6::jsonb,
                   updated_at = now()
             where id = $1::uuid
            """,
            call_id,
            data.get("summary"),
            data.get("intent"),
            sentiment,
            data.get("caller_name"),
            json.dumps([data["follow_up"]] if data.get("follow_up") else []),
        )

        await _upsert_lead(call, data)

        await call_svc.log_event(
            call_id=call_id,
            org_id=str(call["org_id"]),
            kind="crm_action",
            payload={
                "action": "call_saved",
                "intent": data.get("intent"),
                "sentiment": sentiment,
                "lead_status": data.get("lead_status"),
                "follow_up": data.get("follow_up"),
            },
        )
    except Exception:  # noqa: BLE001
        log.exception("post-call CRM persistence failed for %s", call_id)


async def _upsert_lead(call: dict[str, Any], data: dict[str, Any]) -> None:
    phone = (
        call.get("caller_e164")
        if call.get("direction") == "inbound"
        else call.get("callee_e164")
    )
    if not phone:
        return
    note = f"Call {call['id']}: {data.get('summary') or ''}".strip()
    try:
        existing = await db.fetchrow(
            "select id, notes from public.leads where org_id = $1::uuid and phone = $2 limit 1",
            str(call["org_id"]),
            phone,
        )
        if existing:
            await db.execute(
                "update public.leads set notes = concat_ws(E'\\n', notes, $2), "
                "updated_at = now() where id = $1::uuid",
                str(existing["id"]),
                note,
            )
        else:
            await db.execute(
                """
                insert into public.leads (org_id, full_name, phone, status, notes, source)
                values ($1::uuid, $2, $3, coalesce($4, 'new'), $5, 'inbound_call')
                """,
                str(call["org_id"]),
                data.get("caller_name") or phone,
                phone,
                data.get("lead_status"),
                note,
            )
    except Exception:  # noqa: BLE001
        log.warning("lead upsert skipped", exc_info=True)


def _parse_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:  # noqa: BLE001
        return None
