"""Twilio Voice webhooks: inbound answer, outbound answer, status callbacks."""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request, Response

from ..config import get_settings
from ..security import media_stream_token, validate_twilio_signature, ws_url_for
from ..services import calls, escalation, twilio_client

log = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks/twilio", tags=["webhooks"])

XML = "text/xml"


async def _form(request: Request) -> dict[str, str]:
    form = await request.form()
    return {k: str(v) for k, v in form.items()}


def _public_url(request: Request) -> str:
    base = get_settings().public_base_url.rstrip("/")
    return base + request.url.path + (f"?{request.url.query}" if request.url.query else "")


async def _verify(request: Request, params: dict[str, str], number: dict | None) -> bool:
    s = get_settings()
    if not s.twilio_validate_signature:
        return True
    try:
        creds = twilio_client.credentials_for(number)
    except Exception:  # noqa: BLE001
        log.error("signature validation impossible: no credentials")
        return False
    return validate_twilio_signature(
        auth_token=creds.auth_token,
        url=_public_url(request),
        params=params,
        signature=request.headers.get("X-Twilio-Signature"),
    )


def _after_hours_message(number: dict) -> str | None:
    config = number.get("config") or {}
    hours = config.get("business_hours") if isinstance(config, dict) else None
    if not isinstance(hours, dict) or hours.get("always_on", True):
        return None
    try:
        now = datetime.now(ZoneInfo(str(hours.get("timezone") or "UTC")))
        day_index = (now.weekday() + 1) % 7
        day = next(
            (d for d in hours.get("days", []) if int(d.get("day", -1)) == day_index),
            None,
        )
        if day and day.get("enabled"):
            current = now.strftime("%H:%M")
            if str(day.get("open", "00:00")) <= current < str(day.get("close", "00:00")):
                return None
    except Exception:  # noqa: BLE001
        log.exception("invalid business hours for number %s", number.get("id"))
    return str(
        hours.get("after_hours_message")
        or "Thanks for calling. We are currently closed. Please call again during business hours."
    )


@router.post("/voice")
async def inbound_voice(request: Request) -> Response:
    """Twilio hits this when a call arrives on a configured business number."""
    params = await _form(request)
    to_number = params.get("To") or ""
    from_number = params.get("From") or ""
    call_sid = params.get("CallSid")

    number = await calls.get_number_by_e164(to_number)
    if not number:
        log.warning("inbound call to unknown number %s", to_number)
        return Response(
            twilio_client.twiml_say_hangup(
                "This number is not configured for automated answering. Goodbye."
            ),
            media_type=XML,
        )

    if not await _verify(request, params, number):
        log.error("invalid Twilio signature for inbound call to %s", to_number)
        return Response("Invalid signature", status_code=403)

    after_hours = _after_hours_message(number)
    if after_hours:
        return Response(twilio_client.twiml_say_hangup(after_hours), media_type=XML)

    if not number.get("agent_id"):
        call = await calls.create_call(
            org_id=str(number["org_id"]),
            agent_id=None,
            provider_id=str(number["provider_id"]),
            phone_number_id=str(number["id"]),
            direction="inbound",
            external_call_id=call_sid,
            caller_e164=from_number,
            callee_e164=to_number,
            status="missed",
        )
        await calls.log_event(
            call_id=str(call["id"]),
            org_id=str(number["org_id"]),
            kind="error",
            payload={"stage": "routing", "message": "No agent assigned to this number"},
        )
        await calls.end_call(str(call["id"]), "missed")
        return Response(
            twilio_client.twiml_say_hangup(
                "No agent is available for this number right now. Please try again later."
            ),
            media_type=XML,
        )

    call = await calls.create_call(
        org_id=str(number["org_id"]),
        agent_id=str(number["agent_id"]),
        provider_id=str(number["provider_id"]),
        phone_number_id=str(number["id"]),
        direction="inbound",
        external_call_id=call_sid,
        caller_e164=from_number,
        callee_e164=to_number,
        status="ringing",
    )
    call_id = str(call["id"])
    await calls.log_event(
        call_id=call_id,
        org_id=str(number["org_id"]),
        agent_id=str(number["agent_id"]),
        kind="status",
        payload={"status": "ringing", "from": from_number, "to": to_number, "sid": call_sid},
    )

    stream_url = ws_url_for(
        f"/ws/media-stream?call_id={call_id}&token={media_stream_token(call_id)}"
    )
    status_url = f"{get_settings().public_base_url.rstrip('/')}/webhooks/twilio/status?call_id={call_id}"
    await calls.update_status(call_id, "connecting")
    if get_settings().automatic_recording and call_sid:
        try:
            base = get_settings().public_base_url.rstrip("/")
            await twilio_client.start_recording(
                twilio_client.credentials_for(number),
                call_sid,
                callback_url=f"{base}/webhooks/twilio/recording?call_id={call_id}",
            )
        except Exception as exc:  # noqa: BLE001
            await calls.log_event(
                call_id=call_id,
                org_id=str(number["org_id"]),
                kind="error",
                payload={"stage": "recording_start", "message": str(exc)},
            )
    return Response(
        twilio_client.twiml_stream_answer(
            stream_url=stream_url, status_url=status_url, call_id=call_id
        ),
        media_type=XML,
    )


@router.post("/answer")
async def outbound_answer(request: Request) -> Response:
    """TwiML served when an outbound call we placed is answered."""
    params = await _form(request)
    call_id = request.query_params.get("call_id") or ""
    number_id = request.query_params.get("number_id") or ""
    number = await calls.get_number_by_id(number_id) if number_id else None

    if not await _verify(request, params, number):
        return Response("Invalid signature", status_code=403)

    call = await calls.get_call(call_id) if call_id else None
    if not call:
        return Response(
            twilio_client.twiml_say_hangup("This call is no longer available. Goodbye."),
            media_type=XML,
        )

    if params.get("AnsweredBy", "").startswith("machine"):
        await calls.log_event(
            call_id=call_id,
            org_id=str(call["org_id"]),
            kind="status",
            payload={"status": "voicemail_detected"},
        )

    sid = params.get("CallSid")
    if sid:
        await calls.set_external_call_id(call_id, sid)
    await calls.update_status(call_id, "connecting")

    resume = request.query_params.get("resume_escalation")
    suffix = f"&resume_escalation={resume}" if resume else ""
    stream_url = ws_url_for(
        f"/ws/media-stream?call_id={call_id}&token={media_stream_token(call_id)}{suffix}"
    )
    status_url = f"{get_settings().public_base_url.rstrip('/')}/webhooks/twilio/status?call_id={call_id}"
    return Response(
        twilio_client.twiml_stream_answer(
            stream_url=stream_url, status_url=status_url, call_id=call_id
        ),
        media_type=XML,
    )


@router.post("/hold")
async def hold(request: Request) -> Response:
    """Professional hold music while a supervisor is consulted.

    Every music loop re-enters this endpoint; as soon as the operator answers in the
    dashboard the caller is reconnected to the agent, which says "Thank you for waiting."
    """
    s = get_settings()
    call_id = request.query_params.get("call_id") or ""
    escalation_id = request.query_params.get("escalation_id") or ""
    loops = int(request.query_params.get("loops") or "0")
    base = s.public_base_url.rstrip("/")

    row = await escalation.get_escalation(escalation_id) if escalation_id else None
    call = await calls.get_call(call_id) if call_id else None
    if not row or not call:
        return Response(
            twilio_client.twiml_say_hangup("Thanks for your patience. We'll follow up shortly. Goodbye."),
            media_type=XML,
        )

    if row.get("operator_answer") and row.get("status") in ("answered", "calling_back"):
        status_url = f"{base}/webhooks/twilio/status?call_id={call_id}"
        stream_url = ws_url_for(
            f"/ws/media-stream?call_id={call_id}&token={media_stream_token(call_id)}"
            f"&resume_escalation={escalation_id}"
        )
        await calls.update_status(call_id, "connecting")
        return Response(
            twilio_client.twiml_stream_answer(
                stream_url=stream_url, status_url=status_url, call_id=call_id
            ),
            media_type=XML,
        )

    # Roughly 30 s of music per loop; bail out to a callback promise after the cap.
    if loops * 30 >= s.hold_max_seconds:
        await calls.log_event(
            call_id=call_id,
            org_id=str(call["org_id"]),
            kind="escalation",
            payload={"escalation_id": escalation_id, "status": "hold_timeout"},
        )
        await calls.end_call(call_id, "completed")
        return Response(
            twilio_client.twiml_say_hangup(
                "Thank you for waiting. My supervisor is still checking, so we'll call "
                "you back as soon as we have the answer. Goodbye."
            ),
            media_type=XML,
        )

    poll_url = (
        f"{base}/webhooks/twilio/hold?call_id={call_id}"
        f"&escalation_id={escalation_id}&loops={loops + 1}"
    )
    return Response(
        twilio_client.twiml_hold(music_url=s.hold_music_url, poll_url=poll_url),
        media_type=XML,
    )


@router.post("/conference")
async def conference_twiml(request: Request) -> Response:
    """Moves a live call into a conference so supervisors can listen/whisper/join."""
    name = request.query_params.get("name") or ""
    if not name:
        return Response(
            twilio_client.twiml_say_hangup("Conference unavailable. Goodbye."),
            media_type=XML,
        )
    base = get_settings().public_base_url.rstrip("/")
    call_id = request.query_params.get("call_id") or ""
    return Response(
        twilio_client.twiml_conference(
            name, status_url=f"{base}/webhooks/twilio/status?call_id={call_id}"
        ),
        media_type=XML,
    )


@router.post("/supervisor")
async def supervisor_twiml(request: Request) -> Response:
    """TwiML for the supervisor leg: listen (muted), whisper (coach) or join."""
    name = request.query_params.get("name") or ""
    mode = request.query_params.get("mode") or "listen"
    coach_sid = request.query_params.get("coach")
    if not name:
        return Response(
            twilio_client.twiml_say_hangup("Conference unavailable. Goodbye."),
            media_type=XML,
        )
    announce = {
        "listen": "You are listening to this call. Your microphone is muted.",
        "whisper": "Whisper mode. Only the agent can hear you.",
        "join": "Joining the call now.",
        "takeover": "You are taking over this call now.",
    }.get(mode)
    return Response(
        twilio_client.twiml_supervisor_conference(
            name,
            muted=mode == "listen",
            coach_sid=coach_sid if mode == "whisper" else None,
            announce=announce,
        ),
        media_type=XML,
    )


@router.post("/status")
async def status_callback(request: Request) -> Response:
    params = await _form(request)
    call_id = request.query_params.get("call_id")
    call = None
    if call_id:
        call = await calls.get_call(call_id)
    elif params.get("CallSid"):
        call = await calls.get_call_by_external_id(params["CallSid"])
    if not call:
        return Response(status_code=204)
    number = (
        await calls.get_number_by_id(str(call["phone_number_id"]))
        if call.get("phone_number_id")
        else None
    )
    if not await _verify(request, params, number):
        return Response("Invalid signature", status_code=403)

    call_status = (params.get("CallStatus") or "").lower()
    await calls.log_event(
        call_id=str(call["id"]),
        org_id=str(call["org_id"]),
        kind="status",
        payload={"provider_status": call_status, **{k: v for k, v in params.items() if k.startswith(("Recording", "Duration", "Sip"))}},
    )
    terminal = {
        "completed": "completed",
        "busy": "missed",
        "no-answer": "missed",
        "failed": "failed",
        "canceled": "failed",
    }
    if call_status in terminal:
        await calls.end_call(str(call["id"]), terminal[call_status])
    elif call_status == "in-progress":
        await calls.update_status(str(call["id"]), "active")
    return Response(status_code=204)


@router.post("/recording")
async def recording_callback(request: Request) -> Response:
    params = await _form(request)
    call_id = request.query_params.get("call_id") or ""
    call = await calls.get_call(call_id) if call_id else None
    if not call:
        return Response(status_code=204)
    number = (
        await calls.get_number_by_id(str(call["phone_number_id"]))
        if call.get("phone_number_id")
        else None
    )
    if not await _verify(request, params, number):
        return Response("Invalid signature", status_code=403)
    recording_url = params.get("RecordingUrl")
    if recording_url:
        await calls.set_recording_url(call_id, f"{recording_url}.mp3")
    await calls.log_event(
        call_id=call_id,
        org_id=str(call["org_id"]),
        kind="recording",
        payload={
            "status": params.get("RecordingStatus"),
            "recording_sid": params.get("RecordingSid"),
            "duration": params.get("RecordingDuration"),
        },
    )
    return Response(status_code=204)


@router.post("/transfer")
async def transfer_twiml(request: Request) -> Response:
    """Redirect target used by POST /api/calls/transfer."""
    to = request.query_params.get("to") or ""
    caller_id = request.query_params.get("caller_id")
    announce = request.query_params.get("announce")
    if not to:
        return Response(
            twilio_client.twiml_say_hangup("Transfer target missing. Goodbye."),
            media_type=XML,
        )
    xml = twilio_client.twiml_dial(to, caller_id)
    if announce:
        xml = xml.replace(
            "<Response>", f'<Response><Say voice="Polly.Joanna">{announce}</Say>', 1
        )
    return Response(xml, media_type=XML)
