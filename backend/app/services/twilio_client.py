"""Twilio REST + TwiML generation. Credentials resolved per provider row."""

from __future__ import annotations

import logging
from typing import Any
from xml.sax.saxutils import escape

import httpx

from ..config import get_settings
from ..security import decrypt_credentials

log = logging.getLogger(__name__)

API_ROOT = "https://api.twilio.com/2010-04-01"


class TwilioCredentials:
    def __init__(self, account_sid: str, auth_token: str) -> None:
        self.account_sid = account_sid
        self.auth_token = auth_token


def credentials_for(provider_row: dict[str, Any] | None) -> TwilioCredentials:
    s = get_settings()
    if provider_row and provider_row.get("credentials_ciphertext"):
        creds = decrypt_credentials(provider_row["credentials_ciphertext"])
        sid = (
            creds.get("accountSid")
            or creds.get("account_sid")
            or creds.get("apiKey")
            or ""
        )
        token = creds.get("authToken") or creds.get("auth_token") or creds.get("apiSecret") or ""
        if sid and token:
            return TwilioCredentials(str(sid), str(token))
    if s.twilio_account_sid and s.twilio_auth_token:
        return TwilioCredentials(s.twilio_account_sid, s.twilio_auth_token)
    raise RuntimeError("No Twilio credentials available for this provider")


async def _post(creds: TwilioCredentials, path: str, data: dict[str, Any]) -> dict[str, Any]:
    url = f"{API_ROOT}/Accounts/{creds.account_sid}{path}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            url,
            data={k: v for k, v in data.items() if v is not None},
            auth=(creds.account_sid, creds.auth_token),
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Twilio {path} failed {resp.status_code}: {resp.text[:400]}")
    return resp.json()


async def create_call(
    creds: TwilioCredentials,
    *,
    to: str,
    from_: str,
    answer_url: str,
    status_callback_url: str,
    machine_detection: bool = True,
    timeout: int = 45,
    recording_callback_url: str | None = None,
) -> dict[str, Any]:
    return await _post(
        creds,
        "/Calls.json",
        {
            "To": to,
            "From": from_,
            "Url": answer_url,
            "Method": "POST",
            "StatusCallback": status_callback_url,
            "StatusCallbackMethod": "POST",
            "StatusCallbackEvent": "initiated ringing answered completed",
            "Timeout": timeout,
            "MachineDetection": "DetectMessageEnd" if machine_detection else None,
            "AsyncAmd": "true" if machine_detection else None,
            "Record": "true" if recording_callback_url else None,
            "RecordingStatusCallback": recording_callback_url,
            "RecordingStatusCallbackMethod": "POST" if recording_callback_url else None,
            "RecordingStatusCallbackEvent": "completed" if recording_callback_url else None,
        },
    )


async def hangup(creds: TwilioCredentials, call_sid: str) -> dict[str, Any]:
    return await _post(creds, f"/Calls/{call_sid}.json", {"Status": "completed"})


async def redirect_call(
    creds: TwilioCredentials, call_sid: str, url: str
) -> dict[str, Any]:
    return await _post(creds, f"/Calls/{call_sid}.json", {"Url": url, "Method": "POST"})


async def start_recording(
    creds: TwilioCredentials, call_sid: str, callback_url: str | None = None
) -> dict[str, Any]:
    return await _post(
        creds,
        f"/Calls/{call_sid}/Recordings.json",
        {
            "RecordingChannels": "dual",
            "RecordingTrack": "both",
            "RecordingStatusCallback": callback_url,
            "RecordingStatusCallbackMethod": "POST" if callback_url else None,
            "RecordingStatusCallbackEvent": "completed" if callback_url else None,
        },
    )


async def stop_recording(
    creds: TwilioCredentials, call_sid: str, recording_sid: str
) -> dict[str, Any]:
    return await _post(
        creds,
        f"/Calls/{call_sid}/Recordings/{recording_sid}.json",
        {"Status": "stopped"},
    )


async def send_sms(
    creds: TwilioCredentials, *, to: str, from_: str, body: str
) -> dict[str, Any]:
    return await _post(creds, "/Messages.json", {"To": to, "From": from_, "Body": body})


# --------------------------- TwiML builders ---------------------------


def twiml_stream_answer(*, stream_url: str, status_url: str, call_id: str) -> str:
    """Bidirectional Media Stream: Twilio sends and receives audio over the WS."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Connect action="{escape(status_url)}">'
        f'<Stream url="{escape(stream_url)}">'
        f'<Parameter name="callId" value="{escape(call_id)}" />'
        "</Stream>"
        "</Connect>"
        "</Response>"
    )


def twiml_say_hangup(message: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Say voice="Polly.Joanna">{escape(message)}</Say>'
        "<Hangup/>"
        "</Response>"
    )


def twiml_hold(*, music_url: str, poll_url: str, say: str | None = None) -> str:
    """Professional hold music that re-polls the backend for the operator's answer."""
    intro = f'<Say voice="Polly.Joanna">{escape(say)}</Say>' if say else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{intro}"
        f'<Play>{escape(music_url)}</Play>'
        f'<Redirect method="POST">{escape(poll_url)}</Redirect>'
        "</Response>"
    )


def twiml_conference(name: str, *, status_url: str | None = None) -> str:
    cb = f' statusCallback="{escape(status_url)}" statusCallbackEvent="join leave end"' if status_url else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        '<Dial><Conference beep="false" startConferenceOnEnter="true" '
        f'endConferenceOnExit="false"{cb}>{escape(name)}</Conference></Dial>'
        "</Response>"
    )


def twiml_supervisor_conference(
    name: str, *, muted: bool, coach_sid: str | None, announce: str | None = None
) -> str:
    """Listen (muted), whisper (coach) or join/take over (full participant)."""
    intro = f'<Say voice="Polly.Joanna">{escape(announce)}</Say>' if announce else ""
    coach = f' coach="{escape(coach_sid)}"' if coach_sid else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{intro}"
        f'<Dial><Conference beep="false" muted="{"true" if muted else "false"}"'
        f'{coach} startConferenceOnEnter="false" endConferenceOnExit="false">'
        f"{escape(name)}</Conference></Dial>"
        "</Response>"
    )


def twiml_say_then_stream(message: str, *, stream_url: str, status_url: str, call_id: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Say voice="Polly.Joanna">{escape(message)}</Say>'
        f'<Connect action="{escape(status_url)}">'
        f'<Stream url="{escape(stream_url)}">'
        f'<Parameter name="callId" value="{escape(call_id)}" />'
        "</Stream></Connect></Response>"
    )


def twiml_dial(number: str, caller_id: str | None = None) -> str:
    attrs = f' callerId="{escape(caller_id)}"' if caller_id else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Dial{attrs}>{escape(number)}</Dial></Response>"
    )
