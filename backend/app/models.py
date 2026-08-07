"""Request/response models for the control API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

E164 = Field(pattern=r"^\+[1-9]\d{6,14}$")


class OutboundCallRequest(BaseModel):
    number_id: str
    to: str = E164
    agent_id: str | None = None
    metadata: dict[str, Any] | None = None
    lead_id: str | None = None
    scheduled_at: datetime | None = None
    max_retries: int | None = Field(default=None, ge=0, le=10)


class OutboundCallResponse(BaseModel):
    call_id: str
    external_call_id: str | None = None
    status: str


class CallActionRequest(BaseModel):
    call_id: str


class TransferRequest(BaseModel):
    call_id: str
    to: str = E164
    announce: str | None = None


class MuteRequest(BaseModel):
    call_id: str
    muted: bool = True


class RecordRequest(BaseModel):
    call_id: str
    action: Literal["start", "stop"] = "start"
    recording_sid: str | None = None


class CampaignRequest(BaseModel):
    number_id: str
    agent_id: str | None = None
    targets: list[str] = Field(min_length=1, max_length=5000)
    concurrency: int | None = Field(default=None, ge=1, le=50)
    max_retries: int | None = Field(default=None, ge=0, le=10)
    scheduled_at: datetime | None = None


class CampaignResponse(BaseModel):
    campaign_id: str
    accepted: int
    scheduled_at: datetime | None = None


class EscalationAnswerRequest(BaseModel):
    escalation_id: str
    answer: str = Field(min_length=1, max_length=4000)
    answered_by: str | None = None


class EscalationAnswerResponse(BaseModel):
    escalation_id: str
    delivery: Literal["resume_live_call", "callback_scheduled"]


class DoNotCallRequest(BaseModel):
    org_id: str
    e164: str = E164
    reason: str | None = None


class SupervisorRequest(BaseModel):
    call_id: str
    supervisor_e164: str = E164
    mode: Literal["listen", "whisper", "join", "takeover"] = "listen"
