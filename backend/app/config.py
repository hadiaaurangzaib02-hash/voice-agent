"""Runtime configuration for the voice telephony backend."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Service ---
    app_name: str = "VocaScribe Voice Backend"
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    # Public https base URL of THIS backend (used to build Twilio webhook + wss URLs)
    public_base_url: str
    allowed_origins: str = "*"
    # Shared secret required on /api/* control endpoints (sent as X-API-Key)
    backend_api_key: str

    # --- Database (same Postgres the SaaS dashboard reads) ---
    database_url: str
    db_pool_min: int = 1
    db_pool_max: int = 10

    # --- Credential encryption (must match the dashboard's TELEPHONY_CRED_SECRET) ---
    telephony_cred_secret: str

    # --- Twilio (fallback when a provider row has no stored credentials) ---
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_validate_signature: bool = True

    # --- Speech to text ---
    stt_provider: Literal["deepgram", "openai"] = "deepgram"
    deepgram_api_key: str | None = None
    deepgram_model: str = "nova-2-phonecall"
    deepgram_language: str = "en-US"
    openai_api_key: str | None = None

    # --- LLM ---
    llm_provider: Literal["groq", "openai"] = "groq"
    groq_api_key: str | None = None
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.6
    llm_max_tokens: int = 220

    # --- Embeddings for RAG retrieval over kb_chunks ---
    embedding_provider: Literal["openai"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    rag_top_k: int = 5
    rag_min_similarity: float = 0.25

    # --- Text to speech ---
    tts_provider: Literal["elevenlabs", "openai"] = "elevenlabs"
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model: str = "eleven_turbo_v2_5"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"

    # --- Escalation / supervisor handoff ---
    # Gmail SMTP (use an app password, not the account password)
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    escalation_email_to: str | None = None
    # Public URL of the dashboard (used in escalation emails)
    dashboard_url: str = "http://localhost:8080"
    # Hold music played while a supervisor is being consulted
    hold_music_url: str = "http://com.twilio.sounds.music.s3.amazonaws.com/MARKOVICHAMP-Borghestral.mp3"
    hold_max_seconds: int = 300
    callback_retry_hours: int = 2
    callback_max_attempts: int = 6

    # --- Call behaviour ---
    max_call_seconds: int = 1800
    silence_hangup_seconds: int = 20
    barge_in_enabled: bool = True
    outbound_retry_max: int = 3
    outbound_retry_delay_seconds: int = 300
    campaign_concurrency: int = 5
    automatic_recording: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
