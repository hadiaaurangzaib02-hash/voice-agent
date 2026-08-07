"""Credential decryption, Twilio signature validation and API-key auth."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import Header, HTTPException, Request, status

from .config import get_settings

log = logging.getLogger(__name__)


def _aes_key() -> bytes:
    raw = get_settings().telephony_cred_secret
    try:
        decoded = base64.b64decode(raw, validate=True)
        if len(decoded) == 32:
            return decoded
    except Exception:  # noqa: BLE001 - not base64, fall through to hashing
        pass
    return hashlib.sha256(raw.encode("utf-8")).digest()


def decrypt_credentials(ciphertext_b64: str) -> dict[str, Any]:
    """Mirror of the dashboard's AES-256-GCM `decryptJson` (iv|tag|ciphertext)."""
    buf = base64.b64decode(ciphertext_b64)
    iv, tag, ct = buf[:12], buf[12:28], buf[28:]
    plain = AESGCM(_aes_key()).decrypt(iv, ct + tag, None)
    return json.loads(plain.decode("utf-8"))


def validate_twilio_signature(
    *, auth_token: str, url: str, params: dict[str, str], signature: str | None
) -> bool:
    """RFC-compliant Twilio request validation (no SDK dependency)."""
    if not signature:
        return False
    data = url + "".join(f"{k}{params[k]}" for k in sorted(params))
    digest = hmac.new(
        auth_token.encode("utf-8"), data.encode("utf-8"), hashlib.sha1
    ).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().backend_api_key
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )


def public_url_for(request: Request, path: str) -> str:
    base = get_settings().public_base_url.rstrip("/")
    return f"{base}{path}"


def ws_url_for(path: str) -> str:
    base = get_settings().public_base_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :] + path
    if base.startswith("http://"):
        return "ws://" + base[len("http://") :] + path
    return "wss://" + base + path


def media_stream_token(call_id: str) -> str:
    return hmac.new(
        get_settings().backend_api_key.encode("utf-8"),
        call_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_media_stream_token(call_id: str, token: str | None) -> bool:
    return bool(token) and hmac.compare_digest(media_stream_token(call_id), token)
