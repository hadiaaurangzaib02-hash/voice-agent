"""FastAPI application entrypoint for the real telephony voice backend."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import db
from .config import get_settings
from .routers import calls as calls_router
from .routers import escalations as escalations_router
from .routers import media as media_router
from .routers import webhooks as webhooks_router
from .security import require_api_key, ws_url_for

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("voice-backend")


async def _escalation_retry_loop() -> None:
    """Retries unanswered callbacks (every 2 h by default) until resolved."""
    import asyncio

    from .services import escalation as esc

    while True:
        try:
            for row in await esc.due_retries():
                await esc.schedule_callback(row)
        except Exception:  # noqa: BLE001
            log.exception("escalation retry sweep failed")
        await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    await db.init_pool()
    sweeper = asyncio.create_task(_escalation_retry_loop())
    log.info("voice backend ready at %s", settings.public_base_url)
    yield
    sweeper.cancel()
    await db.close_pool()


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(webhooks_router.router)
app.include_router(media_router.router)
app.include_router(calls_router.router)
app.include_router(escalations_router.router)


@app.get("/health")
async def health(response: Response) -> dict[str, object]:
    ok = True
    try:
        await db.fetchval("select 1")
    except Exception:  # noqa: BLE001
        ok = False
        response.status_code = 503
    return {
        "status": "ok" if ok else "degraded",
        "database": ok,
        "stt": settings.stt_provider,
        "llm": settings.llm_provider,
        "tts": settings.tts_provider,
    }


@app.get("/webhooks/urls", dependencies=[Depends(require_api_key)])
async def webhook_urls() -> dict[str, str]:
    """Exact URLs to paste into the Twilio console for each number."""
    base = settings.public_base_url.rstrip("/")
    return {
        "voice_incoming": f"{base}/webhooks/twilio/voice",
        "call_status": f"{base}/webhooks/twilio/status",
        "outbound_answer": f"{base}/webhooks/twilio/answer",
        "hold_music": f"{base}/webhooks/twilio/hold",
        "supervisor_conference": f"{base}/webhooks/twilio/supervisor",
        "media_stream": ws_url_for("/ws/media-stream"),
    }


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception) -> JSONResponse:  # pragma: no cover
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
