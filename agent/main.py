import gc
import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.routes import voice
from agent.services import stt_service, llm_service, tts_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _log_memory():
    """Log current process RSS for debugging memory pressure."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    logger.info("  Memory (RSS): %s", line.strip())
                    return
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("VOICE AGENT API — STARTUP")
    logger.info("=" * 60)
    _log_memory()

    logger.info("[1/3] STT: checking faster-whisper model…")
    stt_service.load_model()
    logger.info("[1/3] STT: READY")
    _log_memory()
    gc.collect()

    logger.info("[2/3] LLM: checking Ollama + LLaMA…")
    llm_service.ensure_model()
    logger.info("[2/3] LLM: READY")
    _log_memory()

    logger.info("[3/3] TTS: loading XTTS v2 model (freeing STT memory first)…")
    stt_service.unload_model()
    gc.collect()
    _log_memory()
    try:
        tts_service.load_model()
    except Exception:
        logger.critical(
            "TTS MODEL LOADING FAILED — the API will start without TTS.\n%s",
            traceback.format_exc(),
        )
    else:
        logger.info("[3/3] TTS: READY")
    _log_memory()

    logger.info("Reloading STT model…")
    stt_service.load_model()
    logger.info("STT: READY (reloaded)")
    _log_memory()

    logger.info("=" * 60)
    logger.info("ALL MODELS LOADED — API IS LIVE")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down Voice Agent API")


app = FastAPI(
    title="Voice Agent API",
    description="Audio in → STT → LLaMA → XTTS v2 → Audio out",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(voice.router)


@app.get("/")
async def root():
    return {
        "service": "Voice Agent API",
        "version": "1.0.0",
        "docs": "/docs",
    }
