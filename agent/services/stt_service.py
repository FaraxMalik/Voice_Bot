import io
import logging

from faster_whisper import WhisperModel

from agent.config import get_settings

logger = logging.getLogger(__name__)

_model: WhisperModel | None = None


def _ensure_model() -> WhisperModel:
    """
    faster-whisper auto-downloads the model on first use if it is not
    already cached. This call handles both check and download.
    """
    settings = get_settings()
    logger.info(
        "Loading STT model: faster-whisper '%s' (device=%s, compute=%s)",
        settings.stt_model_size,
        settings.stt_device,
        settings.stt_compute_type,
    )
    logger.info("If not cached, the model will be downloaded automatically…")
    model = WhisperModel(
        settings.stt_model_size,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
    )
    logger.info("STT model ready")
    return model


def load_model() -> None:
    global _model
    _model = _ensure_model()


def unload_model() -> None:
    global _model
    _model = None
    logger.info("STT model unloaded to free memory")


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe raw audio bytes to text."""
    if _model is None:
        raise RuntimeError("STT model not loaded — call load_model() first")

    audio_stream = io.BytesIO(audio_bytes)
    segments, info = _model.transcribe(audio_stream, beam_size=5)
    transcript = " ".join(segment.text.strip() for segment in segments)

    logger.info(
        "Transcribed %.1fs of audio [%s] → %d chars",
        info.duration,
        info.language,
        len(transcript),
    )
    return transcript


def is_ready() -> bool:
    return _model is not None
