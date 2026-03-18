import logging
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from agent.services import stt_service, llm_service, tts_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/process")
async def process_voice(
    audio: UploadFile = File(...),
    session_id: str = Form("default"),
):
    """Full pipeline: audio in → STT → LLM → TTS → audio out."""
    audio_bytes = await audio.read()

    transcript = stt_service.transcribe(audio_bytes)
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe audio — no speech detected")

    reply_text = llm_service.chat(transcript, session_id=session_id)

    reply_audio = tts_service.synthesize(reply_text)

    return Response(
        content=reply_audio,
        media_type="audio/wav",
        headers={
            "X-Transcript": transcript,
            "X-Reply-Text": reply_text,
            "Content-Disposition": 'attachment; filename="response.wav"',
        },
    )


@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """STT only: audio in → transcript text out."""
    audio_bytes = await audio.read()
    transcript = stt_service.transcribe(audio_bytes)
    return {"transcript": transcript}


@router.post("/chat")
async def chat(text: str = Form(...), session_id: str = Form("default")):
    """LLM only: text in → reply text out."""
    reply = llm_service.chat(text, session_id=session_id)
    return {"reply": reply}


@router.post("/speak")
async def speak(text: str = Form(...)):
    """TTS only: text in → audio out."""
    audio_bytes = tts_service.synthesize(text)
    return Response(
        content=audio_bytes, 
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'}
    )


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session."""
    llm_service.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.get("/health")
async def health():
    """Check readiness of all three models."""
    status = {
        "stt": stt_service.is_ready(),
        "llm": llm_service.is_ready(),
        "tts": tts_service.is_ready(),
    }
    all_ready = all(status.values())
    return JSONResponse(
        content={"healthy": all_ready, "services": status},
        status_code=200 if all_ready else 503,
    )
