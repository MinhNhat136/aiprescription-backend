"""STT router for audio transcription endpoint."""
import tempfile
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.services.stt_service import get_stt_service

router = APIRouter(prefix="/stt", tags=["Speech-to-Text"])


@router.post(
    "/transcribe",
    summary="Transcribe audio file to text",
    description="Accepts an audio file upload and returns the transcribed text using faster-whisper.",
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe (wav, mp3, webm, m4a)"),
) -> JSONResponse:
    """Transcribe an uploaded audio file to text.

    Args:
        file: Audio file upload.

    Returns:
        JSONResponse with transcribed text.

    Raises:
        HTTPException: If transcription fails or service not initialized.
    """
    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".webm", ".m4a", ".ogg", ".flac"}
    file_ext = Path(file.filename or "").suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp_path = Path(tmp.name)
        content = await file.read()
        tmp.write(content)

    try:
        # Get initialized STT service
        stt_service = get_stt_service()
        status_info = stt_service.get_status()

        if status_info["status"] != "ready":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"STT service not ready. Status: {status_info['status']}",
            )

        # Transcribe audio
        transcribed_text = await stt_service.transcribe(str(tmp_path))

        return JSONResponse(
            content={
                "text": transcribed_text,
                "language": "vi",
                "model": status_info["model"],
                "device": status_info["device"],
            }
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}",
        )
    finally:
        # Cleanup temp file
        if tmp_path.exists():
            tmp_path.unlink()


@router.get(
    "/status",
    summary="Get STT service status",
    description="Returns the current status of the Whisper STT service.",
)
async def get_stt_status() -> JSONResponse:
    """Get the current STT service status.

    Returns:
        JSONResponse with service status information.
    """
    try:
        stt_service = get_stt_service()
        return JSONResponse(content=stt_service.get_status())
    except RuntimeError as e:
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
