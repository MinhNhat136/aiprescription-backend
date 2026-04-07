"""Health check endpoints for the backend service."""
from fastapi import APIRouter

from app.services.stt_service import get_stt_service

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        dict: Health status and version information.
    """
    return {"status": "healthy", "version": "0.1.0"}


@router.get("/stt/status")
async def stt_status() -> dict:
    """Get STT service status.

    Returns:
        dict: STT service status including model and device info.
    """
    service = get_stt_service()
    return service.get_status()