"""Routers package."""
from app.routers.health import router as health_router
from app.routers.audio import router as audio_router

__all__ = ["health_router", "audio_router"]