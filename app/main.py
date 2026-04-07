"""FastAPI application entry point."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.routers import health, stt
from app.services.stt_service import initialize_stt_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup/shutdown events."""
    # Startup: Initialize Whisper model (10-30 seconds)
    initialize_stt_service(model_size=settings.whisper_model, device=settings.device)
    yield
    # Shutdown: Cleanup if needed


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(stt.router)
    return app


app = create_app()