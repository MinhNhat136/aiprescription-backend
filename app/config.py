"""Application configuration using Pydantic Settings."""
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env")

    # Application
    app_name: str = "VoiceFillPrescription"
    version: str = "0.1.0"

    # Whisper STT
    whisper_model: str = "large-v3-turbo"
    # Local directory to store Whisper models (default: ~/.cache/huggingface/hub/)
    whisper_model_path: str = ""
    # HuggingFace cache directory for model files
    hf_home: str = ""
    hf_hub_download_timeout: int = 600  # Timeout for model download in seconds (10 min)
    device: str = "cuda"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/voicefillprescription"
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600


settings = Settings()
