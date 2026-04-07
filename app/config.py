"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "VoiceFillPrescription"
    version: str = "0.1.0"
    whisper_model: str = "large-v3-turbo"
    device: str = "cuda"


settings = Settings()