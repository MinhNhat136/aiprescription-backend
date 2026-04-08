"""Whisper STT service with GPU detection and configurable model storage."""
import asyncio
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
import logging

logger = logging.getLogger(__name__)


class STTServiceInterface(ABC):
    """Abstract interface for Speech-to-Text services."""

    @abstractmethod
    async def transcribe(self, audio_path: str, language: str = "vi") -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to the audio file.
            language: Language code for transcription (default: Vietnamese).

        Returns:
            str: Transcribed text.
        """
        pass

    @abstractmethod
    def get_status(self) -> dict:
        """Get the current status of the STT service.

        Returns:
            dict: Status information including model and device.
        """
        pass


class FasterWhisperService(STTServiceInterface):
    """Faster-Whisper implementation of STT service with GPU acceleration."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        model_path: Optional[str] = None,
        hf_home: Optional[str] = None,
    ):
        """Initialize the FasterWhisperService.

        Args:
            model_size: Whisper model size to use.
            device: Device to run inference on ("cuda" or "cpu").
            model_path: Optional local directory for model files. If None, uses HuggingFace cache.
            hf_home: Optional HuggingFace cache directory. Overrides default ~/.cache/huggingface/.
        """
        self.model_size = model_size
        self.device = device
        self.model_path = model_path
        self.hf_home = hf_home
        self.model: Optional[WhisperModel] = None
        self._initialized = False
        self._model_cache_dir: Optional[str] = None

    def initialize(self) -> bool:
        """Initialize the Whisper model.

        Returns:
            bool: True if initialization successful, False otherwise.
        """
        try:
            # Try GPU first, fall back to CPU
            actual_device = self._detect_device()

            # Build kwargs for WhisperModel
            kwargs: dict = {
                "model_size_or_path": self.model_size,
                "device": actual_device,
                "compute_type": "float16" if actual_device == "cuda" else "int8",
            }

            # If model_path provided, use local files (for pre-downloaded models)
            if self.model_path:
                model_dir = Path(self.model_path).expanduser().resolve()
                if model_dir.exists() and any(model_dir.iterdir()):
                    logger.info(
                        f"Loading Whisper model from local path: {model_dir} (device: {actual_device})"
                    )
                    kwargs["model_size_or_path"] = str(model_dir)
                    self._model_cache_dir = str(model_dir)
                else:
                    logger.warning(
                        f"Model path '{model_dir}' not found or empty. "
                        f"faster-whisper will download to default HuggingFace cache."
                    )
                    self._model_cache_dir = str(model_dir)

            # Set HuggingFace cache directory if specified
            if self.hf_home:
                hf_cache = Path(self.hf_home).expanduser().resolve()
                logger.info(f"Setting HuggingFace cache to: {hf_cache}")
                os.environ["HF_HOME"] = str(hf_cache)
                os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
                os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(600)  # 10 min timeout
                if not self._model_cache_dir:
                    self._model_cache_dir = str(hf_cache)

            logger.info(
                f"Initializing Whisper model '{self.model_size}' on {actual_device}"
            )

            self.model = WhisperModel(**kwargs)
            self._initialized = True
            logger.info(f"Whisper model initialized successfully on {actual_device}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            self._initialized = False
            return False

    def _detect_device(self) -> str:
        """Detect available device, preferring GPU.

        Returns:
            str: Device to use ("cuda" or "cpu").
        """
        try:
            import ctranslate2

            gpu_count = ctranslate2.get_cuda_device_count()
            if gpu_count > 0:
                logger.info(f"CUDA available: {gpu_count} GPU(s) detected")
                return "cuda"
        except Exception as e:
            logger.warning(f"CUDA detection failed: {e}")
        logger.info("Falling back to CPU device")
        return "cpu"

    async def transcribe(self, audio_path: str, language: str = "vi") -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to the audio file.
            language: Language code for transcription (default: Vietnamese).

        Returns:
            str: Transcribed text.

        Raises:
            RuntimeError: If model not initialized.
        """
        if not self._initialized or self.model is None:
            raise RuntimeError("Whisper model not initialized")

        def _transcribe_sync():
            segments, _ = self.model.transcribe(
                audio_path, language=language, vad_filter=True, word_timestamps=True
            )
            return "".join([segment.text for segment in segments])

        return await asyncio.to_thread(_transcribe_sync)

    def get_status(self) -> dict:
        """Get the current status of the STT service.

        Returns:
            dict: Status information.
        """
        return {
            "model": self.model_size,
            "device": self.device if self._initialized else "not_initialized",
            "status": "ready" if self._initialized else "not_initialized",
            "model_path": self._model_cache_dir or "default (HuggingFace cache)",
        }


# Global service instance
_stt_service: Optional[FasterWhisperService] = None


def get_stt_service() -> FasterWhisperService:
    """Get the global STT service instance.

    Returns:
        FasterWhisperService: The initialized STT service.

    Raises:
        RuntimeError: If STT service not initialized.
    """
    global _stt_service
    if _stt_service is None:
        raise RuntimeError(
            "STT service not initialized. Call initialize_stt_service() first."
        )
    return _stt_service


def initialize_stt_service(
    model_size: str = "large-v3-turbo",
    device: str = "cuda",
    model_path: Optional[str] = None,
    hf_home: Optional[str] = None,
) -> FasterWhisperService:
    """Initialize the global STT service instance.

    Args:
        model_size: Whisper model size to use.
        device: Device to run inference on.
        model_path: Optional local directory for model files.
        hf_home: Optional HuggingFace cache directory.

    Returns:
        FasterWhisperService: The initialized STT service.
    """
    global _stt_service
    _stt_service = FasterWhisperService(
        model_size=model_size,
        device=device,
        model_path=model_path,
        hf_home=hf_home,
    )
    _stt_service.initialize()
    return _stt_service