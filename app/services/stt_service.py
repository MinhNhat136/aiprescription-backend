"""Whisper STT service with GPU detection."""
from abc import ABC, abstractmethod
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

    def __init__(self, model_size: str = "large-v3-turbo", device: str = "cuda"):
        """Initialize the FasterWhisperService.

        Args:
            model_size: Whisper model size to use.
            device: Device to run inference on ("cuda" or "cpu").
        """
        self.model_size = model_size
        self.device = device
        self.model: Optional[WhisperModel] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the Whisper model.

        Returns:
            bool: True if initialization successful, False otherwise.
        """
        try:
            # Try GPU first, fall back to CPU
            actual_device = self._detect_device()
            logger.info(
                f"Initializing Whisper model '{self.model_size}' on {actual_device}"
            )

            self.model = WhisperModel(
                self.model_size,
                device=actual_device,
                compute_type="float16" if actual_device == "cuda" else "int8",
            )
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

        segments, _ = self.model.transcribe(
            audio_path, language=language, vad_filter=True, word_timestamps=True
        )
        return "".join([segment.text for segment in segments])

    def get_status(self) -> dict:
        """Get the current status of the STT service.

        Returns:
            dict: Status information.
        """
        return {
            "model": self.model_size,
            "device": self.device if self._initialized else "not_initialized",
            "status": "ready" if self._initialized else "not_initialized",
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
    model_size: str = "large-v3-turbo", device: str = "cuda"
) -> FasterWhisperService:
    """Initialize the global STT service instance.

    Args:
        model_size: Whisper model size to use.
        device: Device to run inference on.

    Returns:
        FasterWhisperService: The initialized STT service.
    """
    global _stt_service
    _stt_service = FasterWhisperService(model_size=model_size, device=device)
    _stt_service.initialize()
    return _stt_service