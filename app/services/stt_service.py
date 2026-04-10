"""Whisper STT service with GPU detection and configurable model storage."""
import asyncio
import io
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
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
    async def transcribe_bytes(
        self, audio_data: bytes, file_extension: str = ".webm", language: str = "vi"
    ) -> str:
        """Transcribe raw audio bytes to text.

        Writes audio bytes to a temporary file, transcribes, and cleans up.

        Args:
            audio_data: Raw audio bytes.
            file_extension: File extension for the temporary file (default: .webm).
            language: Language code for transcription (default: Vietnamese).

        Returns:
            str: Transcribed text.
        """
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        file_extension: str = ".webm",
        language: str = "vi",
        partial_interval: float = 2.0,
    ) -> AsyncGenerator[dict, None]:
        """Stream transcribe audio chunks with partial results.

        Receives audio chunks via async generator and yields partial
        transcription results at regular intervals, followed by a final
        result when the stream ends.

        Args:
            audio_chunks: Async generator yielding audio bytes.
            file_extension: File extension for audio format (default: .webm).
            language: Language code for transcription (default: Vietnamese).
            partial_interval: Seconds between partial transcriptions (default: 2.0).

        Yields:
            dict: Transcription result with keys:
                - text: str - Transcribed text
                - is_final: bool - True for the last result
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

        Verifies both GPU driver presence (via ctranslate2) and CUDA runtime
        library availability (cuBLAS). This prevents selecting "cuda" when
        the driver is present but CUDA toolkit libraries are missing — a
        common issue on Windows where the GPU driver is installed but the
        CUDA toolkit is not on PATH.

        Returns:
            str: Device to use ("cuda" or "cpu").
        """
        try:
            import ctranslate2

            gpu_count = ctranslate2.get_cuda_device_count()
            if gpu_count > 0:
                # GPU driver found, but verify cuBLAS is loadable.
                # CTranslate2 needs cuBLAS at inference time; detecting the
                # GPU alone is insufficient.
                try:
                    import ctypes
                    import platform

                    if platform.system() == "Windows":
                        ctypes.CDLL("cublas64_12.dll")
                    else:
                        ctypes.CDLL("libcublas.so.12")
                    logger.info(
                        f"CUDA available: {gpu_count} GPU(s) detected, cuBLAS verified"
                    )
                    return "cuda"
                except OSError as cublas_err:
                    logger.warning(
                        f"CUDA GPU detected but cuBLAS not loadable: {cublas_err}. "
                        f"Install CUDA toolkit or add its libraries to PATH. "
                        f"Falling back to CPU."
                    )
                    self.device = "cpu"
                    return "cpu"
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

    async def transcribe_bytes(
        self, audio_data: bytes, file_extension: str = ".webm", language: str = "vi"
    ) -> str:
        """Transcribe raw audio bytes to text.

        Writes audio bytes to a temporary file, transcribes it, and cleans up.

        Args:
            audio_data: Raw audio bytes.
            file_extension: File extension for the temporary file (default: .webm).
            language: Language code for transcription (default: Vietnamese).

        Returns:
            str: Transcribed text.

        Raises:
            RuntimeError: If model not initialized.
        """
        if not self._initialized or self.model is None:
            raise RuntimeError("Whisper model not initialized")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_extension
            ) as tmp:
                tmp.write(audio_data)
                tmp_path = Path(tmp.name)

            # Reuse the file-based transcription
            return await self.transcribe(str(tmp_path), language=language)
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        file_extension: str = ".webm",
        language: str = "vi",
        partial_interval: float = 2.0,
    ) -> AsyncGenerator[dict, None]:
        """Stream transcribe audio chunks with partial results.

        Accumulates audio chunks and periodically transcribes the growing
        buffer to produce partial results. Yields a final result when
        the audio stream ends.

        Args:
            audio_chunks: Async generator yielding audio bytes.
            file_extension: File extension for audio format (default: .webm).
            language: Language code for transcription (default: Vietnamese).
            partial_interval: Seconds between partial transcriptions (default: 2.0).

        Yields:
            dict: Transcription result with keys:
                - text: str - Transcribed text
                - is_final: bool - True for the last result
        """
        if not self._initialized or self.model is None:
            raise RuntimeError("Whisper model not initialized")

        buffer = bytearray()
        last_partial_time = asyncio.get_event_loop().time()

        async for chunk in audio_chunks:
            buffer.extend(chunk)
            elapsed = asyncio.get_event_loop().time() - last_partial_time

            if elapsed >= partial_interval and len(buffer) > 0:
                partial_text = await self._transcribe_buffer(bytes(buffer), file_extension, language)
                last_partial_time = asyncio.get_event_loop().time()
                if partial_text:
                    yield {"text": partial_text, "is_final": False}

        # Final transcription of the complete buffer
        if len(buffer) > 0:
            final_text = await self._transcribe_buffer(bytes(buffer), file_extension, language)
            yield {"text": final_text, "is_final": True}
        else:
            yield {"text": "", "is_final": True}

    async def _transcribe_buffer(
        self, audio_data: bytes, file_extension: str = ".webm", language: str = "vi"
    ) -> str:
        """Transcribe an audio buffer by writing to a temp file.

        Args:
            audio_data: Complete audio bytes.
            file_extension: File extension for the temp file.
            language: Language code for transcription.

        Returns:
            str: Transcribed text, or empty string on error.
        """
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_extension
            ) as tmp:
                tmp.write(audio_data)
                tmp_path = Path(tmp.name)

            return await self.transcribe(str(tmp_path), language=language)
        except Exception as e:
            logger.warning(f"Partial transcription failed (possibly incomplete audio): {e}")
            return ""
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

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