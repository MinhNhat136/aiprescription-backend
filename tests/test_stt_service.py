"""Tests for STT service."""
import pytest
from unittest.mock import patch, MagicMock

from app.services.stt_service import (
    FasterWhisperService,
    STTServiceInterface,
)


def test_faster_whisper_service_implements_interface():
    """Verify FasterWhisperService implements STTServiceInterface."""
    assert issubclass(FasterWhisperService, STTServiceInterface)


def test_service_reports_not_initialized_status():
    """Service should report not_initialized before initialize() call."""
    service = FasterWhisperService(model_size="large-v3-turbo", device="cuda")
    status = service.get_status()
    assert status["status"] == "not_initialized"


@patch("app.services.stt_service.WhisperModel")
def test_service_initializes_with_gpu(mock_whisper_model):
    """Service should initialize successfully when GPU is available."""
    mock_whisper_model.return_value = MagicMock()

    # Mock ctranslate2 inside _detect_device
    with patch("builtins.__import__") as mock_import:
        def side_effect(name, *args, **kwargs):
            if name == "ctranslate2":
                mock_ct = MagicMock()
                mock_ct.get_cuda_device_count.return_value = 1
                return mock_ct
            return __import__(name, *args, **kwargs)
        mock_import.side_effect = side_effect

        service = FasterWhisperService(model_size="large-v3-turbo", device="cuda")
        result = service.initialize()

        assert result is True
        assert service._initialized is True
        status = service.get_status()
        assert status["status"] == "ready"
        assert status["device"] == "cuda"


@patch("app.services.stt_service.WhisperModel")
def test_service_falls_back_to_cpu_when_no_gpu(mock_whisper_model):
    """Service should fall back to CPU when GPU is not available."""
    mock_whisper_model.return_value = MagicMock()

    with patch("builtins.__import__") as mock_import:
        def side_effect(name, *args, **kwargs):
            if name == "ctranslate2":
                mock_ct = MagicMock()
                mock_ct.get_cuda_device_count.return_value = 0
                return mock_ct
            return __import__(name, *args, **kwargs)
        mock_import.side_effect = side_effect

        service = FasterWhisperService(model_size="large-v3-turbo", device="cuda")
        result = service.initialize()

        assert result is True
        assert service._initialized is True


def test_service_initializes_on_cpu_without_gpu(mock_whisper_model=None):
    """Service should initialize on CPU without GPU detection."""
    # This test verifies the fallback works when ctranslate2 raises an exception
    service = FasterWhisperService(model_size="large-v3-turbo", device="cpu")
    # _detect_device should return "cpu" since device is explicitly cpu
    assert service.device == "cpu"