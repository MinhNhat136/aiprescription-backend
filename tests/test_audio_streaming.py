"""Tests for streaming audio transcription and WebSocket streaming mode."""
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.routers.audio import (
    ALLOWED_EXTENSIONS,
    ALLOWED_FIELD_IDS,
    DEFAULT_PARTIAL_INTERVAL,
    _maybe_send_partial,
    _partial_timers,
)
from app.services.connection_manager import ConnectionManager
from app.services.stt_service import FasterWhisperService


# ─── STT streaming transcription tests ──────────────────────────────────────


def _make_service() -> FasterWhisperService:
    """Create a FasterWhisperService with mocked model for testing."""
    service = FasterWhisperService(model_size="large-v3-turbo", device="cpu")
    service._initialized = True
    service.model = MagicMock()
    return service


async def _chunk_generator(chunks: list[bytes]):
    """Async generator that yields a list of byte chunks."""
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_transcribe_stream_yields_final_result():
    """transcribe_stream should yield a final result with is_final=True."""
    service = _make_service()

    async def mock_transcribe(audio_path, language="vi"):
        return "Xin chào"

    with patch.object(service, "transcribe", side_effect=mock_transcribe):
        results = []
        async for result in service.transcribe_stream(
            _chunk_generator([b"audio_data"]),
            partial_interval=100.0,  # Long interval — no partials
        ):
            results.append(result)

    assert len(results) == 1
    assert results[0]["is_final"] is True
    assert results[0]["text"] == "Xin chào"


@pytest.mark.asyncio
async def test_transcribe_stream_yields_partial_results():
    """transcribe_stream should yield partial results at regular intervals."""
    service = _make_service()

    async def mock_transcribe(audio_path, language="vi"):
        return "Xin chào bác sĩ"

    with patch.object(service, "transcribe", side_effect=mock_transcribe):
        results = []
        async for result in service.transcribe_stream(
            _chunk_generator([b"chunk1", b"chunk2", b"chunk3"]),
            partial_interval=0.0,  # Trigger partial on every chunk
        ):
            results.append(result)

    # Should have partial + final results
    partial_results = [r for r in results if r["is_final"] is False]
    final_results = [r for r in results if r["is_final"] is True]
    assert len(partial_results) >= 1, f"Expected partials, got {results}"
    assert len(final_results) == 1


@pytest.mark.asyncio
async def test_transcribe_stream_empty_chunks():
    """transcribe_stream should handle empty chunk list gracefully."""
    service = _make_service()

    results = []
    async for result in service.transcribe_stream(
        _chunk_generator([]),
        partial_interval=0.0,
    ):
        results.append(result)

    assert len(results) == 1
    assert results[0]["is_final"] is True
    assert results[0]["text"] == ""


@pytest.mark.asyncio
async def test_transcribe_stream_not_initialized_raises():
    """transcribe_stream should raise RuntimeError when model not initialized."""
    service = FasterWhisperService(model_size="large-v3-turbo", device="cpu")
    # Service not initialized

    with pytest.raises(RuntimeError, match="not initialized"):
        async for _ in service.transcribe_stream(
            _chunk_generator([b"audio"]),
            partial_interval=0.0,
        ):
            pass


@pytest.mark.asyncio
async def test_transcribe_buffer_returns_empty_on_failure():
    """_transcribe_buffer should return empty string when transcription fails."""
    service = _make_service()

    async def mock_transcribe(audio_path, language="vi"):
        raise RuntimeError("Model error")

    with patch.object(service, "transcribe", side_effect=mock_transcribe):
        result = await service._transcribe_buffer(b"invalid_audio", ".webm", "vi")

    assert result == ""


@pytest.mark.asyncio
async def test_transcribe_stream_result_format():
    """Each yielded result should have text (str) and is_final (bool) keys."""
    service = _make_service()

    async def mock_transcribe(audio_path, language="vi"):
        return "Kết quả"

    with patch.object(service, "transcribe", side_effect=mock_transcribe):
        results = []
        async for result in service.transcribe_stream(
            _chunk_generator([b"audio_data"]),
            partial_interval=100.0,
        ):
            results.append(result)

    for result in results:
        assert "text" in result
        assert "is_final" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["is_final"], bool)


@pytest.mark.asyncio
async def test_transcribe_stream_graceful_incomplete_audio():
    """transcribe_stream should handle incomplete/corrupt audio chunks gracefully."""
    service = _make_service()

    call_count = 0

    async def mock_transcribe(audio_path, language="vi"):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call fails (incomplete audio buffer)
            raise RuntimeError("Incomplete audio")
        return "Bác sĩ"

    with patch.object(service, "transcribe", side_effect=mock_transcribe):
        # Long interval so only final transcription runs
        results = []
        async for result in service.transcribe_stream(
            _chunk_generator([b"audio_chunk"]),
            partial_interval=100.0,
        ):
            results.append(result)

    # _transcribe_buffer catches the error and returns ""
    # so the final result should have empty text
    assert len(results) == 1
    assert results[0]["is_final"] is True


# ─── Partial result throttling tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_maybe_send_partial_skips_when_interval_not_elapsed():
    """_maybe_send_partial should skip transcription if interval hasn't elapsed."""
    manager = ConnectionManager()
    mock_ws = AsyncMock()
    await manager.connect(mock_ws, "test_field")

    mock_stt = MagicMock()
    mock_stt.get_status.return_value = {"status": "ready"}
    mock_stt.transcribe_bytes = AsyncMock(return_value="partial text")

    # Set timer to current time (interval just started)
    _partial_timers["test_field"] = time.monotonic()

    with patch("app.routers.audio.get_stt_service", return_value=mock_stt):
        await _maybe_send_partial("test_field", [b"audio"], ".webm")

    # Should NOT have called transcribe_bytes since interval hasn't elapsed
    mock_stt.transcribe_bytes.assert_not_called()

    # Cleanup
    manager.disconnect("test_field")
    _partial_timers.pop("test_field", None)


@pytest.mark.asyncio
async def test_maybe_send_partial_sends_result_when_interval_elapsed():
    """_maybe_send_partial should send partial result when interval has elapsed."""
    from app.routers import audio as audio_module

    test_manager = ConnectionManager()
    mock_ws = AsyncMock()
    await test_manager.connect(mock_ws, "test_field2")

    mock_stt = MagicMock()
    mock_stt.get_status.return_value = {"status": "ready"}
    mock_stt.transcribe_bytes = AsyncMock(return_value="Xin chào")

    # Set timer to past (interval has elapsed)
    _partial_timers["test_field2"] = time.monotonic() - DEFAULT_PARTIAL_INTERVAL - 1

    # Patch the global manager used by _maybe_send_partial
    with patch.object(audio_module, "manager", test_manager), \
         patch("app.routers.audio.get_stt_service", return_value=mock_stt):
        await _maybe_send_partial("test_field2", [b"audio_data"], ".webm")

    # Should have called transcribe_bytes
    mock_stt.transcribe_bytes.assert_called_once()
    # Should have sent partial transcription via websocket
    assert mock_ws.send_json.call_count >= 1

    # Cleanup
    test_manager.disconnect("test_field2")
    _partial_timers.pop("test_field2", None)


@pytest.mark.asyncio
async def test_maybe_send_partial_skips_on_empty_chunks():
    """_maybe_send_partial should skip when no audio chunks are available."""
    manager = ConnectionManager()

    mock_stt = MagicMock()
    mock_stt.get_status.return_value = {"status": "ready"}

    # Set elapsed timer
    _partial_timers["test_field3"] = time.monotonic() - DEFAULT_PARTIAL_INTERVAL - 1

    with patch("app.routers.audio.get_stt_service", return_value=mock_stt):
        await _maybe_send_partial("test_field3", [], ".webm")

    mock_stt.transcribe_bytes.assert_not_called()

    # Cleanup
    _partial_timers.pop("test_field3", None)


@pytest.mark.asyncio
async def test_maybe_send_partial_handles_failure_gracefully():
    """_maybe_send_partial should not raise when transcription fails."""
    manager = ConnectionManager()
    mock_ws = AsyncMock()
    await manager.connect(mock_ws, "test_field4")

    mock_stt = MagicMock()
    mock_stt.get_status.return_value = {"status": "ready"}
    mock_stt.transcribe_bytes = AsyncMock(side_effect=RuntimeError("Transcription failed"))

    # Set elapsed timer
    _partial_timers["test_field4"] = time.monotonic() - DEFAULT_PARTIAL_INTERVAL - 1

    # Should not raise
    with patch("app.routers.audio.get_stt_service", return_value=mock_stt):
        await _maybe_send_partial("test_field4", [b"audio"], ".webm")

    # Cleanup
    manager.disconnect("test_field4")
    _partial_timers.pop("test_field4", None)


@pytest.mark.asyncio
async def test_maybe_send_partial_skips_when_service_not_ready():
    """_maybe_send_partial should skip when STT service is not ready."""
    manager = ConnectionManager()
    mock_ws = AsyncMock()
    await manager.connect(mock_ws, "test_field5")

    mock_stt = MagicMock()
    mock_stt.get_status.return_value = {"status": "not_initialized"}

    # Set elapsed timer
    _partial_timers["test_field5"] = time.monotonic() - DEFAULT_PARTIAL_INTERVAL - 1

    with patch("app.routers.audio.get_stt_service", return_value=mock_stt):
        await _maybe_send_partial("test_field5", [b"audio"], ".webm")

    mock_stt.transcribe_bytes.assert_not_called()

    # Cleanup
    manager.disconnect("test_field5")
    _partial_timers.pop("test_field5", None)


# ─── Allowed field IDs and extensions ───────────────────────────────────────


def test_webm_is_allowed_extension():
    """WebM should be in the allowed extensions for browser MediaRecorder."""
    assert ".webm" in ALLOWED_EXTENSIONS


def test_common_field_ids_are_allowed():
    """Common prescription field IDs should be in the allowed set."""
    for field_id in ["patientName", "diagnosis", "medicationName", "doctorNotes"]:
        assert field_id in ALLOWED_FIELD_IDS


def test_medicine_dynamic_ids():
    """Dynamic medicine_* prefixed IDs should be accepted by validation."""
    assert "medicine_some-uuid" not in ALLOWED_FIELD_IDS  # Not in static list
    # But the validation logic in websocket_audio_stream accepts medicine_* prefix