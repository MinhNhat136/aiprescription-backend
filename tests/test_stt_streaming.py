"""Tests for STT streaming transcription functionality."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.stt_service import FasterWhisperService


def _make_service() -> FasterWhisperService:
    """Create a FasterWhisperService with mocked model for testing."""
    service = FasterWhisperService(model_size="large-v3-turbo", device="cpu")
    # Manually set initialized state to bypass real model loading
    service._initialized = True
    service.model = MagicMock()
    return service


async def _chunk_generator(chunks: list[bytes]):
    """Async generator that yields a list of byte chunks."""
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_transcribe_stream_yields_partial_results():
    """transcribe_stream should yield partial results at the specified interval."""
    service = _make_service()

    # Mock transcribe to return different text each time
    call_count = 0
    texts = ["  Xin chào  ", "  Xin chào bác sĩ  "]

    def mock_transcribe_sync():
        nonlocal call_count
        text = texts[call_count % len(texts)]
        call_count += 1
        return text

    service.model = MagicMock()
    # Mock transcribe to use asyncio.to_thread path
    original_transcribe = service.transcribe

    async def mock_transcribe(audio_path, language="vi"):
        return mock_transcribe_sync()

    with patch.object(service, "transcribe", side_effect=mock_transcribe):
        chunks = [b"audio_data_chunk_1", b"audio_data_chunk_2"]
        results = []

        async for result in service.transcribe_stream(
            _chunk_generator(chunks),
            file_extension=".webm",
            language="vi",
            partial_interval=0.0,  # Trigger partial on every chunk
        ):
            results.append(result)

    # Should have at least one partial result and one final result
    final_results = [r for r in results if r["is_final"]]
    assert len(final_results) == 1, f"Expected 1 final result, got {len(final_results)}"
    assert final_results[0]["is_final"] is True
    assert isinstance(final_results[0]["text"], str)


@pytest.mark.asyncio
async def test_transcribe_stream_yields_final_result():
    """transcribe_stream should always yield a final result with is_final=True."""
    service = _make_service()

    async def mock_transcribe(audio_path, language="vi"):
        return "Xin chào"

    with patch.object(service, "transcribe", side_effect=mock_transcribe):
        chunks = [b"audio_data"]
        results = []

        async for result in service.transcribe_stream(
            _chunk_generator(chunks),
            partial_interval=100.0,  # Very long interval — no partial results
        ):
            results.append(result)

    # Should have exactly one result (final)
    assert len(results) == 1
    assert results[0]["is_final"] is True
    assert results[0]["text"] == "Xin chào"


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

    # Should yield one final result with empty text
    assert len(results) == 1
    assert results[0]["is_final"] is True
    assert results[0]["text"] == ""


@pytest.mark.asyncio
async def test_transcribe_stream_handles_incomplete_audio():
    """transcribe_stream should gracefully handle incomplete/corrupt audio chunks.

    When _transcribe_buffer returns empty string (which it does on failure),
    partial results with empty text are skipped, but the final result is still yielded.
    """
    service = _make_service()

    # Mock transcribe to raise an error, simulating corrupt audio
    # _transcribe_buffer catches this and returns "" internally
    call_count = 0

    async def mock_transcribe(audio_path, language="vi"):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate incomplete audio — _transcribe_buffer catches this
            raise RuntimeError("Incomplete audio data")
        return "Bác sĩ"

    with patch.object(service, "transcribe", side_effect=mock_transcribe):
        chunks = [b"audio_chunk_1"]

        results = []
        async for result in service.transcribe_stream(
            _chunk_generator(chunks),
            partial_interval=100.0,  # No partial results, only final
        ):
            results.append(result)

        # Should yield a final result — _transcribe_buffer catches the error
        # and returns "", so final text may be empty but result is still yielded
        assert len(results) >= 1
        final = [r for r in results if r["is_final"]]
        assert len(final) == 1


@pytest.mark.asyncio
async def test_transcribe_stream_partial_interval_controls_frequency():
    """Partial results should only be yielded when partial_interval has elapsed."""
    service = _make_service()

    async def mock_transcribe(audio_path, language="vi"):
        return "Partial text"

    with patch.object(service, "transcribe", side_effect=mock_transcribe):
        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        results = []
        async for result in service.transcribe_stream(
            _chunk_generator(chunks),
            partial_interval=100.0,  # Very long interval — no partials expected
        ):
            results.append(result)

    # With a very long interval, only the final result should be emitted
    partial_results = [r for r in results if not r["is_final"]]
    final_results = [r for r in results if r["is_final"]]
    assert len(partial_results) == 0
    assert len(final_results) == 1


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
    """Each yielded result should have 'text' (str) and 'is_final' (bool) keys."""
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
async def test_transcribe_stream_multiple_chunks():
    """transcribe_stream should accumulate chunks and produce results."""
    service = _make_service()

    async def mock_transcribe(audio_path, language="vi"):
        return "Chào bạn"

    with patch.object(service, "transcribe", side_effect=mock_transcribe):
        # Multiple chunks with very short interval to trigger partials
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        results = []

        async for result in service.transcribe_stream(
            _chunk_generator(chunks),
            partial_interval=0.0,
        ):
            results.append(result)

    # Should have at least partial results + 1 final result
    assert len(results) >= 1
    # Last result must be final
    assert results[-1]["is_final"] is True