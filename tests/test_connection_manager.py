"""Tests for WebSocket ConnectionManager."""
import pytest

from app.services.connection_manager import ConnectionManager


class MockWebSocket:
    """Mock WebSocket for testing ConnectionManager."""

    def __init__(self):
        self.sent_messages: list = []
        self.accepted = False
        self.closed = False
        self.close_code = None
        self.close_reason = None

    async def accept(self):
        self.accepted = True

    async def send_json(self, data: dict):
        self.sent_messages.append(data)

    async def close(self, code: int = 1000, reason: str = ""):
        self.closed = True
        self.close_code = code
        self.close_reason = reason


@pytest.fixture
def manager():
    """Create a fresh ConnectionManager for each test."""
    return ConnectionManager()


@pytest.mark.asyncio
async def test_connect_accepts_websocket(manager):
    """Test that connect() calls accept() on the websocket."""
    ws = MockWebSocket()
    await manager.connect(ws, "patient_name")
    assert ws.accepted is True


@pytest.mark.asyncio
async def test_connect_registers_connection(manager):
    """Test that connect() registers the connection by field_id."""
    ws = MockWebSocket()
    await manager.connect(ws, "patient_name")
    assert manager.is_connected("patient_name") is True
    assert manager.connection_count == 1


@pytest.mark.asyncio
async def test_connect_replaces_existing_connection(manager):
    """Test that connecting with the same field_id replaces the old connection."""
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    await manager.connect(ws1, "patient_name")
    await manager.connect(ws2, "patient_name")
    # ws1 should be closed, ws2 should be active
    assert ws1.closed is True
    assert manager.is_connected("patient_name") is True
    assert manager.connection_count == 1


@pytest.mark.asyncio
async def test_disconnect_removes_connection(manager):
    """Test that disconnect() removes the connection."""
    ws = MockWebSocket()
    await manager.connect(ws, "diagnosis")
    assert manager.is_connected("diagnosis") is True

    manager.disconnect("diagnosis")
    assert manager.is_connected("diagnosis") is False
    assert manager.connection_count == 0


@pytest.mark.asyncio
async def test_disconnect_nonexistent_field_is_safe(manager):
    """Test that disconnecting a non-existent field_id does not raise."""
    manager.disconnect("nonexistent")  # Should not raise


@pytest.mark.asyncio
async def test_send_transcription(manager):
    """Test that send_transcription() sends correct JSON to the websocket."""
    ws = MockWebSocket()
    await manager.connect(ws, "medication_name")

    await manager.send_transcription("medication_name", "Paracetamol", is_final=True)

    assert len(ws.sent_messages) == 1
    msg = ws.sent_messages[0]
    assert msg["type"] == "transcription"
    assert msg["field_id"] == "medication_name"
    assert msg["text"] == "Paracetamol"
    assert msg["is_final"] is True


@pytest.mark.asyncio
async def test_send_transcription_partial(manager):
    """Test that send_transcription() with is_final=False works."""
    ws = MockWebSocket()
    await manager.connect(ws, "diagnosis")

    await manager.send_transcription("diagnosis", "Cảm lạnh", is_final=False)

    msg = ws.sent_messages[0]
    assert msg["is_final"] is False


@pytest.mark.asyncio
async def test_send_transcription_no_connection_raises(manager):
    """Test that send_transcription() raises RuntimeError when no connection exists."""
    with pytest.raises(RuntimeError, match="No active connection"):
        await manager.send_transcription("nonexistent", "text")


@pytest.mark.asyncio
async def test_send_error(manager):
    """Test that send_error() sends error JSON to the websocket."""
    ws = MockWebSocket()
    await manager.connect(ws, "patient_name")

    await manager.send_error("patient_name", "Transcription failed")

    msg = ws.sent_messages[0]
    assert msg["type"] == "error"
    assert msg["field_id"] == "patient_name"
    assert msg["message"] == "Transcription failed"


@pytest.mark.asyncio
async def test_send_error_no_connection_does_not_raise(manager):
    """Test that send_error() does not raise when no connection exists."""
    await manager.send_error("nonexistent", "error")  # Should not raise


@pytest.mark.asyncio
async def test_multiple_connections(manager):
    """Test that multiple field_ids can be connected simultaneously."""
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    await manager.connect(ws1, "patient_name")
    await manager.connect(ws2, "diagnosis")

    assert manager.connection_count == 2
    assert manager.is_connected("patient_name") is True
    assert manager.is_connected("diagnosis") is True

    # Send to specific field
    await manager.send_transcription("patient_name", "Nguyen Van A")
    assert len(ws1.sent_messages) == 1
    assert len(ws2.sent_messages) == 0


@pytest.mark.asyncio
async def test_disconnect_with_matching_websocket(manager):
    """Test that disconnect() removes connection when websocket matches."""
    ws = MockWebSocket()
    await manager.connect(ws, "patient_name")

    manager.disconnect("patient_name", websocket=ws)
    assert manager.is_connected("patient_name") is False
    assert manager.connection_count == 0


@pytest.mark.asyncio
async def test_disconnect_skips_when_websocket_mismatch(manager):
    """Test that disconnect() with mismatched websocket does NOT remove the entry.

    This prevents a stale handler's finally block from removing a replacement
    connection that was registered for the same field_id.
    """
    ws_old = MockWebSocket()
    ws_new = MockWebSocket()

    # Old connection
    await manager.connect(ws_old, "phone")
    assert manager.is_connected("phone") is True

    # New connection replaces old one
    await manager.connect(ws_new, "phone")
    assert manager.is_connected("phone") is True

    # Old handler tries to disconnect with the old websocket reference
    # This should NOT remove the new connection
    manager.disconnect("phone", websocket=ws_old)
    assert manager.is_connected("phone") is True
    assert manager.connection_count == 1


@pytest.mark.asyncio
async def test_disconnect_without_websocket_removes_any(manager):
    """Test that disconnect() without websocket param always removes the entry."""
    ws = MockWebSocket()
    await manager.connect(ws, "diagnosis")

    # Without websocket param, disconnect removes regardless of identity
    manager.disconnect("diagnosis")
    assert manager.is_connected("diagnosis") is False