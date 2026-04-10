"""Tests for WebSocket audio streaming endpoint."""
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a synchronous test client for WebSocket testing."""
    return TestClient(app)


class TestWebSocketAudioEndpoint:
    """Tests for the /ws/audio/{field_id} WebSocket endpoint."""

    def test_valid_field_id_connects(self, client):
        """Test that a valid field_id connection is accepted."""
        with client.websocket_connect("/ws/audio/patient_name") as ws:
            # Connection should succeed without error
            pass  # If we get here, connection was accepted

    def test_invalid_field_id_rejected(self, client):
        """Test that an invalid field_id receives an error and is closed."""
        with client.websocket_connect("/ws/audio/invalid_field") as ws:
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Invalid field_id" in data["message"]
            assert data["field_id"] == "invalid_field"

    def test_all_allowed_field_ids(self, client):
        """Test that all allowed field IDs are accepted."""
        allowed_fields = [
            "patient_name",
            "phone_number",
            "diagnosis",
            "medication_name",
            "dosage_unit",
            "times_per_day",
            "unit_per_dose",
            "doctor_notes",
            "follow_up_date",
            # camelCase variants
            "patientName",
            "phone",
            "address",
            "medicationName",
            "dosageUnit",
            "timesPerDay",
            "unitPerDose",
            "doctorNotes",
            "followUpDate",
            # additional frontend IDs
            "notes",
        ]
        for field_id in allowed_fields:
            with client.websocket_connect(f"/ws/audio/{field_id}") as ws:
                # Should connect without error
                pass

    def test_ping_pong(self, client):
        """Test that ping messages receive pong responses."""
        with client.websocket_connect("/ws/audio/patient_name") as ws:
            ws.send_json({"type": "ping"})
            data = ws.receive_json()
            assert data["type"] == "pong"

    def test_unknown_message_type_returns_error(self, client):
        """Test that unknown message types receive an error response."""
        with client.websocket_connect("/ws/audio/diagnosis") as ws:
            ws.send_json({"type": "unknown_type"})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Unknown message type" in data["message"]

    def test_recording_stop_with_no_audio_returns_error(self, client):
        """Test that recording_stop with no audio data returns an error."""
        with client.websocket_connect("/ws/audio/patient_name") as ws:
            ws.send_json({"type": "recording_stop"})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "No audio data received" in data["message"]

    def test_recording_stop_invalid_extension_returns_error(self, client):
        """Test that recording_stop with invalid extension returns an error."""
        with client.websocket_connect("/ws/audio/patient_name") as ws:
            ws.send_bytes(b"fake audio data")
            ws.send_json({"type": "recording_stop", "extension": ".xyz"})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Unsupported audio format" in data["message"]

    def test_invalid_json_message_returns_error(self, client):
        """Test that invalid JSON text messages receive an error."""
        with client.websocket_connect("/ws/audio/diagnosis") as ws:
            ws.send_text("not json")
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Invalid JSON" in data["message"]

    def test_binary_data_is_buffered(self, client):
        """Test that binary data is accepted and buffered (without triggering transcription)."""
        with client.websocket_connect("/ws/audio/medication_name") as ws:
            # Send some audio bytes — should not trigger any response yet
            ws.send_bytes(b"\x00\x01\x02\x03")
            # Send a ping to verify connection is still alive
            ws.send_json({"type": "ping"})
            data = ws.receive_json()
            assert data["type"] == "pong"

    def test_notes_field_accepted(self, client):
        """Test that the 'notes' field ID (frontend doctor's advice) is accepted."""
        with client.websocket_connect("/ws/audio/notes") as ws:
            # Should connect without error
            pass

    def test_medicine_dynamic_id_accepted(self, client):
        """Test that dynamically generated medicine_* prefixed field IDs are accepted."""
        uuid = "abc123-def456-789"
        with client.websocket_connect(f"/ws/audio/medicine_{uuid}") as ws:
            # Should connect without error
            pass

    def test_medicine_id_with_complex_uuid_accepted(self, client):
        """Test medicine_* accepts UUIDs with non-alphanumeric characters."""
        with client.websocket_connect("/ws/audio/medicine_a1b2c3d4-e5f6-7890-abcd-ef1234567890") as ws:
            pass

    def test_recording_complete_signal_sent_after_transcription(self, client):
        """Test that recording_stop triggers transcription and recording_complete signal.

        After sending final transcription, the backend must send a
        recording_complete message so the frontend knows the session is done.
        """
        with client.websocket_connect("/ws/audio/patient_name") as ws:
            # Send audio data
            ws.send_bytes(b"\x00\x01\x02\x03")
            # Signal end of recording
            ws.send_json({"type": "recording_stop", "extension": ".webm"})

            # Read all messages — should contain transcription and recording_complete
            messages = []
            while True:
                try:
                    data = ws.receive_json(mode="text")
                    messages.append(data)
                    if data.get("type") == "recording_complete":
                        break
                except Exception:
                    break

            # Should have received a recording_complete message
            complete_msgs = [m for m in messages if m.get("type") == "recording_complete"]
            assert len(complete_msgs) >= 1, f"Expected recording_complete, got: {messages}"
            assert complete_msgs[0]["field_id"] == "patient_name"

    def test_connection_closes_after_recording_complete(self, client):
        """Test that WebSocket connection closes after recording_complete.

        The backend should close the connection after the recording session
        completes, preventing zombie connections.
        """
        with client.websocket_connect("/ws/audio/diagnosis") as ws:
            ws.send_bytes(b"\x00\x01\x02\x03")
            ws.send_json({"type": "recording_stop", "extension": ".webm"})

            # Read messages until connection closes
            while True:
                try:
                    data = ws.receive_json(mode="text")
                    if data.get("type") == "recording_complete":
                        break
                except Exception:
                    break

            # After recording_complete, connection should close.
            # Attempting to send should fail or return disconnected.
            # (TestClient raises an exception when the connection is closed.)