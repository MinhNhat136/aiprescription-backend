"""WebSocket connection manager for audio streaming."""
import logging
from typing import Dict

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections for audio streaming.

    Each connection is keyed by field_id, representing the prescription
    field the doctor is currently filling via voice input. Only one
    active connection per field_id is allowed — a new connection
    replaces the old one.
    """

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, field_id: str) -> None:
        """Accept a WebSocket connection and register it by field_id.

        If a connection already exists for the given field_id, it is
        replaced (the old one is disconnected).

        Args:
            websocket: The incoming WebSocket connection.
            field_id: The prescription field identifier.
        """
        await websocket.accept()

        # Replace existing connection for the same field
        if field_id in self.active_connections:
            old_ws = self.active_connections[field_id]
            try:
                await old_ws.close(code=1012, reason="Replaced by new connection")
            except Exception:
                pass  # Already disconnected

        self.active_connections[field_id] = websocket
        logger.info(f"WebSocket connected for field: {field_id}")

    def disconnect(self, field_id: str, websocket: WebSocket = None) -> None:
        """Remove a WebSocket connection from the manager.

        If websocket is provided, only removes the entry if it still
        points to the same WebSocket object. This prevents a stale
        handler's finally block from removing a replacement connection
        that was registered for the same field_id.

        Args:
            field_id: The prescription field identifier to disconnect.
            websocket: Optional WebSocket instance to match against.
                If provided, disconnect only proceeds if the stored
                connection is this exact object.
        """
        stored = self.active_connections.get(field_id)
        if stored is None:
            return
        if websocket is not None and stored is not websocket:
            # A newer connection has replaced this one — do not remove it
            logger.debug(
                f"Skipping disconnect for field '{field_id}': "
                f"stored connection differs from caller"
            )
            return
        del self.active_connections[field_id]
        logger.info(f"WebSocket disconnected for field: {field_id}")

    async def send_transcription(
        self, field_id: str, text: str, is_final: bool = True
    ) -> None:
        """Send a transcription result to the WebSocket for a field.

        Args:
            field_id: The prescription field to send to.
            text: The transcribed text.
            is_final: Whether this is the final transcription (True)
                or an intermediate/partial result (False).

        Raises:
            RuntimeError: If no active connection exists for the field.
        """
        websocket = self.active_connections.get(field_id)
        if websocket is None:
            raise RuntimeError(f"No active connection for field: {field_id}")

        await websocket.send_json({
            "type": "transcription",
            "field_id": field_id,
            "text": text,
            "is_final": is_final,
        })

    async def send_error(self, field_id: str, message: str) -> None:
        """Send an error message to the WebSocket for a field.

        Args:
            field_id: The prescription field to send to.
            message: The error message.
        """
        websocket = self.active_connections.get(field_id)
        if websocket is None:
            logger.warning(f"No active connection for field: {field_id}, cannot send error")
            return

        await websocket.send_json({
            "type": "error",
            "field_id": field_id,
            "message": message,
        })

    def is_connected(self, field_id: str) -> bool:
        """Check if a connection exists for a field.

        Args:
            field_id: The prescription field to check.

        Returns:
            bool: True if an active connection exists.
        """
        return field_id in self.active_connections

    @property
    def connection_count(self) -> int:
        """Return the number of active connections."""
        return len(self.active_connections)


# Global connection manager instance
manager = ConnectionManager()