"""WebSocket router for real-time audio streaming with partial transcription."""
import asyncio
import json
import logging
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.connection_manager import manager
from app.services.stt_service import get_stt_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Audio Streaming"])

# Allowed prescription field IDs that the frontend can stream audio for.
# Both snake_case (backend convention) and camelCase (frontend convention)
# are accepted to handle naming convention differences gracefully.
# Additionally, dynamically generated medicine_* prefixed IDs (containing UUIDs)
# are allowed for per-row medication voice input.
ALLOWED_FIELD_IDS: Set[str] = {
    # snake_case (backend convention)
    "patient_name",
    "phone_number",
    "diagnosis",
    "medication_name",
    "dosage_unit",
    "times_per_day",
    "unit_per_dose",
    "doctor_notes",
    "follow_up_date",
    # camelCase (frontend convention — maps to snake_case counterparts above)
    "patientName",
    "phone",
    "address",
    "medicationName",
    "dosageUnit",
    "timesPerDay",
    "unitPerDose",
    "doctorNotes",
    "followUpDate",
    # additional frontend field IDs
    "notes",
}

# Allowed audio file extensions for buffered transcription
ALLOWED_EXTENSIONS: Set[str] = {".wav", ".mp3", ".webm", ".m4a", ".ogg", ".flac"}

# Default interval (seconds) between partial transcription results in streaming mode
DEFAULT_PARTIAL_INTERVAL = 2.0

# Minimum buffer size (bytes) before attempting partial transcription.
# MediaRecorder WebM chunks need at least a few KB to form a parseable stream;
# smaller buffers cause FFmpeg "Invalid data" errors inside faster-whisper.
MIN_PARTIAL_BUFFER_SIZE = 4096


@router.websocket("/ws/audio/{field_id}")
async def websocket_audio_stream(websocket: WebSocket, field_id: str):
    """WebSocket endpoint for real-time audio streaming.

    The frontend connects to this endpoint when the doctor presses the
    push-to-talk button for a specific prescription field. Audio chunks
    are sent as binary frames. The client signals recording completion
    by sending a JSON message with type "recording_stop".

    Streaming mode provides partial transcription results as audio arrives,
    giving real-time feedback. The client can request streaming mode by
    sending {"type": "streaming", "extension": ".webm"} as the first
    message after connection. If no streaming mode is requested, the
    endpoint falls back to buffered mode (collect all audio, then transcribe).

    Protocol:
        - Client connects: ws://host/ws/audio/{field_id}
        - Optional: Client sends {"type": "streaming", "extension": ".webm"}
          to enable partial results
        - Client sends binary frames: raw audio chunks
        - Client sends JSON {"type": "recording_stop", "extension": ".webm"}
          to signal end of recording and trigger final transcription
        - Server responds with JSON:
          - {"type": "transcription", "field_id": "...", "text": "...", "is_final": false}
            for partial results (streaming mode only)
          - {"type": "transcription", "field_id": "...", "text": "...", "is_final": true}
            for the final result
          - {"type": "error", "field_id": "...", "message": "..."}

    Args:
        websocket: The WebSocket connection.
        field_id: The prescription field being recorded (e.g. "patient_name").
    """
    # Validate field_id — accept both static field IDs and dynamic medicine_* prefixes
    is_dynamic = field_id.startswith("medicine_")
    is_allowed = field_id in ALLOWED_FIELD_IDS or is_dynamic

    if not is_allowed:
        await websocket.accept()
        await websocket.send_json({
            "type": "error",
            "field_id": field_id,
            "message": f"Invalid field_id: '{field_id}'. Allowed: {', '.join(sorted(ALLOWED_FIELD_IDS))} or medicine_*",
        })
        await websocket.close(code=4004, reason="Invalid field_id")
        return

    # Accept and register connection
    await manager.connect(websocket, field_id)
    logger.info(f"connection open for field: {field_id}")

    audio_chunks: list[bytes] = []
    file_extension = ".webm"  # Default for MediaRecorder API
    streaming_mode = False

    try:
        while True:
            data = await websocket.receive()

            # Handle WebSocket disconnect message from raw ASGI receive()
            # When using receive() instead of receive_bytes()/receive_text(),
            # client disconnects arrive as {"type": "websocket.disconnect"} rather
            # than raising WebSocketDisconnect. If we don't break here, calling
            # receive() again raises RuntimeError.
            if data.get("type") == "websocket.disconnect":
                logger.info(f"WebSocket disconnect received for field: {field_id}")
                break

            if "bytes" in data and data["bytes"] is not None:
                # Audio chunk received
                audio_chunks.append(data["bytes"])

                # In streaming mode, check if we should emit a partial result
                if streaming_mode:
                    await _maybe_send_partial(
                        field_id, audio_chunks, file_extension
                    )

            elif "text" in data and data["text"] is not None:
                # JSON control message
                try:
                    message = json.loads(data["text"])
                except json.JSONDecodeError:
                    await manager.send_error(field_id, "Invalid JSON message")
                    continue

                msg_type = message.get("type")

                if msg_type == "streaming":
                    # Enable streaming mode
                    streaming_mode = True
                    ext = message.get("extension", ".webm")
                    if ext in ALLOWED_EXTENSIONS:
                        file_extension = ext
                    logger.info(
                        f"Streaming mode enabled for field '{field_id}' (ext={file_extension})"
                    )
                    await websocket.send_json({
                        "type": "streaming_started",
                        "field_id": field_id,
                    })

                elif msg_type == "recording_stop":
                    # Override extension if provided
                    ext = message.get("extension", ".webm")
                    if ext in ALLOWED_EXTENSIONS:
                        file_extension = ext

                    if not audio_chunks:
                        await manager.send_error(field_id, "No audio data received")
                        continue

                    # Concatenate all audio chunks into a single buffer
                    audio_buffer = b"".join(audio_chunks)
                    audio_chunks.clear()

                    # Transcribe the complete audio
                    try:
                        stt_service = get_stt_service()
                        status_info = stt_service.get_status()

                        if status_info["status"] != "ready":
                            await manager.send_error(
                                field_id,
                                f"STT service not ready: {status_info['status']}",
                            )
                            continue

                        transcribed_text = await stt_service.transcribe_bytes(
                            audio_buffer, file_extension=file_extension
                        )

                        await manager.send_transcription(
                            field_id, transcribed_text, is_final=True
                        )
                        logger.info(
                            f"Transcription complete for field '{field_id}': "
                            f"{transcribed_text[:50]}..."
                        )

                        # Signal session completion and close the connection.
                        # Without this, the handler stays in the while loop
                        # indefinitely, leaving zombie WebSocket connections.
                        # The frontend relies on this signal (or the subsequent
                        # WebSocket close) to clean up its recording state.
                        await websocket.send_json({
                            "type": "recording_complete",
                            "field_id": field_id,
                        })
                        break  # Exit the receive loop → finally block closes connection

                    except RuntimeError as e:
                        await manager.send_error(
                            field_id, f"STT service error: {str(e)}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Transcription failed for field '{field_id}': {e}"
                        )
                        await manager.send_error(
                            field_id, "Transcription failed"
                        )

                elif msg_type == "ping":
                    # Keepalive ping — respond with pong
                    try:
                        await websocket.send_json({"type": "pong"})
                    except Exception:
                        pass

                else:
                    await manager.send_error(
                        field_id, f"Unknown message type: '{msg_type}'"
                    )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for field: {field_id}")
    except Exception as e:
        logger.error(f"WebSocket error for field '{field_id}': {e}")
    finally:
        manager.disconnect(field_id, websocket=websocket)


# Tracks the last partial transcription time per field to throttle updates
_partial_timers: dict[str, float] = {}


async def _maybe_send_partial(
    field_id: str,
    audio_chunks: list[bytes],
    file_extension: str,
) -> None:
    """Send a partial transcription result if enough time has elapsed.

    Throttles partial results to avoid excessive transcription calls.
    Only transcribes if at least DEFAULT_PARTIAL_INTERVAL has passed since
    the last partial result for this field.

    Args:
        field_id: The prescription field identifier.
        audio_chunks: Accumulated audio chunks.
        file_extension: Audio file extension.
    """
    import time

    now = time.monotonic()
    last_time = _partial_timers.get(field_id, 0.0)

    if now - last_time < DEFAULT_PARTIAL_INTERVAL:
        return  # Not enough time has passed

    if not audio_chunks:
        return  # No audio to transcribe

    # Avoid transcribing tiny buffers that cannot form a valid media stream.
    # Early MediaRecorder chunks (especially WebM) lack enough data for
    # FFmpeg/faster-whisper to parse, causing "Invalid data" errors.
    audio_buffer = b"".join(audio_chunks)
    if len(audio_buffer) < MIN_PARTIAL_BUFFER_SIZE:
        return

    # Update timer before transcription to prevent concurrent calls
    _partial_timers[field_id] = now

    try:
        stt_service = get_stt_service()
        status_info = stt_service.get_status()

        if status_info["status"] != "ready":
            return  # Silently skip partial if service not ready

        transcribed_text = await stt_service.transcribe_bytes(
            audio_buffer, file_extension=file_extension
        )

        if transcribed_text:
            await manager.send_transcription(
                field_id, transcribed_text, is_final=False
            )
            logger.debug(
                f"Partial result for field '{field_id}': {transcribed_text[:50]}..."
            )
    except Exception as e:
        # Partial failures are expected — the audio buffer may still be
        # incomplete or the WebM stream may lack finalization segments.
        # Log at debug to avoid noise; final transcription will retry.
        logger.debug(f"Partial transcription skipped for field '{field_id}': {e}")