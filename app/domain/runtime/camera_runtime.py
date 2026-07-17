from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from app.domain.runtime.setting_runtime import RuntimeDeviceSettings


@dataclass(slots=True)
class CameraRuntime:
    """
    Runtime representation of one camera.

    This object exists only while the AI Engine is running.
    """

    camera_id: UUID

    house_id: UUID

    name: str

    location: str

    stream_url: str

    settings: RuntimeDeviceSettings

    connected: bool = False

    processing: bool = False

    last_frame_at: datetime | None = None

    last_sync_at: datetime | None = None

    current_fps: float = 0.0

    frames_processed: int = 0

    reconnect_attempts: int = 0

    started_at: datetime = field(default_factory=datetime.utcnow)

    def mark_connected(self) -> None:
        self.connected = True

    def mark_disconnected(self) -> None:
        self.connected = False

    def mark_processing_started(self) -> None:
        self.processing = True

    def mark_processing_stopped(self) -> None:
        self.processing = False

    def frame_processed(self) -> None:
        self.frames_processed += 1
        self.last_frame_at = datetime.utcnow()