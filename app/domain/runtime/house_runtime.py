from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from app.domain.runtime.camera_runtime import CameraRuntime
from app.domain.runtime.setting_runtime import RuntimeHouseSettings


@dataclass(slots=True)
class HouseRuntime:
    """
    Runtime state of one house.

    Every active house has one HouseRuntime.

    The AI Engine never stores this permanently.
    """

    house_id: UUID

    security_mode: str

    settings: RuntimeHouseSettings

    cameras: dict[UUID, CameraRuntime] = field(default_factory=dict)

    active_sessions: dict[str, object] = field(default_factory=dict)

    started_at: datetime = field(default_factory=datetime.utcnow)

    last_sync_at: datetime | None = None

    running: bool = False

    def add_camera(self, camera: CameraRuntime) -> None:
        self.cameras[camera.camera_id] = camera

    def remove_camera(self, camera_id: UUID) -> None:
        self.cameras.pop(camera_id, None)

    def get_camera(self, camera_id: UUID) -> CameraRuntime | None:
        return self.cameras.get(camera_id)

    def add_session(self, session_id: str, session: object) -> None:
        self.active_sessions[session_id] = session

    def remove_session(self, session_id: str) -> None:
        self.active_sessions.pop(session_id, None)

    def session_count(self) -> int:
        return len(self.active_sessions)

    def camera_count(self) -> int:
        return len(self.cameras)

    def mark_running(self) -> None:
        self.running = True

    def mark_stopped(self) -> None:
        self.running = False

    def mark_synced(self) -> None:
        self.last_sync_at = datetime.utcnow()

    def apply_settings(
        self,
        settings: RuntimeHouseSettings,
    ) -> None:
        """
        Apply updated house settings from Django.
        """
        self.settings = settings
        self.mark_synced()

    def apply_security_mode(
        self,
        security_mode: str,
    ) -> None:
        """
        Update the current security mode.
        """
        self.security_mode = security_mode
        self.mark_synced()