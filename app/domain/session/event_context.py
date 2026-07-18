from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from app.domain.perception.motion import MotionResult
from app.domain.perception.perception_result import PerceptionResult
from app.domain.perception.track_result import TrackResult
from app.domain.runtime.camera_runtime import CameraRuntime
from app.domain.runtime.setting_runtime import RuntimeHouseSettings
from app.domain.session.event_session import EventSession


@dataclass(slots=True)
class EventContext:
    """Single mutable object threaded through every pipeline stage for
    one frame. Each processor reads what it needs and writes its result
    back, instead of every process() signature growing another arg."""

    frame: object

    house_id: UUID
    device_id: UUID
    camera_runtime: CameraRuntime
    house_settings: RuntimeHouseSettings
    security_mode: str = "NORMAL"

    motion_result: Optional[MotionResult] = None
    perception_result: Optional[PerceptionResult] = None
    track_result: Optional[TrackResult] = None

    active_sessions: list[EventSession] = field(default_factory=list)
    finished_sessions: list[EventSession] = field(default_factory=list)

    @property
    def should_run_detection(self) -> bool:
        """Motion gates detection. If motion detection is disabled for
        this camera, motion_result stays None and detection always runs."""
        if self.motion_result is None:
            return True
        return self.motion_result.has_motion

    @property
    def camera_settings(self):
        return self.camera_runtime.settings
    
    @property
    def has_finished_sessions(self):
        return len(self.finished_sessions)>0