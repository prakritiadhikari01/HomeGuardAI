from __future__ import annotations

from uuid import UUID

from app.domain.perception.track_result import TrackResult
from app.domain.session.event_session import EventSession
from app.application.event_session_manager import EventSessionManager


class SessionProcessor:
    """Stage 4 — session lifecycle. Takes a TrackResult in, EventSessions
    out. house_id/device_id/camera_name/camera_location are supplied
    once at construction — they never change for the lifetime of a
    camera worker — and get stamped onto every EventSession this
    processor creates, so TimelineProcessor/AlertProcessor never guess
    them later.

    FIX: previously called EventSessionManager.update(tracks=...) with
    kwargs the domain manager's update() didn't accept at all (and read
    from an application-layer duplicate manager that referenced a
    session.session_id field EventSession never had). Now calls the one
    real EventSessionManager with its actual signature."""

    def __init__(
        self,
        house_id: UUID,
        device_id: UUID,
        camera_name: str,
        camera_location: str,
    ):
        self._house_id = house_id
        self._device_id = device_id
        self._camera_name = camera_name
        self._camera_location = camera_location
        self._session_manager = EventSessionManager()

    def process(self, track_result: TrackResult) -> tuple[list[EventSession], list[EventSession]]:
        return self._session_manager.update(
            house_id=self._house_id,
            device_id=self._device_id,
            camera_name=self._camera_name,
            camera_location=self._camera_location,
            track_result=track_result,
        )

    @property
    def active_session_count(self) -> int:
        return self._session_manager.active_count
