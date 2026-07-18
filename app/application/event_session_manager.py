from __future__ import annotations

from typing import Dict, Tuple
from uuid import UUID

from app.domain.perception.track_result import TrackResult
from app.domain.session.event_session import EventSession

# NOTE: this is the ONLY EventSessionManager in the codebase now. The old
# duplicate at app/application/session/event_session_manager.py referenced
# session.session_id (a field EventSession never had) and has been deleted
# — it was never reachable from the real pipeline anyway.


class EventSessionManager:
    """Maintains all active sessions for one camera worker. Exactly one
    session per ByteTrack track_id."""

    def __init__(self):
        self._sessions: Dict[int, EventSession] = {}

    def update(
        self,
        *,
        house_id: UUID,
        device_id: UUID,
        camera_name: str,
        camera_location: str,
        track_result: TrackResult,
    ) -> Tuple[list[EventSession], list[EventSession]]:
        active_sessions = []
        finished_sessions = []

        for track in track_result.active_tracks:
            session = self._sessions.get(track.track_id)

            if session is None:
                session = EventSession(
                    house_id=house_id,
                    device_id=device_id,
                    camera_name=camera_name,
                    camera_location=camera_location,
                    track=track,
                )
                track.session_id = session.id
                session.activate()
                self._sessions[track.track_id] = session
            else:
                session.update_track(track)

            active_sessions.append(session)

        for track in track_result.ended_tracks:
            session = self._sessions.pop(track.track_id, None)
            if session is None:
                continue
            session.update_track(track)
            session.wait_for_enrichment()
            finished_sessions.append(session)

        return active_sessions, finished_sessions

    def get(self, track_id: int):
        return self._sessions.get(track_id)

    def all(self):
        return list(self._sessions.values())

    def remove(self, track_id: int):
        self._sessions.pop(track_id, None)

    def clear(self):
        self._sessions.clear()

    def active_sessions(self):
        return self._sessions.values()
    
    @property
    def active_count(self):
        return len(self._sessions)
