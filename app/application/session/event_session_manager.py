from datetime import datetime, timedelta

from app.domain.session.event_session import EventSession
from app.domain.session.session_status import SessionStatus


class EventSessionManager:
    """
    Keeps all active sessions in memory.

    One manager per HouseRuntime.
    """

    SESSION_TIMEOUT_SECONDS = 5

    def __init__(self):

        self.active_sessions: dict[str, EventSession] = {}

    def get(self, session_id: str):

        return self.active_sessions.get(session_id)

    def create(
        self,
        session: EventSession,
    ):

        session.activate()

        self.active_sessions[
            session.session_id
        ] = session

        return session

    def remove(
        self,
        session_id: str,
    ):

        self.active_sessions.pop(
            session_id,
            None,
        )

    def touch(
        self,
        session_id: str,
    ):

        session = self.get(session_id)

        if session:

            session.update_last_seen()

    def expired_sessions(self):

        now = datetime.utcnow()

        expired = []

        for session in self.active_sessions.values():

            if (
                now - session.last_seen_at
            ) > timedelta(
                seconds=self.SESSION_TIMEOUT_SECONDS
            ):

                expired.append(session)

        return expired

    def cleanup(self):

        for session in self.expired_sessions():

            session.expire()

            self.remove(
                session.session_id
            )