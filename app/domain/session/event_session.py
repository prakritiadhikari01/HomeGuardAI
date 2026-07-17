from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from app.domain.session.session_status import SessionStatus


@dataclass(slots=True)
class EventSession:
    """
    Represents one real-world incident.

    One person/vehicle/animal -> one EventSession.

    Timeline:
        Motion detected
            ↓
        Person detected
            ↓
        Face recognized
            ↓
        Clip generated
            ↓
        Summary generated
            ↓
        Session completed
    """

    session_id: str

    track_id: int

    home_id: UUID

    device_id: UUID

    started_at: datetime

    last_seen_at: datetime

    status: SessionStatus = SessionStatus.CREATED

    event_id: UUID | None = None

    person_type: str = "UNKNOWN"

    person_label: str | None = None

    member_id: UUID | None = None

    face_profile_id: UUID | None = None

    confidence_score: float = 0.0

    best_frame: object | None = None

    best_snapshot: object | None = None

    thumbnail: object | None = None

    clip_path: str | None = None

    summary: str | None = None

    timeline_created: bool = False

    enrichment_sent: bool = False

    metadata: dict = field(default_factory=dict)

    def activate(self) -> None:
        self.status = SessionStatus.ACTIVE

    def update_last_seen(self) -> None:
        self.last_seen_at = datetime.utcnow()

    def attach_event(self, event_id: UUID) -> None:
        self.event_id = event_id
        self.timeline_created = True

    def attach_identity(
        self,
        person_type: str,
        person_label: str | None,
        member_id: UUID | None,
        face_profile_id: UUID | None,
        confidence: float,
    ) -> None:

        self.person_type = person_type
        self.person_label = person_label
        self.member_id = member_id
        self.face_profile_id = face_profile_id
        self.confidence_score = confidence

    def waiting_for_enrichment(self):
        self.status = SessionStatus.WAITING_FOR_ENRICHMENT

    def complete(self):
        self.status = SessionStatus.COMPLETED

    def expire(self):
        self.status = SessionStatus.EXPIRED