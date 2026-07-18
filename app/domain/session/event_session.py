from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from app.domain.perception.track import Track
from app.domain.session.session_status import SessionStatus


@dataclass(slots=True)
class EventSession:
    """One real-world event: one tracked person/object on one camera,
    from first sighting until the track ends and enrichment completes."""

    id: UUID = field(default_factory=uuid4)

    house_id: Optional[UUID] = None
    device_id: Optional[UUID] = None

    # Stamped once at creation by EventSessionManager — never change for
    # the lifetime of the session, so TimelineProcessor/EnrichmentProcessor
    # never have to look them up elsewhere.
    camera_name: Optional[str] = None
    camera_location: Optional[str] = None

    track: Optional[Track] = None

    # Django timeline event id (returned by /events/ingest/)
    timeline_event_id: Optional[UUID] = None

    status: SessionStatus = SessionStatus.CREATED

    started_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None

    timeline_created: bool = False
    timeline_updated: bool = False

    alert_generated: bool = False

    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None

    summary: Optional[str] = None

    enrichment_started: bool = False
    enrichment_completed: bool = False

    def activate(self):
        self.status = SessionStatus.ACTIVE
        self.updated_at = datetime.utcnow()

    def update_track(self, track: Track):
        self.track = track
        self.updated_at = datetime.utcnow()

    def wait_for_enrichment(self):
        self.status = SessionStatus.WAITING_FOR_ENRICHMENT
        self.ended_at = datetime.utcnow()

    def start_enrichment(self):
        self.status = SessionStatus.ENRICHING
        self.enrichment_started = True
        self.updated_at = datetime.utcnow()

    def complete_enrichment(self):
        self.status = SessionStatus.COMPLETED
        self.enrichment_completed = True
        self.updated_at = datetime.utcnow()

    def expire(self):
        self.status = SessionStatus.EXPIRED
        self.updated_at = datetime.utcnow()

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()

    @property
    def is_active(self):
        return self.status == SessionStatus.ACTIVE

    @property
    def is_completed(self):
        return self.status == SessionStatus.COMPLETED

    @property
    def recognized(self):
        return self.track is not None and self.track.is_recognized

    @property
    def unknown(self):
        return self.track is not None and self.track.is_unknown_person

    @property
    def has_timeline(self) -> bool:
        return self.timeline_created and self.timeline_event_id is not None


    @property
    def has_clip(self) -> bool:
        return self.clip_path is not None


    @property
    def has_summary(self) -> bool:
        return self.summary is not None
