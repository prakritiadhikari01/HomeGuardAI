from datetime import datetime, timezone

from app.domain.perception.detection import PersonStatus
from app.domain.session.event_session import EventSession
from app.infrastructure.api.django_client import DjangoClient

HEARTBEAT_SECONDS = 5  # how often an already-created session re-syncs to Django


class TimelineProcessor:
    """Stage 6 — talks to Django for ACTIVE sessions only. POST
    /events/ingest/ is the fast path; Django's own dedup window
    creates-or-merges. Finished-session enrichment (clip/summary/PATCH)
    is EnrichmentProcessor's job — a session only needs ingesting while
    still being seen.

    A session is only ingested once its track's person_status resolves
    past UNSEEN — Django's PersonType choices are KNOWN/UNKNOWN with no
    "processing" bucket yet, so sending an unresolved track would force
    a guess. True frame-by-frame progressive entries (per the
    architecture doc) would need a PROCESSING value added to Django's
    PersonType/EventType choices — worth doing later, not required for
    a correct v1."""

    def __init__(self, django_client: DjangoClient | None = None):
        self._client = django_client or DjangoClient
        self._last_synced_at: dict = {}  # session.id -> datetime, heartbeat throttle

    def process(self, active_sessions: list[EventSession]) -> None:
        for session in active_sessions:
            self._sync_active(session)

    def _sync_active(self, session: EventSession) -> None:
        track = session.track
        if track is None or track.person_status == PersonStatus.UNSEEN:
            return  # no usable identity yet — wait for RecognitionProcessor

        if not session.timeline_created:
            self._create_or_update_timeline_event(session)
            return

        last_sync = self._last_synced_at.get(session.id)
        now = datetime.now(timezone.utc)
        if last_sync and (now - last_sync).total_seconds() < HEARTBEAT_SECONDS:
            return

        self._create_or_update_timeline_event(session)

    def _create_or_update_timeline_event(self, session: EventSession) -> None:
        track = session.track
        is_known = track.person_status == PersonStatus.KNOWN

        payload = {
            "home_id": str(session.house_id),
            "device_id": str(session.device_id),
            "event_type": "PERSON_DETECTED",
            "person_type": "KNOWN" if is_known else "UNKNOWN",
            "person_label": track.person_label if is_known else None,
            "member_id": str(track.member_id) if is_known and track.member_id else None,
            "face_profile_id": (
                str(track.face_profile_id) if is_known and track.face_profile_id else None
            ),
            "confidence_score": track.recognition_confidence or 0.0,
            "camera_location": session.camera_location,
            "image_url": None,   # attached at enrichment time
            "snapshot_url": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        response = self._client.send_detection_event(payload)
        if not isinstance(response, dict):
            return

        event_id = response.get("id")
        if not event_id:
            return


        event_id = response.get("id") if isinstance(response, dict) else None
        if event_id:
            session.timeline_event_id = event_id
            session.timeline_created = True
            session.timeline_updated = True
            self._last_synced_at[session.id] = datetime.now(timezone.utc)

    def cleanup(self, session_id) -> None:
        self._last_synced_at.pop(session_id, None)