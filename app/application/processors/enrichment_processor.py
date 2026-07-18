import cv2

from app.domain.session.event_session import EventSession
from app.infrastructure.ai.qwen_vl_service import QwenVLService
from app.infrastructure.api.django_client import DjangoClient
from app.infrastructure.media.clip_recorder import ClipRecorder
from app.infrastructure.media.snapshot_service import SnapshotService
from app.infrastructure.storage.cloudinary_service import CloudinaryService


class EnrichmentProcessor:
    """Runs once per finished session — after TimelineProcessor, before
    AlertProcessor. Owns everything "slow": clip finalize/upload,
    snapshot generation, VLM summary pass. Alert evaluation happens
    after this deliberately — a VLM finding can change whether/how
    severely AlertProcessor should fire, only possible once enrichment
    has already run."""

    def __init__(self, clip_recorder: ClipRecorder, django_client: DjangoClient | None = None):
        self._clip_recorder = clip_recorder
        self._client = django_client or DjangoClient

    def process(self, session: EventSession) -> None:
        if session.track is None:
            session.expire()
            return

        track_id = session.track.track_id

        if not session.timeline_created or not session.timeline_event_id:
            # Track ended before ever producing a resolved identity (e.g.
            # someone passed by back-turned the whole time) — nothing in
            # Django to enrich, no clip worth keeping.
            session.expire()
            self._clip_recorder.discard(track_id)
            return

        session.start_enrichment()

        clip_path = self._clip_recorder.finalize(track_id)
        clip_url = CloudinaryService.upload_video(clip_path) if clip_path else None
        if clip_path and clip_url is None:
            session.expire()
            return

        snapshot = SnapshotService.generate(session.track)

        representative_frame = (
            self._pick_representative_frame(clip_path) if clip_path else None
        )
        summary = (
            QwenVLService.summarize(representative_frame, session.camera_location)
            if representative_frame is not None
            else None
        ) or self._fallback_summary(session)

        payload = {
            "clip_url": clip_url,
            "thumbnail_url": snapshot.get("thumbnail_url"),
            "event_summary": summary,
            "attributes": {"duration_seconds": int(session.duration_seconds)},
        }

        self._client.enrich_event(str(session.timeline_event_id), payload)

        session.clip_path = clip_url
        session.summary = summary
        session.snapshot_path =snapshot.get("thumbnail_url")
        session.complete_enrichment()
        # Timeline heartbeat no longer needed once the session is finished.
        # Prevents TimelineProcessor from accumulating completed sessions.
        if hasattr(self._client, "timeline_processor"):
            self._client.timeline_processor.cleanup(session.id)


    @staticmethod
    def _fallback_summary(session: EventSession) -> str:
        """Rule-based summary used whenever the VLM pass fails or Ollama
        is unreachable — the "simple events get structured-data-to-text"
        case from the architecture doc."""
        track = session.track
        who = track.person_label if track.is_recognized else "An unknown person"
        where = f" at {session.camera_location}" if session.camera_location else ""
        return f"{who} was seen for about {int(session.duration_seconds)} seconds{where}."

    @staticmethod
    def _pick_representative_frame(clip_path: str):
        """Re-reads the just-written clip to grab its middle frame,
        avoiding keeping a second copy of every frame in memory just for
        the VLM call — costs one extra disk read per finished session."""
        cap = cv2.VideoCapture(clip_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        success, frame = cap.read()
        cap.release()
        return frame if success else None
