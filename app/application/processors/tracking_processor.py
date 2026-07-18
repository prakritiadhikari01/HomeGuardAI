from app.domain.perception.perception_result import PerceptionResult
from app.domain.perception.track_result import TrackResult
from app.infrastructure.vision.bytetrack_tracker import ByteTrackTracker


class TrackingProcessor:
    """Stage 3 — object tracking only. Owns the ByteTrackTracker
    instance (persistent IDs need per-camera state). Knows nothing about
    EventSession — tracking is "where are the objects", session
    lifecycle is "what real-world events are these tracks", kept as
    separate concerns so a future non-ByteTrack tracker could produce a
    TrackResult and feed the same SessionProcessor unchanged."""

    def __init__(self, tracker: ByteTrackTracker):
        self._tracker = tracker

    def process(self, perception_result: PerceptionResult) -> TrackResult:
        return self._tracker.update(perception_result)
