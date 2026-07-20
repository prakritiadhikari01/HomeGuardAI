from __future__ import annotations

from typing import Dict

import supervision as sv

from app.domain.perception.perception_result import PerceptionResult
from app.domain.perception.track import Track
from app.domain.perception.track_result import TrackResult


class ByteTrackTracker:
    """Wraps ByteTrack: converts YOLO detections into persistent tracks,
    preserves Track objects between frames, drops tracks that have been
    missing too long. One instance per camera — tracker state (next
    track id, active tracks) is per-stream."""

    MAX_MISSING_FRAMES = 30

    def __init__(self):
        self._tracker = sv.ByteTrack()
        self._tracks: Dict[int, Track] = {}

    def update(self, perception: PerceptionResult) -> TrackResult:
        sv_detections = sv.Detections.from_ultralytics(perception.raw_prediction)
        tracked = self._tracker.update_with_detections(sv_detections)

        active_tracks = []
        alive_ids = set()

        detections = perception.detections.detections

        # ByteTrack may return fewer/reordered detections than YOLO.
        count = min(
            len(detections),
            len(tracked.tracker_id),
        )

        for index in range(count):

            tracker_id = tracked.tracker_id[index]

            if tracker_id is None:
                continue

            tracker_id = int(tracker_id)
            alive_ids.add(tracker_id)

            detection = detections[index]
            detection.track_id = tracker_id

            if tracker_id not in self._tracks:
                track = Track(
                    track_id=tracker_id,
                    detection=detection,
                )
                self._tracks[tracker_id] = track
            else:
                self._tracks[tracker_id].update(detection)

            active_tracks.append(self._tracks[tracker_id])

        ended = []

        for track_id in list(self._tracks.keys()):

            if track_id in alive_ids:
                continue

            track = self._tracks[track_id]
            track.mark_missing()

            if track.missing_frames > self.MAX_MISSING_FRAMES:
                track.end()
                ended.append(track)
                del self._tracks[track_id]

        return TrackResult(
            tracks=active_tracks,
            ended_tracks=ended,
        )