from __future__ import annotations

from typing import Dict

import supervision as sv

from app.domain.perception.perception_result import PerceptionResult
from app.domain.perception.track import Track
from app.domain.perception.track_result import TrackResult


class ByteTrackTracker:
    """
    Wraps ByteTrack.

    Responsibilities
    ----------------
    - Convert YOLO detections into persistent tracks.
    - Preserve Track objects between frames.
    - Remove tracks that disappear.
    """

    def __init__(self):

        self._tracker = sv.ByteTrack()

        self._tracks: Dict[int, Track] = {}

    def update(
        self,
        perception: PerceptionResult,
    ) -> TrackResult:

        sv_detections = sv.Detections.from_ultralytics(
            perception.raw_prediction
        )

        tracked = self._tracker.update_with_detections(
            sv_detections
        )

        active_tracks = []

        alive_ids = set()

        for index, tracker_id in enumerate(tracked.tracker_id):

            if tracker_id is None:
                continue

            tracker_id = int(tracker_id)

            alive_ids.add(tracker_id)

            detection = perception.detections.detections[index]

            if tracker_id not in self._tracks:

                track = Track(
                    track_id=tracker_id,
                    detection=detection,
                )

                self._tracks[tracker_id] = track

            else:

                self._tracks[tracker_id].update(
                    detection
                )

            active_tracks.append(
                self._tracks[tracker_id]
            )

        ended = []

        for track_id in list(self._tracks.keys()):

            if track_id in alive_ids:
                continue

            track = self._tracks[track_id]

            track.mark_missing()

            if track.missing_frames > 30:

                track.end()

                ended.append(track)

                del self._tracks[track_id]

        return TrackResult(

            tracks=active_tracks,

            ended_tracks=ended,

        )