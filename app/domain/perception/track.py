from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
from uuid import UUID

from app.domain.perception.detection import Detection, PersonStatus


@dataclass(slots=True)
class Track:
    """
    A tracked object that persists across multiple frames.

    This is the object that connects YOLO detections,
    ByteTrack tracking, and face recognition.
    """

    track_id: int
    detection: Detection

    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)

    age_frames: int = 1
    missing_frames: int = 0

    # History of bounding boxes for movement analysis
    history: List[Tuple[int, int, int, int]] = field(default_factory=list)

    # Recognition state
    person_status: PersonStatus = PersonStatus.UNSEEN
    person_label: Optional[str] = None
    member_id: Optional[UUID] = None
    face_profile_id: Optional[UUID] = None
    recognition_confidence: Optional[float] = None

    # Best face captured during this track
    best_face_bbox: Optional[Tuple[int, int, int, int]] = None
    best_face_confidence: float = 0.0
    best_face_frame: Optional[object] = None

    # Session link (filled by EventSessionManager)
    session_id: Optional[str] = None

    def update(self, detection: Detection) -> None:
        """Update the track with a new detection from the next frame."""
        self.detection = detection
        self.last_seen = datetime.utcnow()
        self.age_frames += 1
        self.missing_frames = 0
        self.history.append(detection.bbox)

    def mark_missing(self) -> None:
        """Called when the tracker temporarily loses this object."""
        self.missing_frames += 1

    def attach_recognition(
        self,
        *,
        person_status: PersonStatus,
        person_label: Optional[str] = None,
        member_id: Optional[UUID] = None,
        face_profile_id: Optional[UUID] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Attach face recognition result to this track."""
        self.person_status = person_status
        self.person_label = person_label
        self.member_id = member_id
        self.face_profile_id = face_profile_id
        self.recognition_confidence = confidence

    def update_best_face(
        self,
        face_bbox: Tuple[int, int, int, int],
        face_confidence: float,
        frame: object,
    ) -> None:
        """Keep the best face seen during this track."""
        if face_confidence > self.best_face_confidence:
            self.best_face_confidence = face_confidence
            self.best_face_bbox = face_bbox
            self.best_face_frame = frame

    @property
    def duration_seconds(self) -> float:
        return (self.last_seen - self.first_seen).total_seconds()

    @property
    def is_recognized(self) -> bool:
        return self.person_status == PersonStatus.KNOWN

    @property
    def is_unknown_person(self) -> bool:
        return self.person_status == PersonStatus.UNKNOWN

    @property
    def current_bbox(self) -> Tuple[int, int, int, int]:
        return self.detection.bbox