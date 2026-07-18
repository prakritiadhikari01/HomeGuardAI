from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Optional, Tuple
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

    active: bool = True
    ended: bool = False

    history: Deque[Tuple[int, int, int, int]] = field(
        default_factory=lambda: deque(maxlen=50)
    )

    recognized_once: bool = False

    person_status: PersonStatus = PersonStatus.UNSEEN
    person_label: Optional[str] = None

    member_id: Optional[UUID] = None
    face_profile_id: Optional[UUID] = None

    recognition_confidence: Optional[float] = None

    best_face_crop: Optional[object] = None
    best_face_bbox: Optional[Tuple[int, int, int, int]] = None
    best_face_confidence: float = 0.0

    session_id: Optional[UUID] = None

    def update(self, detection: Detection):

        self.detection = detection

        self.last_seen = datetime.utcnow()

        self.age_frames += 1

        self.missing_frames = 0

        self.history.append(detection.bbox)

    def mark_missing(self):

        self.missing_frames += 1

    def deactivate(self):

        self.active = False

    def end(self):

        self.active = False
        self.ended = True

    def attach_recognition(
        self,
        *,
        status: PersonStatus,
        label: Optional[str],
        member_id: Optional[UUID],
        face_profile_id: Optional[UUID],
        confidence: Optional[float],
    ):

        self.person_status = status
        self.person_label = label

        self.member_id = member_id
        self.face_profile_id = face_profile_id

        self.recognition_confidence = confidence

        self.recognized_once = True

    def update_best_face(
        self,
        crop,
        bbox,
        confidence,
    ):

        if confidence > self.best_face_confidence:

            self.best_face_confidence = confidence
            self.best_face_bbox = bbox
            self.best_face_crop = crop

    def assign_session(self,session_id: UUID):
        self.session_id=session_id

    def has_session(self):
        return self.session_id is not None
    
    @property
    def is_finished(self):
        return self.ended
    
    @property
    def face_ready(self):
        return (
            self.best_face_confidence>0
            and self.best_face_crop is not None
        )

    @property
    def current_bbox(self):

        return self.detection.bbox

    @property
    def center(self):

        x1, y1, x2, y2 = self.current_bbox

        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2,
        )

    @property
    def duration_seconds(self):

        return (
            self.last_seen - self.first_seen
        ).total_seconds()

    @property
    def is_person(self):

        return self.detection.is_person

    @property
    def is_recognized(self):

        return self.person_status == PersonStatus.KNOWN

    @property
    def is_unknown_person(self):

        return self.person_status == PersonStatus.UNKNOWN
    
    @property
    def object_type(self):
        return self.detection.object_type