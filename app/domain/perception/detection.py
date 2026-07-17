from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List
from uuid import UUID


class ObjectType(str, Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    PACKAGE = "package"
    UNKNOWN = "unknown"


class PersonStatus(str, Enum):
    UNSEEN = "unseen"      # Face not visible yet
    KNOWN = "known"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Detection:
    """
    Represents one detected object in one frame.

    This object is progressively enriched as it moves through
    the AI pipeline.
    """

    # YOLO
    object_type: ObjectType
    confidence: float
    bbox: Tuple[int, int, int, int]

    # Frame information
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None

    # Tracking (filled by ByteTrack)
    track_id: Optional[int] = None

    # Face Detection
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_confidence: Optional[float] = None

    # Face Recognition
    person_status: PersonStatus = PersonStatus.UNSEEN
    person_label: Optional[str] = None
    member_id: Optional[UUID] = None
    face_profile_id: Optional[UUID] = None
    recognition_confidence: Optional[float] = None

    # Convenience

    @property
    def is_person(self) -> bool:
        return self.object_type == ObjectType.PERSON

    @property
    def is_vehicle(self) -> bool:
        return self.object_type == ObjectType.VEHICLE

    @property
    def is_animal(self) -> bool:
        return self.object_type == ObjectType.ANIMAL

    @property
    def is_recognized(self) -> bool:
        return self.person_status == PersonStatus.KNOWN

    @property
    def is_unknown_person(self) -> bool:
        return self.person_status == PersonStatus.UNKNOWN
    

    
@dataclass(slots=True)
class DetectionResult:
    """
    All detections found in a single frame.
    """

    detections: List[Detection] = field(default_factory=list)

    @property
    def has_detection(self) -> bool:
        return len(self.detections) > 0

    @property
    def people(self) -> List[Detection]:
        return [d for d in self.detections if d.is_person]

    @property
    def vehicles(self) -> List[Detection]:
        return [d for d in self.detections if d.is_vehicle]

    @property
    def animals(self) -> List[Detection]:
        return [d for d in self.detections if d.is_animal]