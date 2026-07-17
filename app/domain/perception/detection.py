from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
from uuid import UUID


class ObjectType(str, Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    PACKAGE = "package"
    UNKNOWN = "unknown"


class PersonStatus(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    UNSEEN = "unseen"


@dataclass(slots=True)
class Detection:
    """
    One object detected in a single frame.

    Produced by YOLO and later attached to a Track.
    """

    object_type: ObjectType
    confidence: float
    bbox: Tuple[int, int, int, int]

    frame_index: Optional[int] = None
    timestamp: Optional[float] = None

    # Face information (if available)
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_confidence: Optional[float] = None

    # Recognition information (filled after face recognition)
    person_status: PersonStatus = PersonStatus.UNSEEN
    person_label: Optional[str] = None
    member_id: Optional[UUID] = None
    face_profile_id: Optional[UUID] = None
    recognition_confidence: Optional[float] = None

    @property
    def is_person(self) -> bool:
        return self.object_type == ObjectType.PERSON

    @property
    def is_recognized(self) -> bool:
        return self.person_status == PersonStatus.KNOWN

    @property
    def is_unknown_person(self) -> bool:
        return self.person_status == PersonStatus.UNKNOWN