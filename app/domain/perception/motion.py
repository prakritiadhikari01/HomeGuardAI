from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(slots=True)
class MotionResult:
    """
    Result of motion detection for a single frame.

    This is produced by the OpenCV motion detector and consumed
    by the camera worker.
    """

    motion_detected: bool
    motion_score: float =0.0
    motion_area:float = 0.0
    changed_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def has_motion(self) -> bool:
        return self.motion_detected