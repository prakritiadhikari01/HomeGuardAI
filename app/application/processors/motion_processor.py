from app.domain.perception.motion import MotionResult
from app.infrastructure.vision.opencv_motion_detector import OpenCVMotionDetector


class MotionProcessor:
    """Stage 1 — cheap motion gate, run every frame. Wraps
    OpenCVMotionDetector so PipelineProcessor never knows OpenCV exists.
    No domain logic beyond "did anything change" — whether motion
    matters (loitering, ignore_animals) is decided downstream."""

    def __init__(self, motion_detector: OpenCVMotionDetector):
        self._detector = motion_detector

    def process(self, frame) -> MotionResult:
        return self._detector.detect(frame)
