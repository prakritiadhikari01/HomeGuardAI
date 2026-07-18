from app.domain.perception.perception_result import PerceptionResult
from app.infrastructure.vision.yolo_detector import YOLODetector


class DetectionProcessor:
    """Stage 2 — YOLO object detection. Gated by PipelineProcessor (only
    runs when MotionProcessor found motion, or motion detection is off).
    YOLODetector.detect() already builds the full PerceptionResult, so
    this is a thin pass-through that keeps PipelineProcessor depending
    on the application-layer interface, not YOLODetector directly."""

    def __init__(self, yolo_detector: YOLODetector):
        self._detector = yolo_detector

    def process(
        self, frame, frame_index: int | None = None, timestamp: float | None = None
    ) -> PerceptionResult:
        return self._detector.detect(frame, frame_index=frame_index, timestamp=timestamp)
