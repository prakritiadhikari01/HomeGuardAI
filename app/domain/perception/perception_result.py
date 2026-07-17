from dataclasses import dataclass

from ultralytics.engine.results import Results

from app.domain.perception.detection import DetectionResult


@dataclass(slots=True)
class PerceptionResult:
    """
    Output of the perception stage.

    Contains:
    - Domain detections
    - Raw YOLO prediction (for ByteTrack)
    """

    detections: DetectionResult

    raw_prediction: Results