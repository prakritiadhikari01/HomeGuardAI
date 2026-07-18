from __future__ import annotations

from ultralytics import YOLO

from app.core.config import settings
from app.domain.perception.detection import Detection, DetectionResult, ObjectType
from app.domain.perception.perception_result import PerceptionResult


class YOLODetector:
    """Runs YOLO inference and converts results into domain Detection
    objects. Model loads once per instance — RuntimeManager keeps one
    per camera worker (Ultralytics YOLO objects are not thread-safe to
    share across concurrent predict() calls)."""

    CLASS_MAPPING = {
        "person": ObjectType.PERSON,
        "car": ObjectType.VEHICLE,
        "truck": ObjectType.VEHICLE,
        "bus": ObjectType.VEHICLE,
        "motorcycle": ObjectType.VEHICLE,
        "bicycle": ObjectType.VEHICLE,
        "dog": ObjectType.ANIMAL,
        "cat": ObjectType.ANIMAL,
        "horse": ObjectType.ANIMAL,
        "cow": ObjectType.ANIMAL,
        "sheep": ObjectType.ANIMAL,
        "bird": ObjectType.ANIMAL,
        "backpack": ObjectType.PACKAGE,
        "suitcase": ObjectType.PACKAGE,
        "handbag": ObjectType.PACKAGE,
    }

    def __init__(self, confidence_threshold: float | None = None):
        self.model = YOLO(settings.YOLO_MODEL_PATH)
        self.confidence_threshold = confidence_threshold or settings.YOLO_CONFIDENCE

    def detect(
        self,
        frame,
        frame_index: int | None = None,
        timestamp: float | None = None,
    ) -> PerceptionResult:
        prediction = self.model.predict(
            source=frame, conf=self.confidence_threshold, verbose=False
        )[0]

        detections = []
        for box in prediction.boxes:
            class_id = int(box.cls.item())
            class_name = prediction.names[class_id]
            object_type = self.CLASS_MAPPING.get(class_name, ObjectType.UNKNOWN)
            if object_type == ObjectType.UNKNOWN:
                continue

            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detections.append(
                Detection(
                    object_type=object_type,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    frame_index=frame_index,
                    timestamp=timestamp,
                )
            )

        return PerceptionResult(
            detections=DetectionResult(detections=detections),
            raw_prediction=prediction,
        )
