# app/detection/yolo_detector.py
import threading
from ultralytics import YOLO
from app.core.config import settings

PERSON_CLASS_ID = 0  # COCO index for "person"


class YoloDetector:
    """
    Second pipeline stage: confirms the motion that triggered this frame
    was actually caused by a person (not a pet, curtain, or lighting
    change) before paying the cost of face detection/recognition.

    Singleton — the model loads once, shared across all camera threads,
    same reasoning as ModelRegistry.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._load()
        return cls._instance

    def _load(self):
        print("[YoloDetector] Loading YOLOv8n — should print ONCE per process.")
        self.model = YOLO("yolov8n.pt")  # auto-downloads on first run, needs internet once

    def detect_people(self, frame):
        results = self.model.predict(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=settings.YOLO_PERSON_CONFIDENCE,
            verbose=False,
        )

        people = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                people.append({
                    "bbox": [int(v) for v in xyxy],
                    "confidence": float(box.conf[0]),
                })

        return people

    def has_person(self, frame) -> bool:
        return len(self.detect_people(frame)) > 0