# app/services/camera_recognition_service.py
import cv2
import time

from app.detection.motion_detection import MotionDetector
from app.services.ai_event_orchestrator import AIEventOrchestrator
from app.detection.yolo_detector import YoloDetector
from app.core.config import settings


class CameraRecognitionService:

    def __init__(self, device_id, home_id, camera_name, location, stream_url):
        self.device_id = device_id
        self.home_id = home_id
        self.camera_name = camera_name
        self.location = location
        self.stream_url = stream_url

        self.running = True
        self.last_detection = 0
        self.motion_detector = MotionDetector(min_area=settings.MOTION_MIN_AREA)
        self.yolo_detector = YoloDetector() if settings.ENABLE_YOLO_GATE else None

    def stop(self):
        self.running = False

    def start(self):
        cap = cv2.VideoCapture(self.stream_url)

        while self.running:
            success, frame = cap.read()

            if not success:
                time.sleep(2)
                cap.release()
                cap = cv2.VideoCapture(self.stream_url)
                continue

            # Stage 1: motion — cheapest check, runs on every single frame.
            if not self.motion_detector.detect(frame):
                continue

            current_time = time.time()
            if current_time - self.last_detection < settings.EVENT_COOLDOWN_SECONDS:
                continue

            # Stage 2: object detection — confirms a person caused the motion
            # (toggle via ENABLE_YOLO_GATE if this is too slow on CPU).
            if self.yolo_detector is not None and not self.yolo_detector.has_person(frame):
                continue

            # Stage 3+: face recognition + event creation.
            AIEventOrchestrator.process_frame(
                frame=frame,
                device_id=self.device_id,
                home_id=self.home_id,
                location=self.location,
            )
            self.last_detection = current_time

        cap.release()