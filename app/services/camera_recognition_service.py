# app/services/camera_recognition_service.py

import cv2
import time

from app.services.ai_event_orchestrator import (
    AIEventOrchestrator
)


class CameraRecognitionService:

    EVENT_COOLDOWN = 5

    def __init__(
        self,
        device_id,
        home_id,
        camera_name,
        location,
        stream_url,
    ):
        self.device_id = device_id
        self.home_id = home_id
        self.camera_name = camera_name
        self.location = location
        self.stream_url = stream_url

        self.running = True
        self.last_detection = 0

    def stop(self):
        self.running = False

    def start(self):

        cap = cv2.VideoCapture(
            self.stream_url
        )

        while self.running:

            success, frame = cap.read()

            if not success:

                time.sleep(2)

                cap.release()

                cap = cv2.VideoCapture(
                    self.stream_url
                )

                continue

            current_time = time.time()

            if (
                current_time
                - self.last_detection
                >= self.EVENT_COOLDOWN
            ):

                AIEventOrchestrator.process_frame(
                    frame=frame,
                    device_id=self.device_id,
                    home_id=self.home_id,
                    location=self.location,
                )

                self.last_detection = (
                    current_time
                )

        cap.release()