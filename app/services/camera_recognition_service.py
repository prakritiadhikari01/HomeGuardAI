#app\services\camera_recognition_service.py
import cv2

from app.services.ai_event_orchestrator import (
    AIEventOrchestrator
)
import time

class CameraRecognitionService:
    EVENT_COOLDOWN = 5  # seconds
    def __init__(
        self,
        camera_url,
        device_id
    ):
        self.camera_url = camera_url
        self.device_id = device_id
        self.last_detection = 0

    def start(self):

        cap = cv2.VideoCapture(
            self.camera_url
        )

        while True:

            success, frame = cap.read()

            if not success:
                break

            current_time = time.time()

            if current_time - self.last_detection >= self.EVENT_COOLDOWN:

                AIEventOrchestrator.process_frame(
                    frame=frame,
                    device_id=self.device_id
                )

                self.last_detection = current_time

            cv2.imshow(
                "HomeGuard AI",
                frame
            )

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()