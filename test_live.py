from app.services.camera_recognition_service import (
    LiveRecognitionService
)

service = LiveRecognitionService()

service.start()