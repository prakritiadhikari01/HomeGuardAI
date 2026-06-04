#app\services\camera_registry.py
from threading import Thread

from app.services.camera_recognition_service import CameraRecognitionService

class CameraRegistry:
    active_cameras = {}

    @classmethod
    def register_camera(cls, payload):
        device_id = payload["camera_id"]
        
        if device_id  in cls.active_cameras:
            print(f"Camera {device_id} is already registered.")
            return
        
        worker=CameraRecognitionService(
            device_id=device_id,
            stream_url=payload["stream_url"],
            home_id=payload["home_id"]
            )
        thread=Thread(target=worker.run_recognition_loop, daemon=True)
        thread.start()
        cls.active_cameras[device_id] = worker
        print(f"Camera {device_id} registered and recognition loop started.")