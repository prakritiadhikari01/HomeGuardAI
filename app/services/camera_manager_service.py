# app/services/camera_manager_service.py

import threading

from app.services.django_client import (
    DjangoClient
)

from app.services.camera_recognition_service import (
    CameraRecognitionService
)


class CameraManagerService:

    def __init__(self):

        self.running_cameras = {}

    def sync_cameras(self):
        print("Syncing cameras with Django backend...")
        cameras = (
            DjangoClient.get_active_cameras()
        )

        for camera in cameras:

            camera_id = camera["id"]

            if camera_id in self.running_cameras:
                continue

            worker = (
                CameraRecognitionService(
                    camera_url=camera["stream_url"],
                    device_id=camera_id
                )
            )

            thread = threading.Thread(
                target=worker.start,
                daemon=True
            )

            thread.start()

            self.running_cameras[
                camera_id
            ] = thread

            print(
                f"Started camera "
                f"{camera['name']}"
            )