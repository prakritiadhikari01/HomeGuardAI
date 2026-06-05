# app/services/camera_manager_service.py

import threading

from app.services.django_client import DjangoClient
from app.services.camera_recognition_service import (
    CameraRecognitionService,
)


class CameraManagerService:

    def __init__(self):
        self.running_cameras = {}

    def sync_cameras(self):

        cameras = DjangoClient.get_active_cameras()

        incoming = {
            camera["id"]: camera
            for camera in cameras
        }

        # remove offline/deleted cameras
        for camera_id in list(
            self.running_cameras.keys()
        ):

            if camera_id not in incoming:

                self.running_cameras[
                    camera_id
                ]["worker"].stop()

                del self.running_cameras[
                    camera_id
                ]

                print(
                    f"Stopped camera "
                    f"{camera_id}"
                )

        # create/update cameras
        for camera_id, camera in incoming.items():

            existing = (
                self.running_cameras
                .get(camera_id)
            )

            if not existing:

                self._start_camera(camera)

                continue

            old_url = existing["stream_url"]

            if old_url != camera["stream_url"]:

                print(
                    f"URL changed for "
                    f"{camera['name']}"
                )

                existing[
                    "worker"
                ].stop()

                del self.running_cameras[
                    camera_id
                ]

                self._start_camera(camera)

    def _start_camera(
        self,
        camera
    ):

        worker = (
            CameraRecognitionService(
                device_id=camera["id"],
                home_id=camera["home_id"],
                camera_name=camera["name"],
                location=camera["location"],
                stream_url=camera["stream_url"],
            )
        )

        thread = threading.Thread(
            target=worker.start,
            daemon=True,
        )

        thread.start()

        self.running_cameras[
            camera["id"]
        ] = {
            "worker": worker,
            "thread": thread,
            "stream_url": camera[
                "stream_url"
            ],
        }

        print(
            f"Started camera "
            f"{camera['name']}"
        )