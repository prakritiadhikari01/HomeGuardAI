# app/main.py


from fastapi import FastAPI

from app.api.routes import router

from app.services.camera_manager_service import (
    CameraManagerService
)

import threading
import time

app = FastAPI(
    title="HomeGuard AI Service",
    version="1.0.0"
)

app.include_router(router)

camera_manager = (
    CameraManagerService()
)


def sync_loop():
    print("Starting camera sync loop...")
    while True:

        try:

            camera_manager.sync_cameras()

        except Exception as e:
            print(
                "Camera sync error:",
                e
            )

        time.sleep(30)


@app.on_event("startup")
def startup():

    threading.Thread(
        target=sync_loop,
        daemon=True
    ).start()