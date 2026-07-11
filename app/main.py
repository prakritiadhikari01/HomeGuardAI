#app/main.py

from fastapi import FastAPI

from app.api.routes import router
from app.api.enrollment_api import router as enrollment_router

from app.services.camera_manager_service import (
    CameraManagerService
)
from fastapi.middleware.cors import CORSMiddleware

import threading
import time


app = FastAPI(
    title="HomeGuard AI Service",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Existing APIs
app.include_router(router)

# Enrollment APIs
app.include_router(enrollment_router)

camera_manager = CameraManagerService()


def sync_loop():

    print("Starting camera sync loop...")

    while True:

        try:

            camera_manager.sync_cameras()

        except Exception as e:

            print("Camera sync error:", e)

        time.sleep(30)


@app.get("/")
def root():

    return {

        "service": "HomeGuard AI Engine",

        "version": "1.0.0",

        "status": "running"

    }


@app.get("/health")
def health():

    return {

        "status": "healthy"

    }


@app.on_event("startup")
def startup():

    threading.Thread(

        target=sync_loop,

        daemon=True

    ).start()