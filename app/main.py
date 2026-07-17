# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import threading
import time

from app.api.routes import router
from app.api.enrollment_api import router as enrollment_router
from app.services.camera_manager_service import CameraManagerService
from app.recognition.known_faces_store import KnownFacesStore
from app.core.config import settings

app = FastAPI(title="HomeGuard AI Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(enrollment_router)

camera_manager = CameraManagerService()
known_faces_store = KnownFacesStore()


def sync_loop():
    print("Starting camera + known-faces sync loop...")
    while True:
        try:
            camera_manager.sync_cameras()
            known_faces_store.refresh()
        except Exception as e:
            print("Sync loop error:", e)
        time.sleep(settings.CAMERA_SYNC_INTERVAL_SECONDS)


@app.get("/")
def root():
    return {"service": "HomeGuard AI Engine", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.on_event("startup")
def startup():
    known_faces_store.refresh()  # populate cache immediately, don't wait for the first 30s tick
    threading.Thread(target=sync_loop, daemon=True).start()