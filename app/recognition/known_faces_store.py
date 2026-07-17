# app/recognition/known_faces_store.py
import threading
from app.services.django_client import DjangoClient


class KnownFacesStore:
    """
    In-memory cache of enrolled face profiles, refreshed periodically
    (every CAMERA_SYNC_INTERVAL_SECONDS in main.py) instead of being
    re-fetched from Django on every frame — previous code pulled the
    full ~300KB /faces/all/ response every ~5 seconds.

    Keeps Django's raw nested pose structure intact (profile["embeddings"]
    [pose]["embeddings"][i]["embedding"]) since FaceRecognitionService
    matches against every pose, not a flattened single embedding.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.profiles = []
        return cls._instance

    def refresh(self):
        try:
            profiles = DjangoClient.get_all_faces()
            self.profiles = profiles
            print(f"[KnownFacesStore] Refreshed — {len(profiles)} face profiles cached")
        except Exception as e:
            print("[KnownFacesStore] Refresh error:", e)

    def get_all_faces(self):
        return self.profiles