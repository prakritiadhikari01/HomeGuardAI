import threading

from app.infrastructure.api.django_client import DjangoClient

# Moved from app/recognition/known_faces_store.py — same class, now
# imports the real infrastructure/api DjangoClient instead of the old
# app/services one, which no longer exists.


class KnownFacesStore:
    """In-memory cache of enrolled face profiles, refreshed on an
    interval (RuntimeManager's sync loop) instead of re-fetched from
    Django every frame. Singleton — one cache shared by every camera
    worker in the process."""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.profiles = []
        return cls._instance

    def refresh(self) -> None:
        try:
            profiles = DjangoClient.get_all_faces()
            with self._lock:
                self.profiles = profiles
            print(f"[KnownFacesStore] Refreshed — {len(profiles)} face profiles cached")
        except Exception as e:
            print(f"[KnownFacesStore] Refresh error: {e}")

    def get_all_faces(self) -> list[dict]:
        with self._lock:
            return self.profiles
