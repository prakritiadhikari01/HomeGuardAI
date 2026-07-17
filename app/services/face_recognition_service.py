# app/services/face_recognition_service.py
from typing import Optional
import numpy as np
from scipy.spatial.distance import cosine

from app.services.face_embedding_service import FaceEmbeddingService
from app.recognition.known_faces_store import KnownFacesStore
from app.core.config import settings


class FaceRecognitionService:
    def __init__(self):
        self.embedding_service = FaceEmbeddingService()
        self.known_faces_store = KnownFacesStore()  # reads the cache, never fetches per-frame

    def recognize(self, frame) -> dict:
        face_data = self.embedding_service.get_face_data(frame)
        if face_data is None:
            return {"status": "unknown"}

        query_embedding = face_data["embedding"]
        profiles = self.known_faces_store.get_all_faces()

        if not profiles:
            return {"status": "unknown"}

        best_profile = None
        best_distance = float("inf")

        for profile in profiles:
            distance = self._best_profile_distance(query_embedding, profile)
            if distance is None:
                continue
            if distance < best_distance:
                best_distance = distance
                best_profile = profile

        if best_profile is not None and best_distance < settings.FACE_MATCH_THRESHOLD:
            return {
                "status": "known",
                "person_label": best_profile["label_name"],
                "member_id": best_profile["member_id"],
                "face_profile_id": best_profile["id"],
                "confidence_score": round(1 - best_distance, 4),
                "distance": round(best_distance, 4),
            }

        return {"status": "unknown"}

    def _best_profile_distance(self, query_embedding, profile) -> Optional[float]:
        embeddings_by_pose = profile.get("embeddings", {})
        best_distance = float("inf")

        for pose_data in embeddings_by_pose.values():
            for item in pose_data.get("embeddings", []):
                stored_embedding = np.asarray(item["embedding"], dtype=np.float32)
                distance = cosine(query_embedding, stored_embedding)
                if distance < best_distance:
                    best_distance = distance

        return best_distance if best_distance != float("inf") else None