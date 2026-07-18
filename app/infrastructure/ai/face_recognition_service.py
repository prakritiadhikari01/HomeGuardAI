from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine

from app.core.config import settings
from app.infrastructure.ai.known_faces_store import KnownFacesStore


class FaceRecognitionService:
    """
    Matches ONE already-extracted embedding against the cached known-faces
    profiles. This was previously folded together with face DETECTION in
    the old app/services/face_recognition_service.py (which called
    FaceEmbeddingService.get_face_data() itself, detecting AND matching
    in one method). Splitting them lets RecognitionProcessor call
    InsightFaceService for quality scoring on every frame, and only call
    match_embedding() here once a track's best crop clears the quality
    bar — matching runs far less often than detection now.

    Keeps FaceProfile's raw nested pose structure intact:
    profile["embeddings"][pose]["embeddings"][i]["embedding"], since a
    query embedding is compared against every stored pose, not a single
    flattened embedding per person.
    """

    def __init__(self, known_faces_store: Optional[KnownFacesStore] = None):
        self._store = known_faces_store or KnownFacesStore()

    def match_embedding(self, embedding: np.ndarray) -> dict:
        profiles = self._store.get_all_faces()
        if not profiles:
            return {"status": "unknown"}

        best_profile = None
        best_distance = float("inf")

        for profile in profiles:
            distance = self._best_profile_distance(embedding, profile)
            if distance is None:
                continue
            if distance < best_distance:
                best_distance = distance
                best_profile = profile

        if best_profile is not None and best_distance < settings.FACE_MATCH_THRESHOLD:
            return {
                "status": "known",
                "person_label": best_profile.get("label_name"),
                "member_id": best_profile.get("member_id"),
                "face_profile_id": best_profile.get("id"),
                "confidence_score": round(1 - best_distance, 4),
                "distance": round(best_distance, 4),
            }

        return {
            "status": "unknown",
            "person_label":None,
            "member_id":None,
            "face_profile_id":None,
            "confidence_score":None,
        }

    @staticmethod
    def _best_profile_distance(query_embedding: np.ndarray, profile: dict) -> Optional[float]:
        embeddings_by_pose = profile.get("embeddings", {}) or {}
        best_distance = float("inf")

        for pose_data in embeddings_by_pose.values():
            for item in pose_data.get("embeddings", []):
                stored_embedding = np.asarray(item["embedding"], dtype=np.float32)
                distance = cosine(query_embedding, stored_embedding)
                if distance < best_distance:
                    best_distance = distance

        return best_distance if best_distance != float("inf") else None
