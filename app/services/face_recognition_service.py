# app/services/face_recognition_service.py

from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine

from app.services.django_client import DjangoClient
from app.services.face_embedding_service import FaceEmbeddingService


class FaceRecognitionService:
    """
    Recognizes a person by comparing the incoming embedding
    against all enrolled embeddings stored in Django.
    """

    THRESHOLD = 0.40

    def __init__(self):

        self.embedding_service = FaceEmbeddingService()

    def recognize(self, frame) -> dict:

        face_data = self.embedding_service.get_face_data(frame)

        if face_data is None:

            return {
                "status": "unknown"
            }

        query_embedding = face_data["embedding"]

        profiles = DjangoClient.get_all_faces()

        if not profiles:

            return {
                "status": "unknown"
            }

        best_profile = None
        best_distance = float("inf")

        for profile in profiles:

            distance = self._best_profile_distance(
                query_embedding=query_embedding,
                profile=profile,
            )

            if distance is None:
                continue

            if distance < best_distance:

                best_distance = distance
                best_profile = profile

        if (
            best_profile is not None
            and best_distance < self.THRESHOLD
        ):

            return {

                "status": "known",

                "person_label": best_profile["label_name"],

                "member_id": best_profile["member_id"],

                "face_profile_id": best_profile["id"],

                "confidence_score": round(
                    1 - best_distance,
                    4,
                ),

                "distance": round(
                    best_distance,
                    4,
                ),

            }

        return {

            "status": "unknown"

        }

    def _best_profile_distance(
        self,
        query_embedding: np.ndarray,
        profile: dict,
    ) -> Optional[float]:
        """
        Finds the closest stored embedding
        for one person's profile.
        """

        embeddings_by_pose = profile.get(
            "embeddings",
            {}
        )

        best_distance = float("inf")

        for pose_data in embeddings_by_pose.values():

            stored_embeddings = pose_data.get(
                "embeddings",
                []
            )

            for item in stored_embeddings:

                stored_embedding = np.asarray(
                    item["embedding"],
                    dtype=np.float32,
                )

                distance = cosine(
                    query_embedding,
                    stored_embedding,
                )

                if distance < best_distance:

                    best_distance = distance

        if best_distance == float("inf"):
            return None

        return best_distance