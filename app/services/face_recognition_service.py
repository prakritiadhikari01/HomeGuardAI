# app/services/face_recognition_service.py

import numpy as np
from scipy.spatial.distance import cosine

from app.services.face_embedding_service import (
    FaceEmbeddingService
)

from app.services.django_client import (
    DjangoClient
)


class FaceRecognitionService:

    THRESHOLD = 0.40

    def __init__(self):

        self.embedding_service = (
            FaceEmbeddingService()
        )

    def recognize(self, frame):

        embedding = (
            self.embedding_service.get_embedding(
                frame
            )
        )

        if embedding is None:

            return {
                "status": "unknown"
            }

        faces = (
            DjangoClient.get_all_faces()
        )

        best_match = None
        best_distance = 999

        for face in faces:

            stored_embedding = np.array(
                face["embedding"],
                dtype=np.float32
            )

            distance = cosine(
                embedding,
                stored_embedding
            )

            if distance < best_distance:

                best_distance = distance
                best_match = face

        if (
            best_match
            and best_distance < self.THRESHOLD
        ):

            return {
                "status": "known",
                "person_label": best_match["label_name"],
                "member_id": best_match["member_id"],
                "face_profile_id": best_match["id"],
                "confidence_score": float(
                    1 - best_distance
                )
            }

        return {
            "status": "unknown"
        }