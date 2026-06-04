#app\services\face_recognition_service.py
import numpy as np
from scipy.spatial.distance import cosine

from app.recognition import known_faces_store
from app.services.face_embedding_service import FaceEmbeddingService
from app.services.django_client import DjangoClient


class FaceRecognitionService:

    def __init__(self):

        self.embedding_service = FaceEmbeddingService()

    def recognize(self, frame):

        embedding = self.embedding_service.get_embedding(frame)

        if embedding is None:
            return None

        registered_faces = known_faces_store.get_all_faces()
        best_match = None
        best_distance = 999

        for registered_face in registered_faces:

            stored_embedding = np.array(
                registered_face["embedding"],
                dtype=np.float32
            )

            distance = cosine(
                embedding,
                stored_embedding
            )

            if distance < best_distance:
                best_distance = distance
                best_match = registered_face

        if best_match and best_distance < 0.4:

            return {
                "status": "known",
                "name": best_match["label_name"],
                "member_id": best_match["member_id"],
                "distance": float(best_distance)
            }

        return {
            "status": "unknown"
        }