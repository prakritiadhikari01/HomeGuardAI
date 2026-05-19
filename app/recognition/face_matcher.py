import numpy as np

from app.recognition.known_faces_store import KnownFacesStore


class FaceMatcher:

    THRESHOLD = 0.50

    @staticmethod
    def cosine_similarity(a, b):

        return np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )

    @classmethod
    def match_face(cls, embedding):

        known_faces = KnownFacesStore.get_all_faces()

        best_score = 0
        best_user_id = None

        for user_id, stored_embedding in known_faces.items():

            similarity = cls.cosine_similarity(
                embedding,
                stored_embedding
            )

            if similarity > best_score:
                best_score = similarity
                best_user_id = user_id

        if best_score >= cls.THRESHOLD:
            return {
                "matched": True,
                "user_id": best_user_id,
                "confidence": float(best_score)
            }

        return {
            "matched": False,
            "user_id": None,
            "confidence": float(best_score)
        }