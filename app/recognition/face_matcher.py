import numpy as np


class FaceMatcher:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def cosine_similarity(self, emb1, emb2):
        if emb1 is None or emb2 is None:
            return -1

        emb1 = np.asarray(emb1, dtype=np.float32)
        emb2 = np.asarray(emb2, dtype=np.float32)

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return -1

        emb1 = emb1 / norm1
        emb2 = emb2 / norm2

        return float(np.dot(emb1, emb2))

    def find_best_match(self, live_embedding, known_faces):
        if live_embedding is None or not known_faces:
            return {
                "matched": False,
                "user": None,
                "score": 0
            }

        best_score = -1
        best_match = None

        for face in known_faces:
            score = self.cosine_similarity(
                live_embedding,
                face.get("embedding")
            )

            if score > best_score:
                best_score = score
                best_match = face

        if best_match is None:
            return {
                "matched": False,
                "user": None,
                "score": 0
            }

        return {
            "matched": best_score >= self.threshold,
            "user": best_match,
            "score": float(best_score)
        }