# app/recognition/face_embedder.py

import numpy as np

class FaceEmbedder:
    def get_embedding(self, face):
        try:
            if face is None:
                return None

            emb = face.embedding  # already computed by InsightFace

            if emb is None:
                return None

            emb = np.asarray(emb, dtype=np.float32)

            norm = np.linalg.norm(emb)
            if norm == 0:
                return None

            return emb / norm

        except Exception as e:
            print("[Embedding Error]", e)
            return None