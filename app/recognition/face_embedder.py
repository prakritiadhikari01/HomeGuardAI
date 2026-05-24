# app/recognition/face_embedder.py

import numpy as np
from insightface.app import FaceAnalysis


class FaceEmbedder:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(320, 320)
        )

    def get_embedding(self, face_image):
        try:
            faces = self.app.get(face_image)

            if len(faces) == 0:
                return None

            embedding = faces[0].embedding

            embedding = embedding.astype(np.float32)

            embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            print("Embedding Error:", e)
            return None