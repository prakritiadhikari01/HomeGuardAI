#app\services\face_embedding_service.py
from typing import Optional

import numpy as np
from insightface.app import FaceAnalysis


class FaceEmbeddingService:
    """
    Central service responsible for:
    - Loading InsightFace
    - Detecting faces
    - Returning the primary face
    - Extracting normalized embeddings
    """

    def __init__(self):

        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=0)

    def detect_faces(self, frame):

        return self.face_app.get(frame)

    def get_primary_face(self, frame):

        faces = self.detect_faces(frame)

        if not faces:
            return None

        return faces[0]

    def normalize_embedding(self, face) -> Optional[np.ndarray]:

        if face is None:
            return None

        embedding = face.embedding

        if embedding is None:
            return None

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return None

        return embedding / norm
    
    def get_face_data(self, frame):
        """
        Returns the primary detected face.
        """
        
        face = self.get_primary_face(frame)
        embedding=self.normalize_embedding(face)

        if face is None:
            return None

        if embedding is None:
            return None
        
        return {

            "embedding": embedding,

            "confidence": float(face.det_score),

            "bbox": face.bbox.tolist(),

        }