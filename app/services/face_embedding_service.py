# app/services/face_embedding_service.py
from typing import Optional
import numpy as np
from app.core.model_registry import ModelRegistry


class FaceEmbeddingService:
    def __init__(self):
        self.registry = ModelRegistry()  # shared singleton, does not reload models

    def detect_faces(self, frame):
        return self.registry.get(frame)

    def get_primary_face(self, frame):
        faces = self.detect_faces(frame)
        return faces[0] if faces else None

    def normalize_embedding(self, face) -> Optional[np.ndarray]:
        if face is None or face.embedding is None:
            return None
        embedding = np.asarray(face.embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm != 0 else None

    def get_face_data(self, frame):
        face = self.get_primary_face(frame)
        if face is None:
            return None
        embedding = self.normalize_embedding(face)
        if embedding is None:
            return None
        return {
            "embedding": embedding,
            "confidence": float(face.det_score),
            "bbox": face.bbox.tolist(),
        }