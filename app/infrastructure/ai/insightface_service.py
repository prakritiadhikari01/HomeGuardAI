from typing import Optional

import numpy as np

from app.core.model_registry import ModelRegistry


def crop_region(frame, bbox):
    """Crop a bbox out of a frame, clamped to frame bounds. Lives next
    to the model that consumes the crop, rather than in
    RecognitionProcessor — "which pixels does the face model need" is a
    face-service concern, not a recognition decision."""
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    return frame[y1:y2, x1:x2]


class InsightFaceService:
    """Split into independent steps so callers can stop early:
    RecognitionProcessor needs quality scoring (detect_face_in_region
    alone) far more often than a full embedding; EnrichmentProcessor's
    VLM pass needs the raw crop, not an embedding at all.

    Also used directly by FaceEnrollmentService — one face model, one
    place it's loaded (ModelRegistry singleton), for both live camera
    recognition and the enrollment webcam flow."""

    def __init__(self):
        self.registry = ModelRegistry()  # shared singleton, does not reload models

    def detect_faces(self, frame):
        return self.registry.get(frame)

    def detect_face(self, frame):
        """Highest det_score face in `frame`, or None."""
        faces = self.detect_faces(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: float(f.det_score))

    def detect_face_in_region(self, frame, bbox):
        """Crops bbox out of frame, detects within just that region.
        Returns (face, crop) — caller often needs to keep the crop for
        best-face tracking, avoiding a second crop later."""
        crop = crop_region(frame, bbox)
        if crop is None or crop.size == 0:
            return None, None
        return self.detect_face(crop), crop

    def extract_embedding(self, face) -> Optional[np.ndarray]:
        """face object -> L2-normalized embedding, or None."""
        if face is None or face.embedding is None:
            return None
        embedding = np.asarray(face.embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm != 0 else None

    def get_face_data(self, frame) -> Optional[dict]:
        """Convenience wrapper for callers that just want one dict
        (FaceEnrollmentService, the /extract-embedding debug route).
        Pipeline code should prefer detect_face_in_region() +
        extract_embedding() directly to avoid detecting twice."""
        face = self.detect_face(frame)
        if face is None:
            return None
        embedding = self.extract_embedding(face)
        if embedding is None:
            return None
        return {
            "embedding": embedding,
            "confidence": float(face.det_score),
            "bbox": face.bbox.tolist(),
        }
