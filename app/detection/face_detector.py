# app/detection/face_detector.py
from app.core.model_registry import ModelRegistry


class FaceDetector:
    """
    Used by the /analyze endpoint (not the live camera pipeline).
    Previously created its own separate FaceAnalysis instance — now
    reuses the same singleton as the camera path, so there's only
    ever one InsightFace model loaded in the whole process.
    """

    def __init__(self):
        self.registry = ModelRegistry()

    def detect_faces(self, frame):
        if frame is None:
            return []

        faces = self.registry.get(frame)

        return [
            {
                "bbox": face.bbox.astype(int).tolist(),
                "embedding": face.embedding,
                "det_score": float(face.det_score) if hasattr(face, "det_score") else 0.0,
            }
            for face in faces
        ]