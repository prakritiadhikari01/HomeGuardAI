# app/services/recognition_service.py

import numpy as np

from app.detection.face_detector import FaceDetector
from app.recognition.face_matcher import FaceMatcher
from app.recognition.known_faces_store import KnownFacesStore


class RecognitionService:
    def __init__(self):
        self.detector = FaceDetector()
        self.matcher = FaceMatcher(threshold=0.65)
        self.store = KnownFacesStore()

        self.store.load_faces_from_django()
        self.known_faces = self.store.get_all_faces()


    def recognize_frame(self, frame):
        faces = self.detector.detect_faces(frame)

        if not faces:
            return []

        results = []

        for face in faces:
            emb = np.asarray(face.get("embedding"), dtype=np.float32)

            norm = np.linalg.norm(emb)
            if norm != 0:
                emb = emb / norm

            if emb is None:
                continue

            match = self.matcher.find_best_match(
                emb,
                self.known_faces,
            )
            if match["matched"]:
                event_type = "KNOWN_FACE"
            else:
                event_type = "UNKNOWN_FACE"

            results.append({
                "bbox": face.get("bbox"),
                "embedding": emb.tolist() if hasattr(emb, "tolist") else emb,
                "match": match,
                "det_score": face.get("det_score", 0.0),
                "is_unknown": not match["matched"]
            })

        return results