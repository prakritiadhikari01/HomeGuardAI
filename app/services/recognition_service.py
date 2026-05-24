# app/services/recognition_service.py

from app.detection.face_detector import FaceDetector
from app.recognition.face_embedder import FaceEmbedder
from app.recognition.face_matcher import FaceMatcher
from app.recognition.known_faces_store import KnownFacesStore


class RecognitionService:
    def __init__(self):
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()

        self.matcher = FaceMatcher(
            threshold=0.65
        )

        self.store = KnownFacesStore()

        self.store.load_faces_from_django()

    def recognize_frame(self, frame):
        faces = self.detector.detect_faces(frame)
        if not faces:
            return []

        results = []
        for face in faces:
            embedding = self.embedder.get_embedding(face)

            if embedding is None:
                continue

            match = self.matcher.find_best_match(
                embedding,
                self.store.get_all_faces()
            )

            results.append(match)

        return results