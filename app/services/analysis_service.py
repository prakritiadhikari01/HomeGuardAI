#app\services\analysis_service.py
from app.detection.face_detector import FaceDetector
from app.recognition.face_matcher import FaceMatcher


class AnalysisService:

    @staticmethod
    def analyze_image(image_base64: str):

        faces = FaceDetector.detect_faces(image_base64)

        if not faces:
            return {
                "face_match": None,
                "confidence": 0.0,
                "type": "UNKNOWN"
            }

        face = faces[0]

        embedding = FaceMatcher.extract_embedding(
            face
        )

        match_result = FaceMatcher.match_face(
            embedding
        )

        if match_result["matched"]:

            return {
                "face_match": match_result["user_id"],
                "confidence": match_result["confidence"],
                "type": "KNOWN"
            }

        return {
            "face_match": None,
            "confidence": match_result["confidence"],
            "type": "UNKNOWN"
        }