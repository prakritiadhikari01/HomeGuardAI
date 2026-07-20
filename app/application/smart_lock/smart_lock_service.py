from app.infrastructure.ai.face_recognition_service import FaceRecognitionService
from app.infrastructure.ai.insightface_service import InsightFaceService
from app.infrastructure.vision.camera_capture import CameraCapture


class SmartLockVerificationService:

    MIN_FACE_CONFIDENCE = 0.60

    @classmethod
    def verify(cls, payload):

        stream_url = payload["stream_url"]

        capture = CameraCapture(stream_url)

        insight = InsightFaceService()
        recognition = FaceRecognitionService()

        best_face = None
        best_score = 0.0

        # Check a few frames (~1 second)
        for _ in range(15):

            frame = capture.read()

            if frame is None:
                continue

            face = insight.detect_face(frame)

            if face is None:
                continue

            score = float(face.det_score)

            if score > best_score:
                best_face = face
                best_score = score

        capture.release()

        if best_face is None:
            return {
                "success": False,
                "reason": "no_face",
            }

        if best_score < cls.MIN_FACE_CONFIDENCE:
            return {
                "success": False,
                "reason": "low_quality",
            }

        embedding = insight.extract_embedding(best_face)

        if embedding is None:
            return {
                "success": False,
                "reason": "embedding_failed",
            }

        result = recognition.match_embedding(embedding)

        if result["status"] == "unknown":
            return {
                "success": True,
                "authorized": False,
                "reason": "unknown_person",
            }

        return {
            "success": True,
            "authorized": True,
            "member_id": result["member_id"],
            "member_name": result["person_label"],
            "confidence": result["confidence_score"],
        }