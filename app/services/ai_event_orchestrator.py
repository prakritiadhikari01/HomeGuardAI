# app/services/ai_event_orchestrator.py

from app.services.django_client import DjangoClient


class AIEventOrchestrator:

    CONFIDENCE_THRESHOLD = 0.65

    @staticmethod
    def process_recognition_result(
        *,
        device_id,
        recognition_result
    ):

        match = recognition_result.get("match", {})

        matched = match.get("matched", False)
        score = match.get("score", 0.0)

        if matched and score >= AIEventOrchestrator.CONFIDENCE_THRESHOLD:

            payload = {
                "device_id": device_id,
                "event_type": "KNOWN_FACE",
                "confidence_score": score,
                "member_id": match["user"]["member_id"],
                "face_profile_id": match["user"]["face_profile_id"]
            }

        else:

            payload = {
                "device_id": device_id,
                "event_type": "UNKNOWN_FACE",
                "confidence_score": score,
            }

        response = DjangoClient.send_detection_event(
            payload
        )

        return response