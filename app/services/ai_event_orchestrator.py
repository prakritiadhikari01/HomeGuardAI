# app/services/ai_event_orchestrator.py

from app.services.face_recognition_service import (
    FaceRecognitionService
)

from app.services.django_client import (
    DjangoClient
)


class AIEventOrchestrator:

    @staticmethod
    def process_frame(
        frame,
        device_id,
        image_url=None
    ):

        result = (
            FaceRecognitionService()
            .recognize(frame)
        )

        payload = {
            "device_id": str(device_id),
            "person_type": "UNKNOWN",
            "confidence_score": 0.0,
            "image_url": image_url
        }

        if (
            result
            and result["status"] == "known"
        ):

            payload.update(
                {
                    "person_type": "KNOWN",
                    "member_id": result["member_id"],
                    "face_profile_id": result["face_profile_id"],
                    "confidence_score": result[
                        "confidence_score"
                    ]
                }
            )

        return (
            DjangoClient.send_detection_event(
                payload
            )
        )