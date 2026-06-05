# app/services/ai_event_orchestrator.py

from datetime import datetime

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
        home_id,
        location,
        image_url=None
    ):

        result = (
            FaceRecognitionService()
            .recognize(frame)
        )

        payload = {
            "home_id": str(home_id),
            "device_id": str(device_id),

            "event_type": "PERSON_DETECTED",

            "person_type": "UNKNOWN",

            "person_label": None,

            "member_id": None,
            "face_profile_id": None,

            "confidence_score": 0.0,

            "camera_location": location,

            "image_url": image_url,

            "timestamp": (
                datetime.utcnow()
                .isoformat()
            )
        }

        if (
            result
            and result.get("status") == "known"
        ):

            payload.update(
                {
                    "person_type": "KNOWN",

                    "person_label": result.get(
                        "person_label"
                    ),

                    "member_id": result.get(
                        "member_id"
                    ),

                    "face_profile_id": result.get(
                        "face_profile_id"
                    ),

                    "confidence_score": result.get(
                        "confidence_score",
                        0.0
                    )
                }
            )

        return (
            DjangoClient.send_detection_event(
                payload
            )
        )