#app/services/ai_event_orchestrator
from datetime import datetime, timezone

from app.services.face_recognition_service import (
    FaceRecognitionService
)

from app.services.django_client import (
    DjangoClient
)

from app.services.evidence_service import (
    EvidenceService
)

from app.services.cloudinary_service import (
    CloudinaryService
)


class AIEventOrchestrator:

    @staticmethod
    def process_frame(
        frame,
        device_id,
        home_id,
        location,
    ):

        try:

            # ----------------------------
            # SAVE SNAPSHOT
            # ----------------------------

            local_snapshot = (
                EvidenceService.save_frame(
                    frame
                )
            )

            snapshot_url = (
                CloudinaryService.upload_image(
                    local_snapshot
                )
            )

            # ----------------------------
            # FACE RECOGNITION
            # ----------------------------

            result = (
                FaceRecognitionService()
                .recognize(frame)
            )

            # ----------------------------
            # DEFAULT EVENT
            # ----------------------------

            payload = {

                "home_id": str(home_id),

                "device_id": str(device_id),

                "event_type":
                    "PERSON_DETECTED",

                "person_type":
                    "UNKNOWN",

                "person_label":
                    None,

                "member_id":
                    None,

                "face_profile_id":
                    None,

                "confidence_score":
                    0.0,

                "camera_location":
                    location,

                "image_url":
                    snapshot_url,

                "snapshot_url":
                    snapshot_url,

                "event_summary":
                    f"Unknown person detected at {location}",

                "duration_seconds":
                    0,

                "metadata": {
                    "recognition_engine":
                        "InsightFace",

                    "camera_location":
                        location,

                    "snapshot_available":
                        snapshot_url is not None
                },

                "timestamp": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )
            }

            # ----------------------------
            # KNOWN PERSON
            # ----------------------------

            if (
                result
                and result.get("status")
                == "known"
            ):

                payload.update({

                    "person_type":
                        "KNOWN",

                    "person_label":
                        result.get(
                            "person_label"
                        ),

                    "member_id":
                        result.get(
                            "member_id"
                        ),

                    "face_profile_id":
                        result.get(
                            "face_profile_id"
                        ),

                    "confidence_score":
                        result.get(
                            "confidence_score",
                            0.0
                        ),

                    "event_summary":
                        (
                            f"{result.get('person_label')} "
                            f"detected at "
                            f"{location}"
                        )
                })

            print(
                f"[AI EVENT] "
                f"{payload['person_type']} | "
                f"{payload['person_label']} | "
                f"{location}"
            )

            response = (
                DjangoClient.send_detection_event(
                    payload
                )
            )

            return response

        except Exception as e:

            print(
                "AI Event Error:",
                e
            )

            return {
                "status": "error",
                "message": str(e)
            }