# app/services/ai_event_orchestrator.py
from datetime import datetime, timezone

from app.services.face_recognition_service import FaceRecognitionService
from app.services.django_client import DjangoClient
from app.services.evidence_service import EvidenceService
from app.services.cloudinary_service import CloudinaryService


class AIEventOrchestrator:
    # One shared instance across all camera threads — avoids reloading
    # InsightFace, and avoids re-instantiating FaceRecognitionService
    # (and therefore FaceEmbeddingService/ModelRegistry) on every frame.
    _face_recognition_service = FaceRecognitionService()

    @staticmethod
    def process_frame(frame, device_id, home_id, location):
        try:
            result = AIEventOrchestrator._face_recognition_service.recognize(frame)

            # By the time we're here, CameraRecognitionService has already
            # confirmed motion + a person object — so this frame is worth
            # storing evidence for, per your "meaningful evidence only" principle.
            local_snapshot = EvidenceService.save_frame(frame)
            snapshot_url = CloudinaryService.upload_image(local_snapshot)

            is_known = result.get("status") == "known"

            payload = {
                "home_id": str(home_id),
                "device_id": str(device_id),
                "event_type": "PERSON_DETECTED",
                "person_type": "KNOWN" if is_known else "UNKNOWN",
                "person_label": result.get("person_label") if is_known else None,
                "member_id": result.get("member_id") if is_known else None,
                "face_profile_id": result.get("face_profile_id") if is_known else None,
                "confidence_score": result.get("confidence_score", 0.0) if is_known else 0.0,
                "camera_location": location,
                "image_url": snapshot_url,
                "snapshot_url": snapshot_url,
                "event_summary": (
                    f"{result.get('person_label')} detected at {location}"
                    if is_known else
                    f"Unknown person detected at {location}"
                ),
                "duration_seconds": 0,
                "metadata": {
                    "recognition_engine": "InsightFace",
                    "camera_location": location,
                    "snapshot_available": snapshot_url is not None,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            print(f"[AI EVENT] {payload['person_type']} | {payload['person_label']} | {location}")

            return DjangoClient.send_detection_event(payload)

        except Exception as e:
            print("AI Event Error:", e)
            return {"status": "error", "message": str(e)}