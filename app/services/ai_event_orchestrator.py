# app/services/ai_event_orchestrator.py
from datetime import datetime, timezone
import uuid

from app.services.face_recognition_service import FaceRecognitionService
from app.services.django_client import DjangoClient
from app.services.analyzer import analyze_frame
from app.services.embedderai import get_embedding, build_event_text
from app.services.vector_storeai import store_event


class AIEventOrchestrator:

    @staticmethod
    def process_frame(frame, device_id, home_id, location, image_url=None):

        # 1. Face recognition (existing)
        result = FaceRecognitionService().recognize(frame)

        timestamp = datetime.now(timezone.utc).isoformat()

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
            "timestamp": timestamp,
        }

        if result and result.get("status") == "known":
            payload.update({
                "person_type": "KNOWN",
                "person_label": result.get("person_label"),
                "member_id": result.get("member_id"),
                "face_profile_id": result.get("face_profile_id"),
                "confidence_score": result.get("confidence_score", 0.0),
            })

        print(f"Detection from {location}: {payload['person_label']}")

        # 2. NEW: BLIP scene analysis
        try:
            scene = analyze_frame(frame)
        except Exception as e:
            print(f"[BLIP] Analysis failed: {e}")
            scene = {
                "general_description": "",
                "clothing_description": "",
                "action_description": "",
                "person_present": False,
            }

        # 3. NEW: Build metadata + store in ChromaDB
        event_metadata = {
            "event_id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "camera_id": str(device_id),
            "location": location,
            "person_type": payload["person_type"],
            "person_label": str(payload["person_label"] or "unknown"),
            "confidence_score": payload["confidence_score"],
            "general_description": scene["general_description"],
            "clothing_description": scene["clothing_description"],
            "action_description": scene["action_description"],
            "person_present": scene["person_present"],
        }

        try:
            event_text = build_event_text(event_metadata)
            embedding = get_embedding(event_text)
            store_event(event_metadata["event_id"], event_text, embedding, event_metadata)
        except Exception as e:
            print(f"[ChromaDB] Storage failed: {e}")

        # 4. Send to Django (existing)
        return DjangoClient.send_detection_event(payload)