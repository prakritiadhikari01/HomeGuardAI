from dataclasses import dataclass
from uuid import uuid4

from app.application.enrollment.face_enrollment_service import FaceEnrollmentService
from app.infrastructure.api.django_client import DjangoClient

# Only import paths changed from the original — logic is identical, just
# pointed at application/enrollment + infrastructure/api instead of the
# now-deleted app/services/*.


@dataclass
class AIEnrollmentSession:
    """One active AI enrollment session. The AI Engine only knows the
    Django EnrollmentSession id and its own FaceEnrollmentService — never
    HomeMember, FaceProfile, Invitation, etc."""

    django_session_id: str
    service: FaceEnrollmentService


class EnrollmentController:
    """Coordinates a complete AI face enrollment session.

    Browser -> Django EnrollmentSession -> AI Session (memory) ->
    Frame Processing -> Embedding Collection -> Django FaceProfile"""

    def __init__(self):
        self.sessions: dict[str, AIEnrollmentSession] = {}

    def start_enrollment(self, django_session_id: str) -> dict:
        print("=" * 70)
        print(f"[START] Enrollment Request Received | Django session: {django_session_id}")

        ai_session_id = str(uuid4())
        self.sessions[ai_session_id] = AIEnrollmentSession(
            django_session_id=django_session_id, service=FaceEnrollmentService()
        )

        print(f"[AI SESSION CREATED] {ai_session_id} | Active sessions: {len(self.sessions)}")
        print("=" * 70)

        return {
            "success": True,
            "session_id": ai_session_id,
            "current_step": FaceEnrollmentService.STEPS[0],
        }

    def process_frame(self, session_id: str, frame) -> dict:
        session = self.sessions.get(session_id)
        if session is None:
            print(f"[ERROR] Unknown AI Session: {session_id}")
            return {"success": False, "message": "Enrollment session not found."}

        result = session.service.process_frame(frame)

        if not result["success"]:
            print(f"[WAITING] {result.get('message')}")
            return result

        print(
            f"[FRAME] Step={result.get('current_step')} | "
            f"Progress={result.get('progress')}% | "
            f"Frames={result.get('step_progress')}/{result.get('frames_required')}"
        )
        if result.get("completed_step"):
            print(f"[POSE COMPLETED] {result['completed_step']}")

        if result["finished"]:
            print("[INFO] Enrollment completed. Saving embeddings to Django...")
            return self.finish_enrollment(session_id)

        return result

    def finish_enrollment(self, session_id: str) -> dict:
        session = self.sessions.get(session_id)
        if session is None:
            print("[ERROR] Session missing during finish.")
            return {"success": False, "message": "Enrollment session not found."}

        payload = {
            "session_id": session.django_session_id,
            "embeddings": session.service.get_embeddings(),
        }

        print("=" * 70)
        print(f"[FINISH] Uploading embeddings to Django | session: {session.django_session_id}")

        django_response = DjangoClient.save_face_profile(payload)
        print(f"[DJANGO RESPONSE] {django_response}")

        success = isinstance(django_response, dict) and (
            django_response.get("success") is True or "face_profile_id" in django_response
        )

        if not success:
            print("[ERROR] Django rejected enrollment. Keeping AI session alive for retry.")
            print("=" * 70)
            return {
                "success": False,
                "finished": False,
                "message": "Failed to save enrollment.",
                "django_response": django_response,
            }

        del self.sessions[session_id]
        print(f"[SUCCESS] Enrollment completed. Active sessions: {len(self.sessions)}")
        print("=" * 70)

        return {"success": True, "finished": True, "django_response": django_response}

    def cancel_enrollment(self, session_id: str) -> dict:
        print(f"[CANCEL] AI Session: {session_id}")
        if session_id in self.sessions:
            del self.sessions[session_id]
            print("[SUCCESS] Session removed.")
        else:
            print("[INFO] Session already removed.")
        return {"success": True, "message": "Enrollment cancelled."}
