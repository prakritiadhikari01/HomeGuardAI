# app/controllers/enrollment_controller.py

from dataclasses import dataclass
from uuid import uuid4

from app.services.django_client import DjangoClient
from app.services.face_enrollment_service import FaceEnrollmentService


# ==========================================================
# AI Enrollment Session
# ==========================================================

@dataclass
class AIEnrollmentSession:
    """
    Represents one active AI enrollment session.

    The AI Engine only knows:
    - Django EnrollmentSession ID
    - FaceEnrollmentService

    It never knows anything about
    HomeMember, FaceProfile, Invitation, etc.
    """

    django_session_id: str
    service: FaceEnrollmentService


# ==========================================================
# Enrollment Controller
# ==========================================================

class EnrollmentController:
    """
    Coordinates a complete AI face enrollment session.

    Flow

        Browser
            ↓
        Django EnrollmentSession
            ↓
        AI Session (memory)
            ↓
        Frame Processing
            ↓
        Embedding Collection
            ↓
        Django FaceProfile
    """

    def __init__(self):

        self.sessions: dict[str, AIEnrollmentSession] = {}

    # ======================================================
    # START
    # ======================================================

    def start_enrollment(
        self,
        django_session_id: str,
    ):

        print("\n" + "=" * 70)
        print("[START] Enrollment Request Received")
        print(f"[DJANGO SESSION] {django_session_id}")

        ai_session_id = str(uuid4())

        self.sessions[ai_session_id] = AIEnrollmentSession(
            django_session_id=django_session_id,
            service=FaceEnrollmentService(),
        )

        print(f"[AI SESSION CREATED] {ai_session_id}")
        print(f"[ACTIVE SESSIONS] {len(self.sessions)}")
        print("=" * 70)

        return {
            "success": True,
            "session_id": ai_session_id,
            "current_step": FaceEnrollmentService.STEPS[0],
        }

    # ======================================================
    # PROCESS FRAME
    # ======================================================

    def process_frame(
        self,
        session_id: str,
        frame,
    ):

        session = self.sessions.get(session_id)

        if session is None:

            print(f"[ERROR] Unknown AI Session: {session_id}")

            return {
                "success": False,
                "message": "Enrollment session not found.",
            }

        result = session.service.process_frame(frame)

        if not result["success"]:

            print(
                f"[WAITING] "
                f"{result.get('message')}"
            )

            return result

        print(
            f"[FRAME] "
            f"Step={result.get('current_step')} | "
            f"Progress={result.get('progress')}% | "
            f"Frames={result.get('step_progress')}/"
            f"{result.get('frames_required')}"
        )

        if result.get("completed_step"):

            print(
                f"[POSE COMPLETED] "
                f"{result['completed_step']}"
            )

        if result["finished"]:

            print("\n[INFO] Enrollment completed.")
            print("[INFO] Saving embeddings to Django...")

            return self.finish_enrollment(session_id)

        return result

    # ======================================================
    # FINISH
    # ======================================================

    def finish_enrollment(
        self,
        session_id: str,
    ):

        session = self.sessions.get(session_id)

        if session is None:

            print("[ERROR] Session missing during finish.")

            return {
                "success": False,
                "message": "Enrollment session not found.",
            }

        payload = {

            "session_id": session.django_session_id,

            "embeddings": session.service.get_embeddings(),

        }

        print("\n" + "=" * 70)
        print("[FINISH] Uploading embeddings to Django")
        print(f"[DJANGO SESSION] {session.django_session_id}")

        django_response = DjangoClient.save_face_profile(
            payload
        )

        print("[DJANGO RESPONSE]")
        print(django_response)

        success = False

        if isinstance(django_response, dict):

            success = (
                django_response.get("success") is True
                or "face_profile_id" in django_response
            )

        if not success:

            print("[ERROR] Django rejected enrollment.")
            print("[INFO] Keeping AI session alive for retry.")
            print("=" * 70)

            return {

                "success": False,

                "finished": False,

                "message": "Failed to save enrollment.",

                "django_response": django_response,

            }

        del self.sessions[session_id]

        print("[SUCCESS] Enrollment completed successfully.")
        print(f"[ACTIVE SESSIONS] {len(self.sessions)}")
        print("=" * 70)

        return {

            "success": True,

            "finished": True,

            "django_response": django_response,

        }

    # ======================================================
    # CANCEL
    # ======================================================

    def cancel_enrollment(
        self,
        session_id: str,
    ):

        print("\n" + "=" * 70)
        print(f"[CANCEL] AI Session: {session_id}")

        if session_id in self.sessions:

            del self.sessions[session_id]

            print("[SUCCESS] Session removed.")

        else:

            print("[INFO] Session already removed.")

        print(f"[ACTIVE SESSIONS] {len(self.sessions)}")
        print("=" * 70)

        return {

            "success": True,

            "message": "Enrollment cancelled.",

        }