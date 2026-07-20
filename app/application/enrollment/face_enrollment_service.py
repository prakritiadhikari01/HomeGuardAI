from app.infrastructure.ai.insightface_service import InsightFaceService


class FaceEnrollmentService:
    """Handles one complete face enrollment session: processes webcam
    frames, guides pose progression, collects embeddings, keeps only
    the best ones per pose, reports progress.

    FIX: previously used a separate FaceEmbeddingService that duplicated
    InsightFaceService's get_face_data() almost exactly (both wrapped
    ModelRegistry). Now uses InsightFaceService directly — one face
    model wrapper for both live recognition and enrollment.

    FIX (this revision): STEPS use LOOK_* naming internally to match
    frontend pose-progression labels, but Django's FaceProfile.embeddings
    schema and CompleteEnrollmentSerializer expect bare pose names
    (CENTER/LEFT/RIGHT/UP/DOWN). POSE_KEY_MAP translates at the boundary
    so internal step names never leak into the Django payload.
    """

    STEPS = ["LOOK_CENTER", "LOOK_LEFT", "LOOK_RIGHT", "LOOK_UP", "LOOK_DOWN"]
    FRAMES_PER_STEP = 10
    BEST_EMBEDDINGS_PER_POSE = 3
    MIN_FACE_CONFIDENCE = 0.60

    # Internal step name -> Django/FaceProfile pose key
    POSE_KEY_MAP = {
        "LOOK_CENTER": "CENTER",
        "LOOK_LEFT": "LEFT",
        "LOOK_RIGHT": "RIGHT",
        "LOOK_UP": "UP",
        "LOOK_DOWN": "DOWN",
    }

    def __init__(self):
        self.face_service = InsightFaceService()
        self.current_step = 0
        self.frame_count = 0
        self.pose_embeddings = {step: {"embeddings": []} for step in self.STEPS}

    def process_frame(self, frame) -> dict:
        if self.current_step >= len(self.STEPS):
            return {
                "success": True,
                "finished": True,
                "progress": 100,
                "message": "Enrollment already completed.",
            }

        face = self.face_service.get_face_data(frame)

        if face is None:
            return self._waiting_response("Face not detected.")

        if face["confidence"] < self.MIN_FACE_CONFIDENCE:
            return self._waiting_response("Face confidence too low.")

        current_pose = self.STEPS[self.current_step]
        self.pose_embeddings[current_pose]["embeddings"].append(
            {
                "embedding": face["embedding"].tolist(),
                "confidence": face["confidence"],
                "quality": {
                    "pose": current_pose,
                    "blur": None,
                    "brightness": None,
                    "yaw": None,
                    "pitch": None,
                },
            }
        )

        self.frame_count += 1
        completed_step = None

        if self.frame_count >= self.FRAMES_PER_STEP:
            completed_step = current_pose
            self.current_step += 1
            self.frame_count = 0

        finished = self.current_step >= len(self.STEPS)

        return {
            "success": True,
            "finished": finished,
            "completed_step": completed_step,
            "current_step": None if finished else self.STEPS[self.current_step],
            "progress": self._calculate_progress(),
            "step_progress": self.frame_count,
            "frames_required": self.FRAMES_PER_STEP,
            "message": (
                "Enrollment completed."
                if finished
                else f"Collecting {self.STEPS[self.current_step]}"
            ),
        }

    def get_embeddings(self) -> dict:
        return self._select_best_embeddings()

    def reset(self) -> None:
        self.current_step = 0
        self.frame_count = 0
        self.pose_embeddings = {step: {"embeddings": []} for step in self.STEPS}

    def _waiting_response(self, message: str) -> dict:
        return {
            "success": False,
            "finished": False,
            "message": message,
            "current_step": self.STEPS[self.current_step],
            "progress": self._calculate_progress(),
            "step_progress": self.frame_count,
            "frames_required": self.FRAMES_PER_STEP,
        }

    def _calculate_progress(self) -> int:
        if self.current_step >= len(self.STEPS):
            return 100
        return int((self.current_step / len(self.STEPS)) * 100)

    def _select_best_embeddings(self) -> dict:
        """Returns {'CENTER': [[512 floats], [512 floats], ...], ...} —
        matches FaceProfile.embeddings schema exactly: bare pose keys,
        each mapping to a list of embedding vectors (not a dict with
        confidence/quality metadata — that's internal-only bookkeeping
        used to pick the best frames)."""
        result = {}
        for pose, data in self.pose_embeddings.items():
            ordered = sorted(data["embeddings"], key=lambda x: x["confidence"], reverse=True)
            best = ordered[: self.BEST_EMBEDDINGS_PER_POSE]
            django_key = self.POSE_KEY_MAP[pose]
            result[django_key] = [item["embedding"] for item in best]
        return result