# app/services/face_enrollment_service.py

from app.services.face_embedding_service import (
    FaceEmbeddingService,
)


class FaceEnrollmentService:
    """
    Handles a complete face enrollment session.

    Responsibilities:
    - Process webcam frames
    - Guide pose progression
    - Collect embeddings
    - Keep only the best embeddings
    - Report enrollment progress
    """

    STEPS = [
        "LOOK_CENTER",
        "LOOK_LEFT",
        "LOOK_RIGHT",
        "LOOK_UP",
        "LOOK_DOWN",
    ]

    FRAMES_PER_STEP = 10

    BEST_EMBEDDINGS_PER_POSE = 3

    MIN_FACE_CONFIDENCE = 0.60

    def __init__(self):

        self.embedding_service = FaceEmbeddingService()

        self.current_step = 0

        self.frame_count = 0

        self.pose_embeddings = {

            step: {

                "embeddings": []

            }

            for step in self.STEPS

        }

    def process_frame(self, frame):
        """
        Process a single webcam frame.

        Returns enrollment progress.
        """

        if self.current_step >= len(self.STEPS):

            return {

                "success": True,

                "finished": True,

                "progress": 100,

                "message": "Enrollment already completed."

            }

        face = self.embedding_service.get_face_data(frame)

        if face is None:

            return {

                "success": False,

                "finished": False,

                "message": "Face not detected.",

                "current_step": self.STEPS[self.current_step],

                "progress": self._calculate_progress(),

                "step_progress": self.frame_count,

                "frames_required": self.FRAMES_PER_STEP,

            }

        if face["confidence"] < self.MIN_FACE_CONFIDENCE:

            return {

                "success": False,

                "finished": False,

                "message": "Face confidence too low.",

                "current_step": self.STEPS[self.current_step],

                "progress": self._calculate_progress(),

                "step_progress": self.frame_count,

                "frames_required": self.FRAMES_PER_STEP,

            }

        current_pose = self.STEPS[self.current_step]

        self.pose_embeddings[current_pose]["embeddings"].append({

            "embedding": face["embedding"].tolist(),

            "confidence": face["confidence"],

            "quality": {

                "pose": current_pose,

                "blur": None,

                "brightness": None,

                "yaw": None,

                "pitch": None,

            }

        })

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

            "current_step": (

                None

                if finished

                else self.STEPS[self.current_step]

            ),

            "progress": self._calculate_progress(),

            "step_progress": self.frame_count,

            "frames_required": self.FRAMES_PER_STEP,

            "message": (

                "Enrollment completed."

                if finished

                else f"Collecting {self.STEPS[self.current_step]}"

            )

        }

    def get_embeddings(self):
        """
        Returns only the highest-quality embeddings
        for each pose.
        """

        return self._select_best_embeddings()

    def reset(self):

        self.current_step = 0

        self.frame_count = 0

        self.pose_embeddings = {

            step: {

                "embeddings": []

            }

            for step in self.STEPS

        }

    def _calculate_progress(self):

        if self.current_step >= len(self.STEPS):

            return 100

        return int(

            (self.current_step / len(self.STEPS)) * 100

        )

    def _select_best_embeddings(self):
        """
        Keep only the best embeddings from each pose.
        """

        result = {}

        for pose, data in self.pose_embeddings.items():

            ordered = sorted(

                data["embeddings"],

                key=lambda x: x["confidence"],

                reverse=True,

            )

            result[pose] = {

                "embeddings": ordered[
                    : self.BEST_EMBEDDINGS_PER_POSE
                ]

            }

        return result