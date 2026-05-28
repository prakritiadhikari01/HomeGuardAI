import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.services.django_client import DjangoClient


class FaceEnrollmentService:

    def __init__(self):

        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=0)

        self.embeddings = []

        self.steps = [
            "Look Center",
            "Turn Left",
            "Turn Right",
            "Look Up",
            "Look Down"
        ]

        self.current_step = 0
        self.frames_per_step = 10
        self.frame_count = 0

    # -----------------------------
    # GET FACE EMBEDDING
    # -----------------------------
    def get_embedding(self, frame):

        faces = self.face_app.get(frame)

        if len(faces) == 0:
            return None

        face = faces[0]

        return face.embedding

    # -----------------------------
    # MAIN ENROLLMENT LOOP
    # -----------------------------
    def enroll_face(self,home_member_id,label_name):

        cap = cv2.VideoCapture(0)

        print("Starting Face Enrollment...")

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            # -------------------------
            # DRAW UI TEXT
            # -------------------------
            instruction = self.steps[self.current_step]

            cv2.putText(
                frame,
                f"Step: {instruction}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Stay in center box",
                (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # -------------------------
            # FACE DETECTION
            # -------------------------
            embedding = self.get_embedding(frame)

            if embedding is not None:

                self.embeddings.append(embedding)
                self.frame_count += 1

            # -------------------------
            # STEP CONTROL
            # -------------------------
            if self.frame_count >= self.frames_per_step:

                self.current_step += 1
                self.frame_count = 0

                print(f"Completed step: {instruction}")

                if self.current_step >= len(self.steps):
                    break

            cv2.imshow("Face Enrollment", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return self.final_embedding(home_member_id, label_name)

    # -----------------------------
    # FINAL EMBEDDING (AVERAGE)
    # -----------------------------
    def final_embedding(self,home_member_id,label_name):

        if len(self.embeddings) == 0:
            return None

        embeddings_array = np.array(self.embeddings)

        avg_embedding = np.mean(embeddings_array, axis=0)

        payload = {
            "home_id": home_member_id,
            "label_name": label_name,
            "embedding": avg_embedding.tolist()
        }

        django_response = DjangoClient.save_face_profile(
            payload
        )

        print("DJANGO RESPONSE:", django_response)

        return {
            "embedding": avg_embedding.tolist(),
            "status": "success",
            "django_response": django_response
        }
        