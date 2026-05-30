import requests
import numpy as np
from scipy.spatial.distance import cosine

DJANGO_API = "http://192.168.1.12:8000/api/v1/faces/"


class RecognitionService:

    def load_registered_faces(self):

        response = requests.get(DJANGO_API)

        response.raise_for_status()

        return response.json()["faces"]

    def recognize_embedding(self, current_embedding):

        faces = self.load_registered_faces()

        best_match = None
        best_distance = 999

        for face in faces:

            stored_embedding = np.array(
                face["embedding"],
                dtype=np.float32
            )

            distance = cosine(
                current_embedding,
                stored_embedding
            )

            if distance < best_distance:
                best_distance = distance
                best_match = face

        if best_match and best_distance < 0.4:

            return {
                "status": "known",
                "name": best_match["label_name"],
                "email": best_match["user_email"],
                "distance": float(best_distance)
            }

        return {
            "status": "unknown"
        }