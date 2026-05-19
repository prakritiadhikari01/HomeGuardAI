import base64
import cv2
import numpy as np

from insightface.app import FaceAnalysis


class EmbeddingService:

    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=0)

    @staticmethod
    def extract_embedding(image_base64):

        try:
            image_data = base64.b64decode(image_base64)

            np_arr = np.frombuffer(image_data, np.uint8)

            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            faces = EmbeddingService.face_app.get(img)

            if not faces:
                return None

            face = faces[0]

            embedding = face.embedding.tolist()

            return embedding

        except Exception as e:
            print("Embedding extraction error:", e)
            return None