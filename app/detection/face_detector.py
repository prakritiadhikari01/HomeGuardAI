import cv2
import numpy as np
import base64

from insightface.app import FaceAnalysis


class FaceDetector:

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(ctx_id=0)

    @staticmethod
    def decode_base64_image(image_base64: str):
        image_data = base64.b64decode(image_base64)

        np_arr = np.frombuffer(image_data, np.uint8)

        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        return image

    @classmethod
    def detect_faces(cls, image_base64: str):

        image = cls.decode_base64_image(image_base64)

        faces = cls.app.get(image)

        return faces