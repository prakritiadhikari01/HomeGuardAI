import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceDetector:

    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0)

    def detect_faces(self, image):
        """
        Accepts OpenCV frame (numpy array)
        """

        if image is None:
            return []

        if len(image.shape) != 3:
            return []

        faces = self.app.get(image)

        return faces if faces is not None else []