import cv2
import uuid
import os


class EvidenceService:

    STORAGE_DIR = "evidence"

    @staticmethod
    def save_frame(frame):

        os.makedirs(
            EvidenceService.STORAGE_DIR,
            exist_ok=True
        )

        filename = (
            f"{uuid.uuid4()}.jpg"
        )

        filepath = os.path.join(
            EvidenceService.STORAGE_DIR,
            filename
        )

        cv2.imwrite(
            filepath,
            frame
        )

        return filepath