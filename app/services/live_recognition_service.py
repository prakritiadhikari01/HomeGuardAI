import cv2

from app.services.recognition_service import (
    RecognitionService
)


class LiveRecognitionService:

    def __init__(self):

        self.recognition_service = (
            RecognitionService()
        )

    def start(self):

        cap = cv2.VideoCapture(0)

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = (
                self.recognition_service
                .recognize_frame(frame)
            )

            for result in results:

                bbox = result["bbox"]
                match = result["match"]

                x1, y1, x2, y2 = bbox

                if match["matched"]:
                    name = match["user"]["name"]
                    label = f"{name} {match['score']:.2f}"
                    color = (0, 255, 0)
                else:
                    label = "UNKNOWN"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        

            cv2.imshow(
                "Live Recognition",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()