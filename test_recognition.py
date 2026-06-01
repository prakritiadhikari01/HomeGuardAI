import cv2

from app.services.face_recognition_service import (
    FaceRecognitionService
)


def main():

    service = FaceRecognitionService()

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = service.recognize(frame)

        if result:

            print(result)

        cv2.imshow("Recognition", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()