import cv2

from app.services.face_embedding_service import FaceEmbeddingService
from app.services.face_recognition_service import RecognitionService

CAMERA_URL = "http://192.168.1.16:8080/video"

cap = cv2.VideoCapture(CAMERA_URL)

embedding_service = FaceEmbeddingService()
recognizer = RecognitionService()

while True:

    success, frame = cap.read()

    if not success:
        break

    faces = embedding_service.detect_faces(frame)

    for face in faces:

        embedding = embedding_service.get_embedding(face)

        result = recognizer.recognize_embedding(
            embedding
        )

        label = result.get("name", "Unknown")

        print(result)

        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()