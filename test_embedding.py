import cv2

from app.services.face_embedding_service import (
    FaceEmbeddingService
)


def main():

    service = FaceEmbeddingService()

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        embedding = service.get_embedding(frame)

        if embedding is not None:

            print(
                "Embedding Length:",
                len(embedding)
            )

            break

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()