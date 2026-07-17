import cv2

from app.infrastructure.vision.yolo_detector import YOLODetector

detector = YOLODetector()

cap = cv2.VideoCapture(0)

while True:

    ok, frame = cap.read()

    if not ok:
        break

    result = detector.detect(frame)

    print(result)

    for detection in result.detections:

        x1, y1, x2, y2 = detection.bbox

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            detection.object_type.value,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("YOLO", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()