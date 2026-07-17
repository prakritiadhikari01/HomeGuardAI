import cv2

from app.infrastructure.vision.opencv_motion_detector import OpenCVMotionDetector



detector = OpenCVMotionDetector()

cap = cv2.VideoCapture(0)

while True:

    ok, frame = cap.read()

    if not ok:
        break

    result = detector.detect(frame)

    for box in result.changed_regions:

        x1, y1, x2, y2 = box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

    text = "Motion" if result.motion_detected else "No Motion"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if result.motion_detected else (0, 0, 255),
        2,
    )

    cv2.imshow("Motion Detector", frame)
    cv2.imshow("Motion Mask", detector.background.apply(frame))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()