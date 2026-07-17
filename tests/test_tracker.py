import cv2

from app.infrastructure.vision.yolo_detector import YOLODetector
from app.infrastructure.vision.bytetrack_tracker import ByteTrackTracker


detector = YOLODetector()

tracker = ByteTrackTracker()

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    perception = detector.detect(frame)

    result = tracker.update(perception)

    for track in result.tracks:

        x1, y1, x2, y2 = track.current_bbox

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Track {track.track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        print(
            track.track_id,
            track.duration_seconds,
        )

    cv2.imshow(
        "ByteTrack Test",
        frame,
    )

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()