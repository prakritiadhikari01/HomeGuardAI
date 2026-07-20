from __future__ import annotations

import cv2

from app.infrastructure.vision.camera_capture import CameraCapture


def run_live_camera():

    print("=" * 60)
    print("Press Q to quit")
    print("=" * 60)

    capture = CameraCapture(0)

    while True:

        frame = capture.read()

        if frame is None:
            print("Unable to read frame.")
            break

        cv2.imshow("HomeGuard AI Camera Test", frame)

        key = cv2.waitKey(1)

        if key & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()