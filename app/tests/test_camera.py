from __future__ import annotations

import cv2

from app.infrastructure.vision.camera_capture import CameraCapture


def camera_tests(runner):

    runner.run(
        "Laptop Camera Connection",
        test_laptop_camera,
    )


def test_laptop_camera():

    print("\nOpening laptop camera...")

    capture = CameraCapture(0)   # Webcam

    frame = capture.read()

    assert frame is not None, "Failed to read frame from webcam."

    h, w = frame.shape[:2]

    print(f"Resolution : {w} x {h}")

    capture.release()