from __future__ import annotations

import numpy as np

from app.infrastructure.vision.opencv_motion_detector import OpenCVMotionDetector


def motion_tests(runner):

    runner.run(
        "Motion Detector Initialization",
        test_detector_creation,
    )

    runner.run(
        "Motion Detection",
        test_motion_detection,
    )


def test_detector_creation():

    detector = OpenCVMotionDetector()

    assert detector is not None

    print("Motion detector created.")


def test_motion_detection():

    detector = OpenCVMotionDetector()

    #
    # First frame
    #

    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)

    detector.detect(frame1)

    #
    # Second frame with motion
    #

    frame2 = frame1.copy()

    frame2[100:200, 100:200] = 255

    result = detector.detect(frame2)

    assert result is not None

    print(f"Motion Detected : {result.motion_detected}")

    print(f"Motion Score    : {result.motion_score}")

    print(f"Changed Regions : {len(result.changed_regions)}")