# app/detection/motion_detector.py
import cv2


class MotionDetector:
    """
    Cheap frame-differencing gate — the first pipeline stage, per section 6
    of your architecture doc. One instance per camera (holds previous-frame
    state), so it must live on the CameraRecognitionService worker.
    """

    def __init__(self, min_area=2500, blur_size=21):
        self.previous_frame = None
        self.min_area = min_area
        self.blur_size = blur_size

    def detect(self, frame) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        if self.previous_frame is None:
            self.previous_frame = gray
            return False  # first frame ever seen — nothing to compare against yet

        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.previous_frame = gray

        return any(cv2.contourArea(c) >= self.min_area for c in contours)