from __future__ import annotations

from typing import List

import cv2

from app.domain.perception.motion import MotionResult


class OpenCVMotionDetector:
    """CPU-friendly motion detector using MOG2 background subtraction.
    One instance per camera — it holds background-model state."""

    def __init__(
        self,
        history: int = 500,
        var_threshold: int = 16,
        detect_shadows: bool = True,
        min_area: int = 1500,
    ):
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def detect(self, frame) -> MotionResult:
        mask = self.background.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # 127 = shadow in MOG2
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[list[int]] = []
        total_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x + w, y + h])
            total_area += area

        return MotionResult(
            motion_detected=len(boxes) > 0,
            motion_score=float(total_area),
            changed_regions=[(x1, y1, x2, y2) for x1, y1, x2, y2 in boxes],
        )
