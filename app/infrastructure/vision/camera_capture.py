from __future__ import annotations

import cv2


class CameraCapture:
    """Owns one cv2.VideoCapture with basic auto-reconnect. CameraWorker
    only calls read() — it never touches cv2 directly.

    Previously this reconnect logic lived inline inside
    CameraRecognitionService.start(); pulled out so CameraWorker's loop
    stays "read a frame, hand it to the pipeline" and nothing else."""

    def __init__(self, stream_url: str):
        self._stream_url = stream_url
        self._cap = cv2.VideoCapture(stream_url)

    def read(self):
        if not self._cap.isOpened():
            self._reconnect()
            return None

        success, frame = self._cap.read()
        if not success:
            self._reconnect()
            return None
        return frame

    def _reconnect(self) -> None:
        self._cap.release()
        self._cap = cv2.VideoCapture(self._stream_url)

    def release(self) -> None:
        self._cap.release()
