from __future__ import annotations

import time

from app.domain.runtime.camera_runtime import CameraRuntime
from app.domain.runtime.house_runtime import HouseRuntime
from app.domain.session.event_context import EventContext
from app.infrastructure.vision.camera_capture import CameraCapture


class CameraWorker:
    """Owns exactly one camera's frame-acquisition loop and nothing
    else. Everything about *what happens* to a frame lives in
    PipelineProcessor — CameraWorker only knows how to get a frame and
    hand it over, so it never grows into a God object as new pipeline
    stages get added.

    Runs on its own daemon thread, started/stopped by RuntimeManager."""

    def __init__(
        self,
        camera_runtime: CameraRuntime,
        house_runtime: HouseRuntime,
        pipeline,  # PipelineProcessor
        capture: CameraCapture,
    ):
        self._camera_runtime = camera_runtime
        self._house_runtime = house_runtime
        self._pipeline = pipeline
        self._capture = capture
        self._should_run = True

    def run(self) -> None:
        self._camera_runtime.mark_connected()
        self._camera_runtime.mark_processing_started()

        try:
            while self._should_run and self._camera_runtime.processing:
                frame = self._capture.read()
                if frame is None:
                    self._camera_runtime.reconnect_attempts += 1
                    time.sleep(1)
                    continue

                context = EventContext(
                    frame=frame,
                    house_id=self._camera_runtime.house_id,
                    device_id=self._camera_runtime.camera_id,
                    camera_runtime=self._camera_runtime,
                    house_settings=self._house_runtime.settings,
                    security_mode=self._house_runtime.security_mode,
                )

                self._pipeline.process(context)
                self._camera_runtime.frame_processed()
        finally:
            self._capture.release()
            self._camera_runtime.mark_processing_stopped()
            self._camera_runtime.mark_disconnected()

    def stop(self) -> None:
        self._should_run = False
        self._camera_runtime.mark_processing_stopped()
