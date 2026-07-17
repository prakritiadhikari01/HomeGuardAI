from __future__ import annotations

from app.application.session.event_session_manager import EventSessionManager


class CameraPipeline:
    """
    Main AI pipeline executed for every frame.

    Pipeline:

        Frame
          │
          ▼
      Motion Detection
          │
          ▼
      Object Detection
          │
          ▼
      Object Tracking
          │
          ▼
      Face Recognition
          │
          ▼
      Event Session Manager
          │
          ▼
      Timeline Update
          │
          ▼
      Alert Evaluation

    The pipeline coordinates processors only.
    It contains no AI implementation itself.
    """

    def __init__(
        self,
        motion_processor,
        detection_processor,
        tracking_processor,
        recognition_processor,
        timeline_processor,
        alert_processor,
        event_session_manager: EventSessionManager,
    ):

        self.motion_processor = motion_processor
        self.detection_processor = detection_processor
        self.tracking_processor = tracking_processor
        self.recognition_processor = recognition_processor
        self.timeline_processor = timeline_processor
        self.alert_processor = alert_processor
        self.event_session_manager = event_session_manager

    def process(
        self,
        frame,
        camera_runtime,
        house_runtime,
    ):
        """
        Process one frame from one camera.

        Returns immediately.
        """

        # -------------------------------------------------
        # Stage 1
        # Motion Detection
        # -------------------------------------------------

        motion = self.motion_processor.process(
            frame=frame,
            camera=camera_runtime,
        )

        if not motion.detected:
            return

        # -------------------------------------------------
        # Stage 2
        # Object Detection
        # -------------------------------------------------

        detections = self.detection_processor.process(
            frame=frame,
            camera=camera_runtime,
        )

        if not detections:
            return

        # -------------------------------------------------
        # Stage 3
        # Tracking
        # -------------------------------------------------

        tracks = self.tracking_processor.process(
            frame=frame,
            detections=detections,
            camera=camera_runtime,
        )

        if not tracks:
            return

        # -------------------------------------------------
        # Stage 4
        # Recognition
        # -------------------------------------------------

        tracks = self.recognition_processor.process(
            frame=frame,
            tracks=tracks,
            camera=camera_runtime,
        )

        # -------------------------------------------------
        # Stage 5
        # Session Management
        # -------------------------------------------------

        sessions = self.event_session_manager.update(
            tracks=tracks,
            frame=frame,
            camera=camera_runtime,
            house=house_runtime,
        )

        # -------------------------------------------------
        # Stage 6
        # Timeline
        # -------------------------------------------------

        self.timeline_processor.process(
            sessions=sessions,
            house=house_runtime,
        )

        # -------------------------------------------------
        # Stage 7
        # Alerts
        # -------------------------------------------------

        self.alert_processor.process(
            sessions=sessions,
            house=house_runtime,
        )