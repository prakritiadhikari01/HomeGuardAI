from __future__ import annotations

import dataclasses
import time

from app.domain.session.event_context import EventContext
from app.infrastructure.api.django_client import DjangoClient


class PipelineProcessor:
    """The one place that knows the pipeline's stage order:

        Motion -> Detection -> Tracking -> Session -> Recognition
              -> Timeline -> Enrichment -> Alert

    CameraWorker's job is "read a frame, hand it to this, do it again"
    — it never touches sessions, tracks, or enrichment. Adding,
    reordering, or gating a stage is a change to this file only."""

    def __init__(
        self,
        motion_processor,
        detection_processor,
        tracking_processor,
        session_processor,
        recognition_processor,
        timeline_processor,
        enrichment_processor,
        alert_processor,
        clip_recorder,
        django_client: DjangoClient | None = None,
    ):
        self._motion = motion_processor
        self._detection = detection_processor
        self._tracking = tracking_processor
        self._sessions = session_processor
        self._recognition = recognition_processor
        self._timeline = timeline_processor
        self._enrichment = enrichment_processor
        self._alert = alert_processor
        self._clip_recorder = clip_recorder
        self._client = django_client or DjangoClient

    def process(self, context: EventContext) -> EventContext:

        if context.camera_settings.motion_detection:
            context.motion_result = self._motion.process(context.frame)

        if not context.should_run_detection:
            return context

        context.perception_result = self._detection.process(
            context.frame,
            frame_index=context.camera_runtime.frames_processed,
            timestamp=time.time(),
        )

        context.track_result = self._tracking.process(context.perception_result)

        context.active_sessions, context.finished_sessions = self._sessions.process(
            context.track_result
        )
        
        if context.camera_settings.face_recognition:
            self._recognition.process(context.frame, context.track_result)

        # ClipRecorder needs "every active track_id on this camera" + "the
        # raw frame" together, every frame — a rolling buffer, not a
        # decision, so it's driven from here rather than owned by any
        # single processor.
        if (
            context.house_settings.clip_recording_enabled
            and 
            context.camera_settings.recording_enabled
        ):
            for track in context.track_result.active_tracks:
                self._clip_recorder.add_frame(track.track_id, context.frame)

        self._timeline.process(context.active_sessions)

        for session in context.finished_sessions:
            self._enrichment.process(session)

            candidate = self._alert.process(
                session, context.house_settings, context.security_mode
            )
            if candidate is not None:
                self._client.send_alert(dataclasses.asdict(candidate))

        return context
