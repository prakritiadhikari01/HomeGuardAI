from __future__ import annotations

import threading
import time
from uuid import UUID

from app.core.config import settings
from app.domain.runtime.camera_runtime import CameraRuntime
from app.domain.runtime.house_runtime import HouseRuntime
from app.domain.runtime.setting_runtime import RuntimeDeviceSettings, RuntimeHouseSettings
from app.application.camera_worker import CameraWorker
from app.application.pipeline_processor import PipelineProcessor
from app.application.processors.alert_processor import AlertProcessor
from app.application.processors.detection_processor import DetectionProcessor
from app.application.processors.enrichment_processor import EnrichmentProcessor
from app.application.processors.motion_processor import MotionProcessor
from app.application.processors.recognition_processor import RecognitionProcessor
from app.application.processors.session_processor import SessionProcessor
from app.application.processors.timeline_processor import TimelineProcessor
from app.application.processors.tracking_processor import TrackingProcessor
from app.infrastructure.ai.face_recognition_service import FaceRecognitionService
from app.infrastructure.ai.insightface_service import InsightFaceService
from app.infrastructure.ai.known_faces_store import KnownFacesStore
from app.infrastructure.api.django_client import DjangoClient
from app.infrastructure.media.clip_recorder import ClipRecorder
from app.infrastructure.vision.bytetrack_tracker import ByteTrackTracker
from app.infrastructure.vision.camera_capture import CameraCapture
from app.infrastructure.vision.opencv_motion_detector import OpenCVMotionDetector
from app.infrastructure.vision.yolo_detector import YOLODetector


def _device_settings_from_payload(raw: dict) -> RuntimeDeviceSettings:
    return RuntimeDeviceSettings(
        enabled=raw.get("enabled", True),
        motion_detection=raw.get("motion_detection", True),
        human_detection=raw.get("human_detection", True),
        face_recognition=raw.get("face_recognition", True),
        animal_detection=raw.get("animal_detection", True),
        vehicle_detection=raw.get("vehicle_detection", True),
        package_detection=raw.get("package_detection", False),
        recording_enabled=raw.get("recording_enabled", True),
        snapshot_enabled=raw.get("snapshot_enabled", True),
        confidence_threshold=raw.get("confidence_threshold", 80),
    )


def _house_settings_from_payload(raw: dict) -> RuntimeHouseSettings:
    return RuntimeHouseSettings(
        timeline_enabled=raw.get("timeline_enabled", True),
        log_known_member_entry=raw.get("log_known_member_entry", True),
        log_unknown_person=raw.get("log_unknown_person", True),
        notify_unknown_person=raw.get("notify_unknown_person", True),
        notify_family_members=raw.get("notify_family_members", False),
        emergency_contact_enabled=raw.get("emergency_contact_enabled", False),
        escalate_repeated_detection=raw.get("escalate_repeated_detection", True),
        ai_summary_enabled=raw.get("ai_summary_enabled", True),
        clip_recording_enabled=raw.get("clip_recording_enabled", True),
        snapshot_enabled=raw.get("snapshot_enabled", True),
        ignore_animals=raw.get("ignore_animals", False),
        loitering_seconds=raw.get("loitering_seconds", 20),
        repeated_detection_window=raw.get("repeated_detection_window", 300),
    )


# Conservative defaults used only until the Django extension below ships —
# keeps the engine runnable today instead of crashing on a missing key.
_DEFAULT_HOUSE_SETTINGS = _house_settings_from_payload({})


class _RunningCamera:
    """Everything RuntimeManager needs to stop/restart one camera."""

    __slots__ = ("worker", "thread", "capture", "stream_url")

    def __init__(self, worker: CameraWorker, thread: threading.Thread, capture: CameraCapture, stream_url: str):
        self.worker = worker
        self.thread = thread
        self.capture = capture
        self.stream_url = stream_url


class RuntimeManager:
    """
    The central coordinator described in the architecture doc. Loads
    devices/settings/security-mode/faces from Django, starts/stops one
    CameraWorker thread per active camera, and periodically re-syncs so
    a device added/removed/reconfigured in Django takes effect without
    restarting the process.

    Never does computer vision itself — only coordinates.
    """

    def __init__(self, django_client: DjangoClient | None = None):
        self._client = django_client or DjangoClient
        self._houses: dict[UUID, HouseRuntime] = {}
        self._cameras: dict[UUID, _RunningCamera] = {}
        self._known_faces_store = KnownFacesStore()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._known_faces_store.refresh()
        self.sync()
        threading.Thread(target=self._sync_loop, daemon=True).start()

    def stop_all(self) -> None:
        with self._lock:
            for camera_id in list(self._cameras.keys()):
                self._stop_camera(camera_id)

    def sync(self) -> None:
        """One reconciliation pass: fetch active cameras from Django,
        start new ones, restart ones whose stream_url changed, stop ones
        that disappeared or went offline, and push settings updates into
        already-running CameraRuntime/HouseRuntime objects."""
        cameras_payload = self._client.get_active_cameras()
        incoming_ids = {UUID(c["id"]) for c in cameras_payload}

        with self._lock:
            # Stop cameras that are no longer active/present.
            for camera_id in list(self._cameras.keys()):
                if camera_id not in incoming_ids:
                    self._stop_camera(camera_id)

            for raw_camera in cameras_payload:
                self._reconcile_camera(raw_camera)

        self._known_faces_store.refresh()

    # ------------------------------------------------------------------
    # Internal reconciliation
    # ------------------------------------------------------------------

    def _reconcile_camera(self, raw_camera: dict) -> None:
        camera_id = UUID(raw_camera["id"])
        home_id = UUID(raw_camera["home_id"])
        stream_url = raw_camera.get("stream_url")

        house_runtime = self._get_or_create_house(home_id, raw_camera)
        house_runtime.mark_synced()

        device_settings = _device_settings_from_payload(raw_camera.get("settings", {}))

        existing = self._cameras.get(camera_id)

        if existing is None:
            if not stream_url or not device_settings.enabled:
                return  # nothing to stream, or explicitly disabled — skip starting
            self._start_camera(house_runtime, raw_camera, device_settings)
            return

        # Already running — settings changed?
        camera_runtime = house_runtime.get_camera(camera_id)
        if camera_runtime is not None:
            camera_runtime.apply_settings(device_settings)
            camera_runtime.mark_synced()

        if not device_settings.enabled:
            self._stop_camera(camera_id)
            return

        # Stream URL changed underneath us — restart with the new URL.
        if existing.stream_url != stream_url:
            self._stop_camera(camera_id)
            self._start_camera(house_runtime, raw_camera, device_settings)

    def _get_or_create_house(self, home_id: UUID, raw_camera: dict) -> HouseRuntime:
        house_runtime = self._houses.get(home_id)
        security_mode = raw_camera.get("security_mode", "NORMAL")
        house_settings_raw = raw_camera.get("home_settings")
        house_settings = (
            _house_settings_from_payload(house_settings_raw)
            if house_settings_raw
            else _DEFAULT_HOUSE_SETTINGS
        )

        if house_runtime is None:
            house_runtime = HouseRuntime(
                house_id=home_id, security_mode=security_mode, settings=house_settings
            )
            house_runtime.mark_running()
            self._houses[home_id] = house_runtime
        else:
            house_runtime.apply_security_mode(security_mode)
            if house_settings_raw:
                house_runtime.apply_settings(house_settings)

        return house_runtime

    def _start_camera(
        self, house_runtime: HouseRuntime, raw_camera: dict, device_settings: RuntimeDeviceSettings
    ) -> None:
        camera_id = UUID(raw_camera["id"])
        stream_url = raw_camera["stream_url"]

        camera_runtime = CameraRuntime(
            camera_id=camera_id,
            house_id=house_runtime.house_id,
            name=raw_camera.get("name", ""),
            location=raw_camera.get("location", ""),
            stream_url=stream_url,
            settings=device_settings,
        )
        house_runtime.add_camera(camera_runtime)

        pipeline = self._build_pipeline(house_runtime, camera_runtime)
        capture = CameraCapture(stream_url)
        worker = CameraWorker(
            camera_runtime=camera_runtime,
            house_runtime=house_runtime,
            pipeline=pipeline,
            capture=capture,
        )

        thread = threading.Thread(target=self._run_worker, args=(camera_id, worker), daemon=True)
        thread.start()

        self._cameras[camera_id] = _RunningCamera(
            worker=worker, thread=thread, capture=capture, stream_url=stream_url
        )
        print(f"[RuntimeManager] Started camera worker: {raw_camera.get('name')} ({camera_id})")

    def _run_worker(self, camera_id: UUID, worker: CameraWorker) -> None:
        try:
            worker.run()
        except Exception as e:
            print(f"[RuntimeManager] Camera worker {camera_id} crashed: {e}")
            with self._lock:
                self._cameras.pop(camera_id, None)

            for house_runtime in self._houses.values():
                house_runtime.remove_camera(camera_id)

    def _stop_camera(self, camera_id: UUID) -> None:
        running = self._cameras.pop(camera_id, None)
        if running is None:
            return
        running.worker.stop()
        for house_runtime in self._houses.values():
            house_runtime.remove_camera(camera_id)
        print(f"[RuntimeManager] Stopped camera worker: {camera_id}")

    def _build_pipeline(self, house_runtime: HouseRuntime, camera_runtime: CameraRuntime) -> PipelineProcessor:
        """Every camera gets its own MotionProcessor/DetectionProcessor/
        TrackingProcessor/ClipRecorder — those hold per-stream state
        (background model, tracker IDs, frame buffers) and must not be
        shared across cameras. InsightFaceService/FaceRecognitionService/
        KnownFacesStore ARE shared (they wrap process-wide singletons),
        since the face model itself is stateless per call."""

        motion_processor = MotionProcessor(OpenCVMotionDetector(min_area=settings.MOTION_MIN_AREA))
        detection_processor = DetectionProcessor(
            YOLODetector(confidence_threshold=camera_runtime.settings.confidence_threshold / 100)
        )
        tracking_processor = TrackingProcessor(ByteTrackTracker())
        session_processor = SessionProcessor(
            house_id=house_runtime.house_id,
            device_id=camera_runtime.camera_id,
            camera_name=camera_runtime.name,
            camera_location=camera_runtime.location,
        )
        recognition_processor = RecognitionProcessor(
            InsightFaceService(), FaceRecognitionService(self._known_faces_store)
        )
        timeline_processor = TimelineProcessor(self._client)
        clip_recorder = ClipRecorder(fps=settings.CLIP_FPS)
        enrichment_processor = EnrichmentProcessor(clip_recorder, self._client)
        alert_processor = AlertProcessor()

        return PipelineProcessor(
            motion_processor=motion_processor,
            detection_processor=detection_processor,
            tracking_processor=tracking_processor,
            session_processor=session_processor,
            recognition_processor=recognition_processor,
            timeline_processor=timeline_processor,
            enrichment_processor=enrichment_processor,
            alert_processor=alert_processor,
            clip_recorder=clip_recorder,
            django_client=self._client,
        )

    # ------------------------------------------------------------------
    # Background sync loop
    # ------------------------------------------------------------------

    def _sync_loop(self) -> None:
        while True:
            time.sleep(settings.CAMERA_SYNC_INTERVAL_SECONDS)
            try:
                self.sync()
            except Exception as e:
                print(f"[RuntimeManager] Sync loop error: {e}")
