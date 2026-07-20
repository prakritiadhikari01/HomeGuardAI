from __future__ import annotations

import time

import cv2

from app.core.config import settings
from app.infrastructure.vision.camera_capture import CameraCapture

from app.infrastructure.vision.opencv_motion_detector import OpenCVMotionDetector
from app.infrastructure.vision.yolo_detector import YOLODetector
from app.infrastructure.vision.bytetrack_tracker import ByteTrackTracker

from app.infrastructure.ai.insightface_service import InsightFaceService
from app.infrastructure.ai.face_recognition_service import FaceRecognitionService
from app.infrastructure.ai.known_faces_store import KnownFacesStore

from app.application.processors.motion_processor import MotionProcessor
from app.application.processors.detection_processor import DetectionProcessor
from app.application.processors.tracking_processor import TrackingProcessor
from app.application.processors.recognition_processor import RecognitionProcessor


def main():

    print("=" * 70)
    print("HOMEGUARD LIVE AI PIPELINE")
    print("=" * 70)

    print("Loading known faces...")
    faces = KnownFacesStore()
    faces.refresh()

    print("Initializing processors...")

    motion = MotionProcessor(
        OpenCVMotionDetector(
            min_area=settings.MOTION_MIN_AREA
        )
    )

    detection = DetectionProcessor(
        YOLODetector()
    )

    tracking = TrackingProcessor(
        ByteTrackTracker()
    )

    recognition = RecognitionProcessor(
        InsightFaceService(),
        FaceRecognitionService(faces)
    )

    capture = CameraCapture(0)

    print()
    print("Press Q to quit")
    print()

    previous = time.time()

    while True:

        frame = capture.read()

        if frame is None:
            continue

        motion_result = motion.process(frame)

        perception = detection.process(
            frame=frame,
            frame_index=0,
            timestamp=time.time()
        )

        track_result = tracking.process(perception)

        recognition.process(
            frame,
            track_result
        )

        #
        # Draw tracks
        #

        for track in track_result.active_tracks:

            x1, y1, x2, y2 = track.detection.bbox

            color = (0,255,0)

            label = f"Track {track.track_id}"

            if getattr(track, "is_recognized", False):

                label += f" | {track.person_label}"

            elif getattr(track, "is_unknown_person", False):

                label += " | Unknown"

            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        #
        # Motion
        #

        motion_text = "Motion"

        if not motion_result.motion_detected:
            motion_text = "No Motion"

        cv2.putText(
            frame,
            motion_text,
            (20,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )

        #
        # FPS
        #

        now = time.time()

        fps = 1/(now-previous)

        previous = now

        cv2.putText(
            frame,
            f"FPS : {fps:.1f}",
            (20,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,0),
            2
        )

        #
        # Counts
        #

        cv2.putText(
            frame,
            f"Tracks : {len(track_result.active_tracks)}",
            (20,90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.imshow(
            "HomeGuard Live Pipeline",
            frame
        )

        key = cv2.waitKey(1)

        if key & 0xFF == ord("q"):
            break

    capture.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()