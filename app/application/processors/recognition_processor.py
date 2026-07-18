from app.domain.perception.detection import PersonStatus
from app.domain.perception.track_result import TrackResult
from app.infrastructure.ai.face_recognition_service import FaceRecognitionService
from app.infrastructure.ai.insightface_service import InsightFaceService


class RecognitionProcessor:
    """Stage 5 — face recognition, run per-TRACK, not per-frame.

    Pipeline:
        InsightFaceService.detect_face_in_region() -> quality scoring, every
            frame while unresolved
        InsightFaceService.extract_embedding()      -> only once a crop
            clears the quality bar
        FaceRecognitionService.match_embedding()     -> only once, against
            the best crop

    Cropping is owned by InsightFaceService (detect_face_in_region),
    not here — "what pixels does the face model need" is a face-service
    concern, not a recognition decision."""

    MIN_FACE_QUALITY = 0.60  # matches FaceEnrollmentService.MIN_FACE_CONFIDENCE

    def __init__(
        self,
        insightface_service: InsightFaceService,
        face_recognition_service: FaceRecognitionService,
    ):
        self._insightface = insightface_service
        self._recognition_service = face_recognition_service

    def process(self, frame, track_result: TrackResult) -> None:
        for track in track_result.person_tracks:
            if not track.active:
                continue

            if track.recognized_once and track.person_status != PersonStatus.UNSEEN:
                continue  # already resolved for this track — stop paying for InsightFace

            face, crop = self._insightface.detect_face_in_region(frame, track.current_bbox)
            if face is None:
                continue

            quality = float(face.det_score)

            if quality > track.best_face_confidence:
                track.update_best_face(crop=crop, bbox=face.bbox.tolist(), confidence=quality)

            if track.best_face_confidence < self.MIN_FACE_QUALITY:
                continue  # face seen, not clean enough to trust yet

            embedding = self._insightface.extract_embedding(face)
            if embedding is None:
                continue

            result = self._recognition_service.match_embedding(embedding)
            is_known = result.get("status") == "known"

            track.attach_recognition(
                status=PersonStatus.KNOWN if is_known else PersonStatus.UNKNOWN,
                label=result.get("person_label") if is_known else None,
                member_id=result.get("member_id") if is_known else None,
                face_profile_id=result.get("face_profile_id") if is_known else None,
                confidence=result.get("confidence_score") if is_known else None,
            )
