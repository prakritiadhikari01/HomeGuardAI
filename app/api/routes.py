import base64

import cv2
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from app.infrastructure.ai.face_recognition_service import FaceRecognitionService
from app.infrastructure.ai.insightface_service import InsightFaceService

router = APIRouter()

_insightface = InsightFaceService()
_recognition = FaceRecognitionService()


def _decode_base64_image(image_b64: str):
    image_bytes = base64.b64decode(image_b64)
    np_image = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(np_image, cv2.IMREAD_COLOR)


class AnalyzeRequest(BaseModel):
    image: str


class EmbeddingRequest(BaseModel):
    image: str


@router.post("/analyze")
def analyze_frame(payload: AnalyzeRequest):
    """Debug/manual-test endpoint — detect + match a single base64 frame
    against the known-faces cache, outside the live camera pipeline."""
    try:
        frame = _decode_base64_image(payload.image)
        face_data = _insightface.get_face_data(frame)
        if face_data is None:
            return {"face_match": None, "confidence": 0.0, "type": "UNKNOWN"}

        result = _recognition.match_embedding(face_data["embedding"])
        if result.get("status") == "known":
            return {
                "face_match": result.get("member_id"),
                "confidence": result.get("confidence_score", 0.0),
                "type": "KNOWN",
            }
        return {"face_match": None, "confidence": 0.0, "type": "UNKNOWN"}
    except Exception as e:
        return {"error": str(e), "face_match": None, "confidence": 0.0, "type": "ERROR"}


@router.post("/extract-embedding")
def extract_embedding(payload: EmbeddingRequest):
    """Debug/manual-test endpoint — returns a raw embedding for a single
    base64 frame, with no matching."""
    frame = _decode_base64_image(payload.image)
    face_data = _insightface.get_face_data(frame)

    if face_data is None:
        return {"success": False, "embedding": None}

    return {"success": True, "embedding": face_data["embedding"].tolist()}
