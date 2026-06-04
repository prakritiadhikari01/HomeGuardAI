#app\api\routes.py
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.analysis_service import AnalysisService
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.face_embedding_service import FaceEmbeddingService


router = APIRouter()


class AnalyzeRequest(BaseModel):
    image: str
class EmbeddingRequest(BaseModel):
    image: str

@router.post("/analyze")
def analyze_frame(payload: AnalyzeRequest):

    try:
        result = AnalysisService.analyze_image(
            image_base64=payload.image
        )

        return result

    except Exception as e:
        return {
            "error": str(e),
            "face_match": None,
            "confidence": 0.0,
            "type": "ERROR"
        }
    
@router.post("/extract-embedding")
def extract_embedding(payload: EmbeddingRequest):

    embedding = FaceEmbeddingService.extract_embedding(
        payload.image
    )

    if embedding is None:
        return {
            "success": False,
            "embedding": None
        }

    return {
        "success": True,
        "embedding": embedding
    }