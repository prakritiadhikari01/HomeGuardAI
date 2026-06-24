# app/services/embedderai.py
from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_embedding(text: str) -> list:
    return get_model().encode(text).tolist()

def build_event_text(event: dict) -> str:
    return (
        f"Time: {event.get('timestamp', 'unknown')}. "
        f"Camera: {event.get('camera_id', 'unknown')}. "
        f"Location: {event.get('location', 'unknown')}. "
        f"General: {event.get('general_description', '')}. "
        f"Clothing: {event.get('clothing_description', '')}. "
        f"Action: {event.get('action_description', '')}. "
        f"Person present: {event.get('person_present', False)}. "
        f"Person type: {event.get('person_type', 'UNKNOWN')}. "
        f"Person label: {event.get('person_label', 'unknown')}."
    )