# app/core/security.py
from app.core.config import settings


def get_django_auth_headers() -> dict:
    """
    Shared-secret header sent on every Django API call from the AI engine.
    The value must exactly match AI_ENGINE_API_KEY in Django's settings/.env.
    """
    return {"X-API-Key": settings.API_SECRET_KEY}