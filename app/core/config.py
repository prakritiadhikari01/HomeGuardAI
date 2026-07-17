#app/core/config.py# app/core/config.py
from decouple import config


class Settings:
    AI_SERVER_PORT = config("AI_SERVER_PORT", default=8001, cast=int)
    DJANGO_API_URL = config("DJANGO_API_URL")
    API_SECRET_KEY = config("API_SECRET_KEY")

    FACE_MATCH_THRESHOLD = config("FACE_MATCH_THRESHOLD", default=0.40, cast=float)
    CAMERA_SYNC_INTERVAL_SECONDS = config("CAMERA_SYNC_INTERVAL_SECONDS", default=30, cast=int)
    EVENT_COOLDOWN_SECONDS = config("EVENT_COOLDOWN_SECONDS", default=5, cast=int)

    MOTION_MIN_AREA = config("MOTION_MIN_AREA", default=2500, cast=int)

    ENABLE_YOLO_GATE = config("ENABLE_YOLO_GATE", default=True, cast=bool)
    YOLO_PERSON_CONFIDENCE = config("YOLO_PERSON_CONFIDENCE", default=0.5, cast=float)

    CLOUDINARY_CLOUD_NAME = config("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY = config("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET = config("CLOUDINARY_API_SECRET")
    CLOUDINARY_FOLDER = config("CLOUDINARY_FOLDER", default="homeguard/events")


settings = Settings()