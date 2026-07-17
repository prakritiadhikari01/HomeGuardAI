from decouple import config


class Settings:

    # SERVER

    AI_SERVER_PORT = config(
        "AI_SERVER_PORT",
        default=8001,
        cast=int,
    )

    DJANGO_API_URL = config("DJANGO_API_URL")

    API_SECRET_KEY = config("API_SECRET_KEY")


    # CAMERA

    CAMERA_SYNC_INTERVAL_SECONDS = config(
        "CAMERA_SYNC_INTERVAL_SECONDS",
        default=30,
        cast=int,
    )

    EVENT_COOLDOWN_SECONDS = config(
        "EVENT_COOLDOWN_SECONDS",
        default=5,
        cast=int,
    )


    # MOTION DETECTION

    MOTION_MIN_AREA = config(
        "MOTION_MIN_AREA",
        default=2500,
        cast=int,
    )


    # OBJECT DETECTION (YOLO)

    YOLO_MODEL_PATH = config(
        "YOLO_MODEL_PATH",
        default="yolov8n.pt",
    )

    YOLO_CONFIDENCE = config(
        "YOLO_CONFIDENCE",
        default=0.45,
        cast=float,
    )

    YOLO_IOU = config(
        "YOLO_IOU",
        default=0.50,
        cast=float,
    )


    # FACE RECOGNITION

    FACE_MATCH_THRESHOLD = config(
        "FACE_MATCH_THRESHOLD",
        default=0.40,
        cast=float,
    )


    # TRACKING

    TRACK_BUFFER = config(
        "TRACK_BUFFER",
        default=30,
        cast=int,
    )

    TRACK_MATCH_THRESHOLD = config(
        "TRACK_MATCH_THRESHOLD",
        default=0.80,
        cast=float,
    )


    # VIDEO RECORDING

    PRE_EVENT_SECONDS = config(
        "PRE_EVENT_SECONDS",
        default=5,
        cast=int,
    )

    POST_EVENT_SECONDS = config(
        "POST_EVENT_SECONDS",
        default=5,
        cast=int,
    )


    # VLM

    VLM_MODEL = config(
        "VLM_MODEL",
        default="qwen2.5vl:7b",
    )


    # CLOUDINARY

    CLOUDINARY_CLOUD_NAME = config("CLOUDINARY_CLOUD_NAME")

    CLOUDINARY_API_KEY = config("CLOUDINARY_API_KEY")

    CLOUDINARY_API_SECRET = config("CLOUDINARY_API_SECRET")

    CLOUDINARY_FOLDER = config(
        "CLOUDINARY_FOLDER",
        default="homeguard/events",
    )


settings = Settings()