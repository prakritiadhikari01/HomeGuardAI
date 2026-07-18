import cloudinary
import cloudinary.uploader

from app.core.config import settings

# FIX: this file previously contained a copy-paste of SnapshotService
# instead of actual Cloudinary code, and had no upload_video() at all
# (EnrichmentProcessor calls upload_video() to store finished clips).

cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True,
)


class CloudinaryService:
    @staticmethod
    def upload_image(image_path: str) -> dict:
        """Returns {"image_url": ..., "thumbnail_url": ...} — dict shape
        so SnapshotService can hand it straight back to EnrichmentProcessor."""
        try:
            result = cloudinary.uploader.upload(
                image_path, folder=settings.CLOUDINARY_FOLDER
            )
            secure_url = result.get("secure_url")
            thumbnail_url = cloudinary.CloudinaryImage(result.get("public_id")).build_url(
                width=200, height=200, crop="thumb"
            ) if result.get("public_id") else None
            return {"image_url": secure_url, "thumbnail_url": thumbnail_url}
        except Exception as e:
            print(f"[CloudinaryService] upload_image error: {e}")
            return {"image_url": None, "thumbnail_url": None}

    @staticmethod
    def upload_video(video_path: str) -> str | None:
        """Uploads a finished event clip (mp4 from ClipRecorder.finalize())
        and returns its secure_url, or None on failure."""
        try:
            result = cloudinary.uploader.upload(
                video_path,
                folder=settings.CLOUDINARY_FOLDER,
                resource_type="video",
            )
            return result.get("secure_url")
        except Exception as e:
            print(f"[CloudinaryService] upload_video error: {e}")
            return None
