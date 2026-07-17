# app/services/cloudinary_service.py
import cloudinary
import cloudinary.uploader

from app.core.config import settings

cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True,
)


class CloudinaryService:
    @staticmethod
    def upload_image(image_path):
        try:
            result = cloudinary.uploader.upload(image_path, folder=settings.CLOUDINARY_FOLDER)
            return result.get("secure_url")
        except Exception as e:
            print("Cloudinary Upload Error:", e)
            return None