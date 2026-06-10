# app/services/cloudinary_service.py

import cloudinary
import cloudinary.uploader

from decouple import config


cloudinary.config(
    cloud_name=config(
        "CLOUDINARY_CLOUD_NAME"
    ),
    api_key=config(
        "CLOUDINARY_API_KEY"
    ),
    api_secret=config(
        "CLOUDINARY_API_SECRET"
    ),
    secure=True
)


class CloudinaryService:

    @staticmethod
    def upload_image(image_path):

        try:

            result = (
                cloudinary.uploader.upload(
                    image_path,
                    folder="homeguard/events"
                )
            )

            return result.get(
                "secure_url"
            )

        except Exception as e:

            print(
                "Cloudinary Upload Error:",
                e
            )

            return None