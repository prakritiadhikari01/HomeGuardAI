# app/services/django_client.py

import requests
from decouple import config


class DjangoClient:

    BASE_URL = config("DJANGO_API_URL")

    @staticmethod
    def send_detection_event(payload):

        url = f"{DjangoClient.BASE_URL}/events/detect/"

        try:

            response = requests.post(
                url,
                json=payload,
                timeout=30
            )

            return response.json()

        except Exception as e:

            return {
                "status": "error",
                "message": str(e)
            }
        
    @staticmethod
    def save_face_profile(payload):

        url = f"{DjangoClient.BASE_URL}/faces/save/"

        try:

            response = requests.post(
                url,
                json=payload,
                timeout=30
            )
            print(f"[DJANGO RESPONSE] Status Code: {response.status_code}")
            print(f"[DJANGO RESPONSE] Response Body: {response.text}")
            return response.json()

        except Exception as e:

            return {
                "status": "error",
                "message": str(e)
            }