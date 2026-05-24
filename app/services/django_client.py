import requests
from decouple import config


class DjangoClient:

    BASE_URL = config("DJANGO_API_URL")

    @staticmethod
    def save_face_profile(payload):

        url = f"{DjangoClient.BASE_URL}/faces/save/"

        try:

            response = requests.post(
                url,
                json=payload,
                timeout=30
            )
            try:
                data=response.json()
            except Exception as e:
                print("Error parsing JSON response:", e)
                data = {
                    "status": "error",
                    "message": "Invalid JSON response from server"
                }
            return data

        except Exception as e:

            return {
                "status": "error",
                "message": str(e)
            }