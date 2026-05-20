import requests


class DjangoClient:

    BASE_URL = "http://192.168.1.9:8000/api"

    @staticmethod
    def save_face_profile(payload):

        url = f"{DjangoClient.BASE_URL}/faces/save/"

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