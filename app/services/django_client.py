# app/services/django_client.py

import requests
from decouple import config


class DjangoClient:

    BASE_URL = config("DJANGO_API_URL")

    @staticmethod
    def send_detection_event(payload):

        url = f"{DjangoClient.BASE_URL}/events/ingest/"

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
    
     
    @staticmethod
    def get_all_faces():

        url = f"{DjangoClient.BASE_URL}/faces/all/"

        try:

            response = requests.get(
                url,
                timeout=30
            )

            return response.json().get("faces", [])

        except Exception as e:

            print(f"Error fetching faces: {e}")
            return []
    
    @staticmethod
    def get_active_cameras():

        url = f"{DjangoClient.BASE_URL}/devices/active/"

        try:

            response = requests.get(
                url,
                timeout=30
            )
            data=response.json()
            print(f"[DJANGO RESPONSE] Status Code: {response.status_code}")
            print(f"[DJANGO RESPONSE] Response Body: {response.text}")
            return data.get("cameras", [])

        except Exception as e:

            print(f"Error fetching active cameras: {e}")
            return []