import requests
from app.core.config import settings
from app.core.security import get_django_auth_headers


class DjangoClient:
    BASE_URL = settings.DJANGO_API_URL

    @staticmethod
    def send_detection_event(payload):
        url = f"{DjangoClient.BASE_URL}/events/ingest/"
        try:
            response = requests.post(url, json=payload, headers=get_django_auth_headers(),timeout=settings.API_TIMEOUT)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def enrich_event(event_id: str, payload: dict):
        """
        PATCH /api/events/<uuid:event_id>/enrich/
        Called once a session finishes — attaches clip_url/thumbnail_url/
        event_summary/attributes. `payload` keys not present are simply
        omitted rather than sent as null, so a partial enrichment (e.g.
        summary ready, clip not yet) doesn't overwrite fields Django
        already has.
        """
        url = f"{DjangoClient.BASE_URL}/events/{event_id}/enrich/"
        body = {k: v for k, v in payload.items() if v is not None}
        try:
            response = requests.patch(url, json=body, headers=get_django_auth_headers(), timeout=settings.API_TIMEOUT)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def send_alert(payload: dict):
        """
        POST /api/alerts/  — endpoint does not exist yet on the Django
        side (not present in urls.py as of this build). Wire this up
        once AlertView + the route are added; until then this will
        return a connection/404 error, which the caller does not
        currently handle specially.
        """
        url = f"{DjangoClient.BASE_URL}/alerts/"
        try:
            response = requests.post(url, json=payload, headers=get_django_auth_headers(), timeout=settings.API_TIMEOUT)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def save_face_profile(payload):
        url = f"{DjangoClient.BASE_URL}/faces/save/"
        try:
            response = requests.post(url, json=payload, headers=get_django_auth_headers(), timeout=settings.API_TIMEOUT)
            print(f"[DJANGO RESPONSE] Status Code: {response.status_code}")
            print(f"[DJANGO RESPONSE] Response Body: {response.text}")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def get_all_faces():
        url = f"{DjangoClient.BASE_URL}/faces/all/"
        try:
            response = requests.get(url, headers=get_django_auth_headers(), timeout=settings.API_TIMEOUT)
            return response.json().get("faces", [])
        except Exception as e:
            print(f"Error fetching faces: {e}")
            return []

    @staticmethod
    def get_active_cameras():
        url = f"{DjangoClient.BASE_URL}/devices/active/"
        try:
            response = requests.get(url, headers=get_django_auth_headers(), timeout=settings.API_TIMEOUT)
            data = response.json()
            print(f"[DJANGO RESPONSE] Status Code: {response.status_code}")
            print(f"[DJANGO RESPONSE] Response Body: {response.text}")
            return data.get("cameras", [])
        except Exception as e:
            print(f"Error fetching active cameras: {e}")
            return []