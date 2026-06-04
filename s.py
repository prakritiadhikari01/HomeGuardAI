from app.api.server import CAMERAS
from app.services.django_client import DjangoClient
import cv2

django_client = DjangoClient()

def run_ai_loop():

    while True:

        for cam_id, cam in CAMERAS.items():

            cap = cv2.VideoCapture(cam["stream_url"])

            ret, frame = cap.read()

            if not ret:
                continue
            
            event={
                "camera_id": cam_id,
                "home_id": cam["home_id"],
                "timestamp": cv2.getTickCount(),
                "person_name":'Unknown',
            }
            django_client.send_detection_event(event)
            # STEP 1: YOLO detect person
            # STEP 2: detect face
            # STEP 3: embedding
            # STEP 4: qdrant search

            print(f"[AI] Processing camera {cam_id}")