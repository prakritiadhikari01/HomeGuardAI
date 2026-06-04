#app\api\server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

CAMERAS={}
FACES={}

class Camera(BaseModel):
    camera_id: str
    home_id: str
    stream_url: str
    location: str

@app.post("/camera/register/")
def register_camera(cam: Camera):
    CAMERAS[cam.camera_id] = cam.dict()
    print(f"Registered Camera: {cam.camera_id} at {cam.location}")


    return {"status": "registered", "message": f"Camera {cam.camera_id} registered successfully."}

class Face(BaseModel):
    face_profile_id: str
    name: str
    image_url: str

@app.post("/face/register/")
def register_face(face: Face):
    FACES[face.face_profile_id] = face.dict()
    print(f"Registered Face: {face.name} with profile ID {face.face_profile_id}")

    return {"status": "registered", "message": f"Face {face.name} registered successfully."}