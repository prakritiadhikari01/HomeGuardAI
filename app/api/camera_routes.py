#app\api\camera_routes.py
from fastapi import APIRouter

router = APIRouter()

@router.post("/camera/register")
def register_camera(payload: dict):
    CameraRegistrationService.register_camera(payload)
    return {"status": "success", "message": "Camera registered successfully."}