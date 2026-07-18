from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import cv2
import numpy as np

from app.controllers.enrollment_controller import EnrollmentController

router = APIRouter(prefix="/enrollment", tags=["Enrollment"])

controller = EnrollmentController()


@router.post("/start")
def start_enrollment(session_id: str = Form(...)):
    """Starts a new AI enrollment session. Browser sends the Django
    EnrollmentSession id; the AI Engine creates its own in-memory
    session keyed separately."""
    print(f"[enrollment/start] django session: {session_id}")
    return controller.start_enrollment(django_session_id=session_id)


@router.post("/process-frame")
async def process_frame(session_id: str = Form(...), image: UploadFile = File(...)):
    """Process one webcam frame. session_id here is the AI Engine's own
    session id (returned by /start), not the Django one."""
    image_bytes = await image.read()
    np_image = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image.")

    return controller.process_frame(session_id=session_id, frame=frame)


@router.post("/cancel")
def cancel_enrollment(session_id: str = Form(...)):
    return controller.cancel_enrollment(session_id=session_id)
