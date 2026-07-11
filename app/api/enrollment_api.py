#app\api\enrollment_api.py
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
import cv2
import numpy as np

from app.controllers.enrollment_controller import (
    EnrollmentController,
)

router = APIRouter(
    prefix="/enrollment",
    tags=["Enrollment"],
)

# ---------------------------------------------------------
# Singleton Controller
# ---------------------------------------------------------

controller = EnrollmentController()


# ---------------------------------------------------------
# START ENROLLMENT
# ---------------------------------------------------------

@router.post("/start")
def start_enrollment(
    session_id: str = Form(...),
):
    """
    Starts a new AI enrollment session.

    Browser sends the Django EnrollmentSession ID.

    AI Engine contacts Django to retrieve:

    - home_member_id
    - label_name

    Then creates its own in-memory enrollment session.
    """
    print("="*50)
    print("start endpoint hit with django session:",session_id)
    print("="*50)
    return controller.start_enrollment(
        django_session_id=session_id,
    )


# ---------------------------------------------------------
# PROCESS FRAME
# ---------------------------------------------------------

@router.post("/process-frame")
async def process_frame(
    session_id: str = Form(...),
    image: UploadFile = File(...),
):
    """
    Process one webcam frame.

    session_id = AI Engine session id
    """

    image_bytes = await image.read()

    np_image = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        np_image,
        cv2.IMREAD_COLOR,
    )

    if frame is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid image.",
        )

    return controller.process_frame(
        session_id=session_id,
        frame=frame,
    )


# ---------------------------------------------------------
# CANCEL ENROLLMENT
# ---------------------------------------------------------

@router.post("/cancel")
def cancel_enrollment(
    session_id: str = Form(...),
):
    """
    Cancel AI enrollment session.
    """

    return controller.cancel_enrollment(
        session_id=session_id,
    )