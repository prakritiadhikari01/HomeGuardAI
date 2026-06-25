# test.py

from app.services.face_enrollment_service import (
    FaceEnrollmentService
)


HOME_MEMBER_ID = "smarika"
LABEL_NAME = "Pratik"


def main():

    service = FaceEnrollmentService()

    result = service.enroll_face(
        home_member_id=HOME_MEMBER_ID,
        label_name=LABEL_NAME
    )

    if not result:
        print("Enrollment failed")
        return

    print("\nEnrollment Successful")
    print(
        "Embedding Length:",
        len(result["embedding"])
    )

    print(
        "Django Response:",
        result["django_response"]
    )



if __name__ == "__main__":
    main()