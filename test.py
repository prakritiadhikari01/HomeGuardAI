from app.services.face_enrollment_service import FaceEnrollmentService
from app.services.django_client import DjangoClient
service = FaceEnrollmentService()

result = service.enroll_face()

print("EMBEDDING LENGTH:", len(result["embedding"]))


from app.services.django_client import DjangoClient

response = DjangoClient.save_face_profile({
    "home_id": "YOUR_HOME_ID",
    "label_name": "Prakriti",
    "embedding": result["embedding"]
})
