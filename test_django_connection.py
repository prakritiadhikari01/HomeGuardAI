

from app.infrastructure.api.django_client import DjangoClient


def main():

    faces = DjangoClient.get_all_faces()

    print(f"Total Faces: {len(faces)}")

    for face in faces:

        print()
        print("Name:", face["label_name"])
        print("Email:", face["user_email"])
        print("Embedding Length:", len(face["embedding"]))


if __name__ == "__main__":
    main()