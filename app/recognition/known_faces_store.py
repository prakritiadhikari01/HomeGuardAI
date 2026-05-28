# app/recognition/known_faces_store.py

import requests
import numpy as np


class KnownFacesStore:
    def __init__(self):
        self.known_faces = []

    def load_faces_from_django(self):
        try:
            response = requests.get("http://192.168.1.13:8000/api/faces/all/")
            data = response.json()

            self.known_faces = []

            for user in data["faces"]:   # ✅ FIX HERE
                embedding = np.asarray(user["embedding"], dtype=np.float32)

            norm = np.linalg.norm(embedding)
            if norm != 0:
                embedding = embedding / norm

                self.known_faces.append({
                    "id": user["id"],
                    "name": user["label_name"],
                    "embedding": embedding
                })

            print(f"Loaded {len(self.known_faces)} known faces")

        except Exception as e:
            print("Error loading known faces:", e)

    def get_all_faces(self):
        if self.known_faces is None:
            return []
        return self.known_faces
    