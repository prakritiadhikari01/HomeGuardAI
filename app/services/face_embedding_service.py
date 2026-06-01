from insightface.app import FaceAnalysis


class FaceEmbeddingService:

    def __init__(self):

        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=0)

    def detect_faces(self, frame):

        return self.face_app.get(frame)

    def get_embedding(self, frame):

        faces = self.detect_faces(frame)

        if len(faces) == 0:
            return None

        return faces[0].embedding

    def get_face(self, frame):

        faces = self.detect_faces(frame)

        if len(faces) == 0:
            return None

        return faces[0]