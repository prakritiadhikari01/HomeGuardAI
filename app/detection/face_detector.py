from insightface.app import FaceAnalysis


class FaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_faces(self, frame):
        if frame is None:
            return []

        faces = self.app.get(frame)

        results = []

        for face in faces:
            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "embedding": face.embedding,
                "det_score": float(face.det_score) if hasattr(face, "det_score") else 0.0
            })

        return results