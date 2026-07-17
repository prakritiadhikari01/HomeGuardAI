# app/core/model_registry.py
import threading
from insightface.app import FaceAnalysis


class ModelRegistry:
    """
    Loads InsightFace exactly once per process.

    Previously, FaceEmbeddingService (and face_detector.FaceDetector,
    separately) each called FaceAnalysis(...).prepare(...) on every
    instantiation. Since AIEventOrchestrator created a fresh
    FaceRecognitionService() on every process_frame() call, this meant
    all 5 ONNX models were reloaded from disk every ~5 seconds, forever —
    which is exactly the repeating "find model: ..." spam in your logs.
    """

    _instance = None
    _lock = threading.Lock()
    _inference_lock = threading.Lock()  # serialize .get() calls across camera threads

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._load()
        return cls._instance

    def _load(self):
        print("[ModelRegistry] Loading InsightFace (buffalo_l) — should print ONCE per process.")
        self.face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def get(self, frame):
        with self._inference_lock:
            return self.face_app.get(frame)