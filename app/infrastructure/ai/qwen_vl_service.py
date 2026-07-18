import base64

import cv2
import requests


class QwenVLService:
    """Calls a locally-running Ollama server (Qwen2.5-VL) to describe a
    representative frame from a finished session's clip. Any failure
    (Ollama down, timeout, model not pulled) returns None — never
    raises — so EnrichmentProcessor's rule-based fallback always has a
    clean path to take over."""

    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "qwen2.5vl"
    TIMEOUT_SECONDS = 30

    @staticmethod
    def summarize(frame, camera_location: str | None) -> str | None:
        if frame is None:
            return None

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return None

        image_b64 = base64.b64encode(buffer).decode("utf-8")
        location_phrase = f" at {camera_location}" if camera_location else ""
        prompt = (
            f"Describe in one short sentence what this person is doing{location_phrase}. "
            "Mention clothing color and anything they are carrying, if visible. "
            "Be factual and concise."
        )

        try:
            response = requests.post(
                QwenVLService.OLLAMA_URL,
                json={
                    "model": QwenVLService.MODEL,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                },
                timeout=QwenVLService.TIMEOUT_SECONDS,
            )
            data = response.json()
            text = data.get("response", "").strip()
            return text or None
        except Exception as e:
            print(f"[QwenVLService] error: {e}")
            return None
