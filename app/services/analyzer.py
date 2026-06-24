# app/services/analyzer.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2
import numpy as np

processor = None
model = None

def load_model():
    global processor, model
    if model is None:
        print("Loading BLIP model...")
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        print("BLIP model loaded!")

def analyze_frame(frame) -> dict:
    """Accept OpenCV frame (numpy array) directly."""
    load_model()

    # Convert OpenCV BGR frame to PIL RGB image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # General description
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=100)
    general = processor.decode(out[0], skip_special_tokens=True)

    # Clothing
    inputs2 = processor(image, "a person wearing", return_tensors="pt").to(device)
    out2 = model.generate(**inputs2, max_new_tokens=60)
    clothing = processor.decode(out2[0], skip_special_tokens=True)

    # Action
    inputs3 = processor(image, "the person is", return_tensors="pt").to(device)
    out3 = model.generate(**inputs3, max_new_tokens=60)
    action = processor.decode(out3[0], skip_special_tokens=True)

    person_present = any(
        word in general.lower()
        for word in ["person", "man", "woman", "people", "human"]
    )

    return {
        "general_description": general,
        "clothing_description": clothing,
        "action_description": action,
        "person_present": person_present,
    }