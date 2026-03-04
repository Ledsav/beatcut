import cv2
import numpy as np
import librosa
import warnings
from PIL import Image

warnings.filterwarnings("ignore")

# Cascade paths (shipped with opencv-python)
_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_PROFILE  = cv2.data.haarcascades + "haarcascade_profileface.xml"


def detect_and_crop(image_path: str, output_w: int, output_h: int, zoom: float):
    """
    Reads image_path, detects the largest face, centers the crop around
    (cx, cy = face_x + w/2, face_y + h*0.35), crops to output_w x output_h
    at the given zoom level (larger zoom = wider context), resizes to output
    dimensions with LANCZOS interpolation.

    Returns (bgr_array, face_found).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    ih, iw = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_found = False
    cx, cy = iw // 2, ih // 2  # fallback: image center

    for cascade_path in [_FRONTAL, _PROFILE]:
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces):
            # Pick largest by area
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cx = x + w // 2
            cy = int(y + h * 0.35)  # include forehead
            face_found = True
            break

    # Compute crop window: output aspect ratio divided by zoom
    aspect = output_w / output_h
    crop_h = int(ih / zoom)
    crop_w = int(crop_h * aspect)
    if crop_w > iw:
        crop_w = iw
        crop_h = int(crop_w / aspect)

    # Clamp so the crop stays within image bounds
    x1 = max(0, min(cx - crop_w // 2, iw - crop_w))
    y1 = max(0, min(cy - crop_h // 2, ih - crop_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    cropped = img[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (output_w, output_h), interpolation=cv2.INTER_LANCZOS4)
    return resized, face_found
