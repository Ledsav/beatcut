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


def detect_beats(audio_path: str, sensitivity: float):
    """
    Loads audio, runs beat tracking, subsamples by sensitivity.

    sensitivity: 0.3 -> step=3 (fewer beats used), 2.0 -> step=1 (all beats).
    Formula: step = max(1, round(1 / sensitivity))

    Returns (beat_times_list, duration_seconds, bpm).
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    step = max(1, round(1.0 / sensitivity))
    beat_times = beat_times[::step]

    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    return beat_times, duration, bpm


def _apply_flash(frame: np.ndarray, t: float, fps: int) -> np.ndarray:
    """White flash overlay decaying by 0.72 per frame. frame is BGR."""
    alpha = 0.9 * (0.72 ** (t * fps))
    white = np.ones_like(frame, dtype=np.uint8) * 255
    return cv2.addWeighted(frame, 1 - alpha, white, alpha, 0)


def _apply_fade(frame_a: np.ndarray, frame_b: np.ndarray, t: float, duration: float) -> np.ndarray:
    """Linear cross-dissolve between two BGR frames."""
    ratio = t / max(duration, 1e-6)
    ratio = min(ratio, 1.0)
    return cv2.addWeighted(frame_a, 1 - ratio, frame_b, ratio, 0)


def _apply_zoom_in(frame: np.ndarray, t: float, duration: float) -> np.ndarray:
    """Scale 1.3->1.0 over the clip duration using warpAffine. frame is BGR."""
    h, w = frame.shape[:2]
    scale = 1.3 - 0.3 * (t / max(duration, 1e-6))
    scale = max(scale, 1.0)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
