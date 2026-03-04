import cv2
import numpy as np
import librosa
import warnings
from PIL import Image as PILImage
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

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


def _make_polaroid_pil(rgb_frame: np.ndarray, pol_size: int,
                       b_side: int, b_top: int, b_bottom: int) -> PILImage.Image:
    """Wrap an RGB numpy frame in a white polaroid border. Returns RGBA PIL Image."""
    pol_w = pol_size + 2 * b_side
    pol_h = pol_size + b_top + b_bottom
    # Slightly warm off-white so it reads as paper
    pol = PILImage.new("RGBA", (pol_w, pol_h), (252, 250, 244, 255))
    photo = PILImage.fromarray(rgb_frame).resize((pol_size, pol_size), PILImage.LANCZOS)
    pol.paste(photo.convert("RGBA"), (b_side, b_top))
    return pol


def _build_polaroid_clips(frames, beat_times, output_w, output_h, fps,
                           polaroid_entry, progress_callback, cancel_flag):
    """
    Build moviepy clips for the Polaroid Stack effect.

    Each beat adds one new polaroid on top of an accumulating pile on a black canvas.
    Landing animation is controlled by polaroid_entry: drop | flash | fade | cut.
    The pile resets every len(frames) photos so it never gets too crowded.
    """
    total = len(beat_times) - 1
    if total == 0:
        return []

    # Polaroid sizing — photo occupies ~52 % of the shorter edge
    pol_size = int(min(output_w, output_h) * 0.52)
    b_side   = max(int(pol_size * 0.07), 8)
    b_top    = max(int(pol_size * 0.07), 8)
    b_bottom = max(int(pol_size * 0.26), 24)

    # Deterministic random angles / offsets (seeded so same photos → same pile every run)
    rng = np.random.default_rng(seed=7)
    angles    = rng.uniform(-14, 14,            total + 1).tolist()
    offsets_x = rng.integers(-output_w // 14,  output_w // 14,  total + 1).tolist()
    offsets_y = rng.integers(-output_h // 14,  output_h // 14,  total + 1).tolist()

    # Pre-render every rotated polaroid as RGBA PIL image (done once, reused per frame)
    pol_images = []
    for i in range(total + 1):
        rgb = frames[i % len(frames)]
        pil = _make_polaroid_pil(rgb, pol_size, b_side, b_top, b_bottom)
        rotated = pil.rotate(-angles[i], expand=True, resample=PILImage.BICUBIC)
        pol_images.append(rotated)

    def _paste(canvas_pil, pol, px, py, alpha_scale=1.0):
        """Paste a polaroid PIL image onto canvas_pil with optional alpha scaling."""
        mask = pol.getchannel("A")
        if alpha_scale < 1.0:
            mask = PILImage.fromarray(
                (np.array(mask) * alpha_scale).clip(0, 255).astype(np.uint8)
            )
        canvas_pil.paste(pol.convert("RGB"), (px, py), mask)

    clips = []
    # pile_canvas is a PIL Image that accumulates as we go
    pile_canvas = PILImage.new("RGB", (output_w, output_h), (0, 0, 0))

    for i in range(total):
        if cancel_flag is not None and cancel_flag[0]:
            raise InterruptedError("Job cancelled")

        # Reset pile at the start of each full photo cycle
        if i > 0 and i % len(frames) == 0:
            pile_canvas = PILImage.new("RGB", (output_w, output_h), (0, 0, 0))

        new_pol = pol_images[i]
        rx, ry  = new_pol.size
        px = output_w // 2 - rx // 2 + int(offsets_x[i])
        py = output_h // 2 - ry // 2 + int(offsets_y[i])

        clip_dur = max(beat_times[i + 1] - beat_times[i], 1.0 / fps)

        # "full" = pile_canvas + new polaroid already composited (used by cut & flash)
        full_canvas = pile_canvas.copy()
        _paste(full_canvas, new_pol, px, py)
        full_rgb = np.array(full_canvas)

        if polaroid_entry == "cut":
            clip = ImageClip(full_rgb, duration=clip_dur)

        elif polaroid_entry == "flash":
            _full = full_rgb.copy()
            _fps  = fps

            def _fl_flash(gf, t, _f=_full, _fps=_fps):
                bgr = cv2.cvtColor(_f, cv2.COLOR_RGB2BGR)
                return cv2.cvtColor(_apply_flash(bgr, t, _fps), cv2.COLOR_BGR2RGB)

            clip = ImageClip(full_rgb, duration=clip_dur).fl(_fl_flash)

        elif polaroid_entry == "fade":
            _base = pile_canvas.copy()
            _pol  = new_pol
            _px, _py, _dur = px, py, clip_dur

            def _fl_fade(gf, t, _b=_base, _p=_pol, _px=_px, _py=_py, _d=_dur):
                alpha = min(t / max(_d, 1e-6), 1.0)
                c = _b.copy()
                _paste(c, _p, _px, _py, alpha_scale=alpha)
                return np.array(c)

            clip = ImageClip(full_rgb, duration=clip_dur).fl(_fl_fade)

        else:  # "drop" (default)
            DROP_PX = int(min(output_w, output_h) * 0.18)
            _base = pile_canvas.copy()
            _pol  = new_pol
            _px, _py, _dur = px, py, clip_dur

            def _fl_drop(gf, t, _b=_base, _p=_pol, _px=_px, _py=_py,
                         _d=_dur, _dp=DROP_PX):
                p = min(t / max(_d, 1e-6), 1.0)
                ease = 1.0 - (1.0 - p) ** 2   # ease-out quad
                dy = int(_dp * (1.0 - ease))
                c = _b.copy()
                _paste(c, _p, _px, _py - dy)
                return np.array(c)

            clip = ImageClip(full_rgb, duration=clip_dur).fl(_fl_drop)

        clips.append(clip)

        # Permanently commit this polaroid to the pile canvas for next iteration
        _paste(pile_canvas, new_pol, px, py)

        if i % max(1, total // 10) == 0:
            pct = 35 + int(55 * (i + 1) / total)
            if progress_callback:
                progress_callback(pct, f"Building polaroid clip {i + 1}/{total}")

    return clips


def assemble_video(
    image_paths: list,
    audio_path: str,
    output_path: str,
    output_w: int,
    output_h: int,
    fps: int,
    sensitivity: float,
    transition: str,
    zoom: float,
    max_duration: float = 0,
    polaroid_entry: str = "drop",
    progress_callback=None,
    cancel_flag=None,
) -> dict:
    """
    Full pipeline: face-crop all images -> detect beats -> build clips -> export.

    cancel_flag is a list with one bool element: [False]. Set [True] externally to abort.
    progress_callback(pct: int, msg: str) is called at key steps.

    Returns {"bpm": float, "duration": float, "beats": int, "photos_used": int}.
    """
    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    def _cancelled():
        return cancel_flag is not None and cancel_flag[0]

    _progress(2, "Loading face detection models...")

    # --- 1. Face-crop all images ---
    frames = []
    n = len(image_paths)
    for i, path in enumerate(image_paths):
        if _cancelled():
            raise InterruptedError("Job cancelled")
        bgr, _ = detect_and_crop(path, output_w, output_h, zoom)
        # Convert BGR -> RGB for moviepy
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        pct = 2 + int(28 * (i + 1) / n)
        _progress(pct, f"Processing image {i + 1}/{n}")

    _progress(30, "Detecting beats...")

    # --- 2. Beat detection ---
    beat_times, duration, bpm = detect_beats(audio_path, sensitivity)

    # Ensure we have at least 2 beat timestamps (cycle if needed)
    if len(beat_times) < 2:
        beat_times = [i * (duration / max(len(frames), 1)) for i in range(len(frames) + 1)]

    # Trim beat list to user-requested max duration (0 = full song)
    if max_duration > 0:
        beat_times = [t for t in beat_times if t <= max_duration]
        if len(beat_times) < 2:
            beat_times = [0, max_duration]

    _progress(35, f"Detected {len(beat_times)} beats at {bpm:.0f} BPM")

    # --- 3. Build clips (photos loop to fill all beats) ---
    total_clips = len(beat_times) - 1

    if transition == "polaroid":
        clips = _build_polaroid_clips(
            frames, beat_times, output_w, output_h, fps,
            polaroid_entry, progress_callback, cancel_flag,
        )
    else:
        clips = []
        prev_rgb = None

        for i in range(total_clips):
            if _cancelled():
                raise InterruptedError("Job cancelled")

            rgb = frames[i % len(frames)]
            clip_duration = beat_times[i + 1] - beat_times[i]
            clip_duration = max(clip_duration, 1.0 / fps)

            if transition == "flash" and prev_rgb is not None:
                curr = rgb.copy()
                _fps = fps

                def _fl_flash(get_frame, t, _rgb=curr, _fps=_fps):
                    f = get_frame(t).astype(np.uint8)
                    bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                    bgr = _apply_flash(bgr, t, _fps)
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                clip = ImageClip(rgb, duration=clip_duration).fl(_fl_flash)

            elif transition == "fade" and prev_rgb is not None:
                prev = prev_rgb.copy()
                curr = rgb.copy()
                _dur = clip_duration

                def _fl_fade(get_frame, t, _prev=prev, _curr=curr, _dur=_dur):
                    bgr_prev = cv2.cvtColor(_prev.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    bgr_curr = cv2.cvtColor(_curr.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    blended = _apply_fade(bgr_prev, bgr_curr, t, _dur)
                    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

                clip = ImageClip(rgb, duration=clip_duration).fl(_fl_fade)

            elif transition == "zoom_in":
                curr = rgb.copy()
                _dur = clip_duration

                def _fl_zoom(get_frame, t, _curr=curr, _dur=_dur):
                    bgr = cv2.cvtColor(_curr.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    bgr = _apply_zoom_in(bgr, t, _dur)
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                clip = ImageClip(rgb, duration=clip_duration).fl(_fl_zoom)

            else:  # "cut" or first clip of flash/fade
                clip = ImageClip(rgb, duration=clip_duration)

            clips.append(clip)
            prev_rgb = rgb

            if i % max(1, total_clips // 10) == 0:
                pct = 35 + int(55 * (i + 1) / total_clips)
                _progress(pct, f"Building clip {i + 1}/{total_clips}")

    _progress(90, "Concatenating clips...")

    # --- 4. Concatenate & attach audio ---
    final = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_path).subclip(0, min(final.duration, duration))
    final = final.set_audio(audio)

    _progress(92, "Exporting video (this may take a while)...")

    final.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )

    _progress(99, "Export complete, preparing download...")

    return {
        "bpm": bpm,
        "duration": final.duration,
        "beats": len(beat_times),
        "photos_used": total_clips,
    }
