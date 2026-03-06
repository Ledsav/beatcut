import warnings

import cv2
import librosa
import numpy as np
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips
from PIL import Image as PILImage
from PIL import ImageFilter as PILImageFilter

warnings.filterwarnings("ignore")

# Cascade classifiers loaded once at import time (not per-image)
_CASCADE_FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_CASCADE_PROFILE  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


def detect_and_crop(image_path: str, output_w: int, output_h: int, zoom: float):
    """
    Reads image_path, detects the largest face, centers the crop around
    (cx, cy = face_x + w/2, face_y + h*0.35), crops to output_w x output_h
    at the given zoom level (larger zoom = tighter crop on the face), resizes to
    output dimensions with LANCZOS interpolation.

    Returns (bgr_array, face_found).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    ih, iw = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_found = False
    cx, cy = iw // 2, ih // 2  # fallback: image center

    # Try frontal then left-profile
    for cascade in [_CASCADE_FRONTAL, _CASCADE_PROFILE]:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces):
            # Pick largest by area
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cx = x + w // 2
            cy = int(y + h * 0.35)  # include forehead
            face_found = True
            break

    # Try right-facing profile (mirror the image, then map x back)
    if not face_found:
        gray_flip = cv2.flip(gray, 1)
        faces = _CASCADE_PROFILE.detectMultiScale(gray_flip, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cx = iw - (x + w // 2)
            cy = int(y + h * 0.35)
            face_found = True

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


def detect_sync_points(audio_path: str, sensitivity: float, sync_mode: str):
    """
    Loads audio and returns cut-point times based on sync_mode.

    sync_mode values:
      "beat"   — rhythmic drum beat (librosa beat_track on full audio)
      "drums"  — percussive component beat (HPSS → beat_track on percussive)
      "melody" — harmonic onset detection (HPSS → onset_detect on harmonic)
      "energy" — RMS energy peaks (find_peaks on rms envelope)

    sensitivity: 0.3 → sparse, 2.0 → dense.
    step = max(1, round(1/sensitivity)) applied to all modes.

    Returns (sync_times_list, duration_seconds, bpm).
    """
    from scipy.signal import find_peaks

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    step = max(1, round(1.0 / sensitivity))

    if sync_mode == "beat":
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        sync_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        sync_times = sync_times[::step]
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    elif sync_mode == "drums":
        _, y_perc = librosa.effects.hpss(y)
        tempo, beat_frames = librosa.beat.beat_track(y=y_perc, sr=sr, units="frames")
        sync_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        sync_times = sync_times[::step]
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    elif sync_mode == "melody":
        y_harm, _ = librosa.effects.hpss(y)
        onset_frames = librosa.onset.onset_detect(y=y_harm, sr=sr)
        sync_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
        sync_times = sync_times[::step]
        # Estimate BPM from median inter-onset interval
        if len(sync_times) >= 2:
            bpm = 60.0 / float(np.median(np.diff(sync_times)))
        else:
            bpm = 0.0

    elif sync_mode == "energy":
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        distance = max(1, round(10 / sensitivity))
        peaks, _ = find_peaks(rms, distance=distance)
        sync_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length).tolist()
        sync_times = sync_times[::step]
        if len(sync_times) >= 2:
            bpm = 60.0 / float(np.median(np.diff(sync_times)))
        else:
            bpm = 0.0
    else:
        raise ValueError(f"Unknown sync_mode: {sync_mode!r}. Expected one of: beat, drums, melody, energy")

    return sync_times, duration, bpm


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

    Each beat adds one new polaroid on a sliding window of max len(frames) visible
    polaroids. When the window is full the oldest is dropped rather than resetting,
    so the pile always looks populated. Shadow is precomputed once per polaroid and
    reused every frame — no per-frame blur cost.

    Landing styles: drop | flash | fade | cut
    """
    total = len(beat_times) - 1
    if total == 0:
        return []

    pol_size = int(min(output_w, output_h) * 0.52)
    b_side   = max(int(pol_size * 0.07), 8)
    b_top    = max(int(pol_size * 0.07), 8)
    b_bottom = max(int(pol_size * 0.26), 24)

    rng = np.random.default_rng(seed=7)
    angles    = rng.uniform(-14, 14,           total + 1).tolist()
    offsets_x = rng.integers(-output_w // 14, output_w // 14, total + 1).tolist()
    offsets_y = rng.integers(-output_h // 14, output_h // 14, total + 1).tolist()

    SHADOW_DX   = max(int(pol_size * 0.012), 5)
    SHADOW_DY   = max(int(pol_size * 0.016), 7)
    SHADOW_BLUR = max(int(pol_size * 0.018), 8)
    DROP_PX     = int(min(output_w, output_h) * 0.18)

    # ── Pre-render: rotated polaroid RGBA + pre-blurred shadow mask (once per slot) ──
    pol_data = []   # list of (rotated_rgba_pil, blurred_shadow_mask_L_pil)
    for i in range(total + 1):
        rgb     = frames[i % len(frames)]
        pil     = _make_polaroid_pil(rgb, pol_size, b_side, b_top, b_bottom)
        rotated = pil.rotate(-angles[i], expand=True, resample=PILImage.BICUBIC)
        alpha   = rotated.getchannel("A")
        shadow  = alpha.point(lambda p: int(p * 0.55))
        shadow  = shadow.filter(PILImageFilter.GaussianBlur(radius=SHADOW_BLUR))
        pol_data.append((rotated, shadow))

    # ── Paste helper: shadow first, polaroid on top ──
    def _paste_pol(canvas, idx, px, py, alpha_scale=1.0, dy_offset=0):
        pol, shad = pol_data[idx]
        black = PILImage.new("RGB", pol.size, (0, 0, 0))
        if alpha_scale < 1.0:
            s = PILImage.fromarray((np.array(shad) * alpha_scale).clip(0, 255).astype(np.uint8))
            p = PILImage.fromarray((np.array(pol.getchannel("A")) * alpha_scale).clip(0, 255).astype(np.uint8))
        else:
            s, p = shad, pol.getchannel("A")
        canvas.paste(black, (px + SHADOW_DX, py + dy_offset + SHADOW_DY), s)
        canvas.paste(pol.convert("RGB"), (px, py + dy_offset), p)

    # ── Render a full pile from a list of (idx, px, py) slots ──
    def _render_pile(slots):
        c = PILImage.new("RGB", (output_w, output_h), (0, 0, 0))
        for idx, px, py in slots:
            _paste_pol(c, idx, px, py)
        return c

    # ── Sliding window state ──
    max_pile   = min(len(frames), 10)
    pile_slots = []          # list of (idx, px, py) — oldest at index 0
    pile_canvas = PILImage.new("RGB", (output_w, output_h), (0, 0, 0))

    clips = []

    for i in range(total):
        if cancel_flag is not None and cancel_flag[0]:
            raise InterruptedError("Job cancelled")

        pol, _ = pol_data[i]
        rx, ry = pol.size
        px = output_w // 2 - rx // 2 + int(offsets_x[i])
        py = output_h // 2 - ry // 2 + int(offsets_y[i])

        clip_dur = max(beat_times[i + 1] - beat_times[i], 1.0 / fps)

        # base = pile *before* this new photo arrives (used as animation start)
        _base = pile_canvas   # reference; animation lambdas capture .copy()

        # full = pile + new photo (static end-state, used by cut & flash)
        full_c = pile_canvas.copy()
        _paste_pol(full_c, i, px, py)
        full_rgb = np.array(full_c)

        _i, _px, _py = i, px, py

        if polaroid_entry == "cut":
            clip = ImageClip(full_rgb, duration=clip_dur)

        elif polaroid_entry == "flash":
            _f, _fps2 = full_rgb.copy(), fps
            def _fl_flash(gf, t, _f=_f, _fps=_fps2):
                bgr = cv2.cvtColor(_f, cv2.COLOR_RGB2BGR)
                return cv2.cvtColor(_apply_flash(bgr, t, _fps), cv2.COLOR_BGR2RGB)
            clip = ImageClip(full_rgb, duration=clip_dur).fl(_fl_flash)

        elif polaroid_entry == "fade":
            _b = _base.copy()
            def _fl_fade(gf, t, _b=_b, _i=_i, _px=_px, _py=_py, _d=clip_dur):
                alpha = min(t / max(_d, 1e-6), 1.0)
                c = _b.copy()
                _paste_pol(c, _i, _px, _py, alpha_scale=alpha)
                return np.array(c)
            clip = ImageClip(full_rgb, duration=clip_dur).fl(_fl_fade)

        else:  # drop (default)
            _b = _base.copy()
            def _fl_drop(gf, t, _b=_b, _i=_i, _px=_px, _py=_py,
                         _d=clip_dur, _dp=DROP_PX):
                p    = min(t / max(_d, 1e-6), 1.0)
                ease = 1.0 - (1.0 - p) ** 2    # ease-out quad
                dy   = int(_dp * (1.0 - ease))
                c    = _b.copy()
                _paste_pol(c, _i, _px, _py, dy_offset=-dy)
                return np.array(c)
            clip = ImageClip(full_rgb, duration=clip_dur).fl(_fl_drop)

        clips.append(clip)

        # ── Update sliding window ──
        if len(pile_slots) >= max_pile:
            # Drop oldest, rerender from scratch (O(max_pile) paste ops)
            pile_slots.pop(0)
            pile_slots.append((i, px, py))
            pile_canvas = _render_pile(pile_slots)
        else:
            # Still growing — just paste on top (O(1))
            pile_slots.append((i, px, py))
            pile_canvas = pile_canvas.copy()
            _paste_pol(pile_canvas, i, px, py)

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
    start_time: float = 0,
    end_time: float = 0,
    sync_mode: str = "beat",
    polaroid_entry: str = "drop",
    progress_callback=None,
    cancel_flag=None,
) -> dict:
    """
    Full pipeline: face-crop all images -> detect sync points -> build clips -> export.

    sync_mode controls audio analysis: "beat", "drums", "melody", or "energy".

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
        bgr, face_found = detect_and_crop(path, output_w, output_h, zoom)
        # Convert BGR -> RGB for moviepy
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        pct = 2 + int(28 * (i + 1) / n)
        label = f"Processing image {i + 1}/{n}" if face_found else f"Processing image {i + 1}/{n} (no face detected, using center)"
        _progress(pct, label)

    _progress(30, "Detecting beats...")

    # --- 2. Beat detection ---
    beat_times, duration, bpm = detect_sync_points(audio_path, sensitivity, sync_mode)

    # Resolve selected time window (end_time=0 means full song)
    t_start = max(0.0, start_time)
    t_end   = end_time if end_time > 0 else duration

    # Trim beat list to selected window, then normalise to 0-based
    span = t_end - t_start
    beat_times = [t for t in beat_times if t_start <= t <= t_end]
    beat_times = [t - t_start for t in beat_times]

    # Always append the selection end as a sentinel so the last clip
    # runs all the way to the end of the chosen section (not just to
    # the last detected beat, which is almost always before the end).
    if not beat_times or beat_times[-1] < span - 1.0 / fps:
        beat_times.append(span)

    # Ensure we have at least 2 beat timestamps
    if len(beat_times) < 2:
        beat_times = [i * (span / max(len(frames), 1)) for i in range(len(frames) + 1)]

    _progress(35, f"Detected {len(beat_times)} sync points ({sync_mode} mode) at {bpm:.0f} BPM")

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
    clip_span = min(final.duration, t_end - t_start)
    audio = AudioFileClip(audio_path).subclip(t_start, t_start + clip_span)
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
