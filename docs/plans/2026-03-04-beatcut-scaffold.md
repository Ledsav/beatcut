# BeatCut Full-Stack App Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local Flask web app where the user uploads photos + a song and receives a beat-synced, face-centered MP4 slideshow.

**Architecture:** A Python Flask server handles file uploads, spawns a background thread for video processing (face detection → beat detection → moviepy assembly), and pushes SSE progress events to a vanilla-JS frontend. The frontend walks the user through 4 stages: upload/configure → processing → done → error.

**Tech Stack:** Python 3.9+, Flask, OpenCV, librosa, moviepy 1.0.3, numpy, Pillow, soundfile, ffmpeg (system), vanilla JS, Google Fonts (Bebas Neue + Space Mono).

---

### Task 1: Project Skeleton & Requirements

**Files:**
- Create: `requirements.txt`
- Create: `README.md`

**Step 1: Create and activate a virtual environment**

```bash
python -m venv .venv
# Windows (Git Bash / PowerShell)
source .venv/Scripts/activate
# macOS / Linux
# source .venv/bin/activate
```

Expected: prompt prefix changes to `(.venv)`.

Add `.venv/` to `.gitignore`:

```bash
echo ".venv/" >> .gitignore
git add .gitignore
```

**Step 2: Write requirements.txt**

```
flask>=2.3
opencv-python>=4.8
librosa>=0.10
moviepy==1.0.3
numpy>=1.24
Pillow>=10.0
soundfile>=0.12
```

**Step 2: Write README.md**

```markdown
# BeatCut

Beat-synced, face-centered slideshow video generator.

## Prerequisites

- Python 3.9+
- ffmpeg installed on your system

### Install ffmpeg

**Windows:**
```
winget install ffmpeg
# or
choco install ffmpeg
```

**macOS:**
```
brew install ffmpeg
```

**Linux (Debian/Ubuntu):**
```
sudo apt update && sudo apt install ffmpeg
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000 in your browser.

## Settings

| Setting | Description |
|---|---|
| **Transition** | Flash: white flash between clips. Hard Cut: instant switch. Fade: cross-dissolve. Zoom In: scale 1.3→1.0 into next clip. |
| **Beat Sensitivity** | 0.3 = every beat, 2.0 = only major beats. Controls `step = max(1, round(1/sensitivity))` subsampling. |
| **Resolution** | Output dimensions. 1080×1080 for square social, 1920×1080 for widescreen, 1080×1920 for vertical/Reels. |
| **Face Zoom** | 1.0 = tight face crop, 2.0 = zoomed-out context. Divides the crop window by this factor. |

## Troubleshooting

| Error | Fix |
|---|---|
| `ffmpeg not found` | Install ffmpeg and ensure it's in your PATH. Run `ffmpeg -version` to confirm. |
| `No faces detected` | App falls back to image center. Check that photos contain forward-facing or profile faces. |
| `Audio decode error` | Ensure the audio file is a valid MP3, WAV, FLAC, or OGG. Re-export from your DAW if corrupt. |
| `moviepy version mismatch` | Run `pip install moviepy==1.0.3` — newer versions break the API used here. |
| `SSE stream hangs` | Ensure Flask is running with `threaded=True` (it is by default in `app.run()`). |
```

**Step 4: Install dependencies into the venv**

```bash
pip install -r requirements.txt
```

**Step 5: Commit**

```bash
git add requirements.txt README.md .gitignore
git commit -m "chore: add requirements, README, and venv gitignore"
```

---

### Task 2: processor.py — Face Detection & Smart Crop

**Files:**
- Create: `processor.py`

**Step 1: Write the face-detection function**

```python
import cv2
import numpy as np
from PIL import Image

# Cascade paths (shipped with opencv-python)
_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_PROFILE  = cv2.data.haarcascades + "haarcascade_profileface.xml"


def detect_and_crop(image_path: str, output_w: int, output_h: int, zoom: float) -> tuple[np.ndarray, bool]:
    """
    Reads image_path, detects the largest face, centers the crop around
    (cx, cy = face_x + w/2, face_y + h*0.35), crops to output_w × output_h
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
```

**Step 2: Verify manually (no test framework needed yet)**

```bash
python -c "
from processor import detect_and_crop
import cv2, numpy as np
# Create a tiny white test image (no face → falls back to center)
img = np.ones((400,400,3), dtype=np.uint8)*200
cv2.imwrite('/tmp/test_face.jpg', img)
out, found = detect_and_crop('/tmp/test_face.jpg', 1080, 1080, 1.5)
print('shape:', out.shape, '| face_found:', found)
assert out.shape == (1080, 1080, 3)
print('PASS')
"
```

Expected: `shape: (1080, 1080, 3) | face_found: False` then `PASS`.

**Step 3: Commit**

```bash
git add processor.py
git commit -m "feat: add face detection and smart crop to processor"
```

---

### Task 3: processor.py — Beat Detection

**Files:**
- Modify: `processor.py` (append)

**Step 1: Add beat_detection function**

```python
import librosa
import warnings
warnings.filterwarnings("ignore")  # suppress audioread/librosa warnings


def detect_beats(audio_path: str, sensitivity: float) -> tuple[list[float], float, float]:
    """
    Loads audio, runs beat tracking, subsamples by sensitivity.

    sensitivity: 0.3 → step=3 (many beats), 2.0 → step=1 (all beats).
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
```

**Step 2: Smoke-test beat detection**

```bash
python -c "
from processor import detect_beats
# Use any short mp3 or wav you have; substitute path below
times, dur, bpm = detect_beats('path/to/test.mp3', 1.0)
print(f'beats: {len(times)}, duration: {dur:.1f}s, bpm: {bpm:.1f}')
"
```

Expected: non-zero beat count, positive duration and BPM.

**Step 3: Commit**

```bash
git add processor.py
git commit -m "feat: add beat detection to processor"
```

---

### Task 4: processor.py — Transition Helpers

**Files:**
- Modify: `processor.py` (append)

**Step 1: Add transition helpers**

```python
def _apply_flash(frame: np.ndarray, t: float, fps: int) -> np.ndarray:
    """White flash overlay decaying by 0.72 per frame."""
    alpha = 0.9 * (0.72 ** (t * fps))
    white = np.ones_like(frame, dtype=np.uint8) * 255
    return cv2.addWeighted(frame, 1 - alpha, white, alpha, 0)


def _apply_fade(frame_a: np.ndarray, frame_b: np.ndarray, t: float, duration: float) -> np.ndarray:
    """Linear cross-dissolve between two BGR frames."""
    ratio = t / max(duration, 1e-6)
    return cv2.addWeighted(frame_a, 1 - ratio, frame_b, ratio, 0)


def _apply_zoom_in(frame: np.ndarray, t: float, duration: float) -> np.ndarray:
    """Scale 1.3→1.0 over the clip duration using warpAffine."""
    h, w = frame.shape[:2]
    scale = 1.3 - 0.3 * (t / max(duration, 1e-6))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
```

**Step 2: No formal test needed — these are pure numpy ops.**

**Step 3: Commit**

```bash
git add processor.py
git commit -m "feat: add transition helper functions to processor"
```

---

### Task 5: processor.py — Video Assembly

**Files:**
- Modify: `processor.py` (append)

**Step 1: Add the main assemble_video function**

```python
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips


def assemble_video(
    image_paths: list[str],
    audio_path: str,
    output_path: str,
    output_w: int,
    output_h: int,
    fps: int,
    sensitivity: float,
    transition: str,
    zoom: float,
    progress_callback=None,
    cancel_flag: list = None,
) -> dict:
    """
    Full pipeline: face-crop all images → detect beats → build clips → export.

    cancel_flag is a list with one bool element: [False]. Set [True] externally to abort.
    progress_callback(pct: int, msg: str) is called at key steps.

    Returns {"bpm": float, "duration": float, "beats": int, "photos_used": int}.
    """
    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    def _cancelled():
        return cancel_flag is not None and cancel_flag[0]

    _progress(2, "Loading face detection models…")

    # --- 1. Face-crop all images ---
    frames = []
    n = len(image_paths)
    for i, path in enumerate(image_paths):
        if _cancelled():
            raise InterruptedError("Job cancelled")
        bgr, _ = detect_and_crop(path, output_w, output_h, zoom)
        # Convert BGR → RGB for moviepy
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        pct = 2 + int(28 * (i + 1) / n)
        _progress(pct, f"Processing image {i+1}/{n}")

    _progress(30, "Detecting beats…")

    # --- 2. Beat detection ---
    beat_times, duration, bpm = detect_beats(audio_path, sensitivity)

    # Ensure we have at least as many beat intervals as frames (cycle if needed)
    if len(beat_times) < 2:
        beat_times = [i * (duration / max(len(frames), 1)) for i in range(len(frames) + 1)]

    _progress(35, f"Detected {len(beat_times)} beats at {bpm:.0f} BPM")

    # --- 3. Build clips ---
    clips = []
    total_clips = min(len(frames), len(beat_times) - 1)
    prev_rgb = None

    for i in range(total_clips):
        if _cancelled():
            raise InterruptedError("Job cancelled")

        rgb = frames[i % len(frames)]
        clip_duration = beat_times[i + 1] - beat_times[i]
        clip_duration = max(clip_duration, 1.0 / fps)

        if transition == "flash" and prev_rgb is not None:
            # Build frames with decaying white overlay at clip start
            def make_frame_flash(t, _rgb=rgb, _fps=fps):
                frame = _rgb.copy()
                return _apply_flash(cv2.cvtColor(
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB
                ), t, _fps)
            # Use make_frame lambda — note moviepy expects RGB
            clip = ImageClip(rgb, duration=clip_duration)
            # Overlay flash via fl_time (simpler: just use static ImageClip for now,
            # flash effect applied per-pixel below)
            def _fl(get_frame, t, _rgb=rgb.copy()):
                f = get_frame(t)
                bgr = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2BGR)
                bgr = _apply_flash(bgr, t, fps)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            clip = clip.fl(_fl)

        elif transition == "fade" and prev_rgb is not None:
            prev = prev_rgb
            curr = rgb

            def _fl_fade(get_frame, t, _prev=prev.copy(), _curr=curr.copy(), _dur=clip_duration):
                bgr_prev = cv2.cvtColor(_prev.astype(np.uint8), cv2.COLOR_RGB2BGR)
                bgr_curr = cv2.cvtColor(_curr.astype(np.uint8), cv2.COLOR_RGB2BGR)
                blended = _apply_fade(bgr_prev, bgr_curr, t, _dur)
                return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

            clip = ImageClip(rgb, duration=clip_duration).fl(_fl_fade)

        elif transition == "zoom_in":
            curr = rgb

            def _fl_zoom(get_frame, t, _curr=curr.copy(), _dur=clip_duration):
                bgr = cv2.cvtColor(_curr.astype(np.uint8), cv2.COLOR_RGB2BGR)
                bgr = _apply_zoom_in(bgr, t, _dur)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            clip = ImageClip(rgb, duration=clip_duration).fl(_fl_zoom)

        else:  # "cut" or fallback
            clip = ImageClip(rgb, duration=clip_duration)

        clips.append(clip)
        prev_rgb = rgb

        if i % max(1, total_clips // 10) == 0:
            pct = 35 + int(55 * (i + 1) / total_clips)
            _progress(pct, f"Building clip {i+1}/{total_clips}")

    _progress(90, "Concatenating clips…")

    # --- 4. Concatenate & attach audio ---
    final = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_path).subclip(0, min(final.duration, duration))
    final = final.set_audio(audio)

    _progress(92, "Exporting video (this may take a while)…")

    final.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )

    _progress(100, "done")

    return {
        "bpm": bpm,
        "duration": final.duration,
        "beats": len(beat_times),
        "photos_used": total_clips,
    }
```

**Step 2: Smoke-test with dummy images (skip if no audio file handy)**

```bash
python -c "
import numpy as np, cv2, os, tempfile
from processor import assemble_video

# Create 3 test images
tmpdir = tempfile.mkdtemp()
paths = []
for i in range(3):
    img = np.random.randint(0,255,(400,400,3), dtype=np.uint8)
    p = f'{tmpdir}/img{i}.jpg'
    cv2.imwrite(p, img)
    paths.append(p)

# You need a real audio file for this test
# assemble_video(paths, 'test.mp3', f'{tmpdir}/out.mp4', 1080, 1080, 24, 1.0, 'cut', 1.0, print)
print('Imports OK')
"
```

Expected: `Imports OK` with no ImportError.

**Step 3: Commit**

```bash
git add processor.py
git commit -m "feat: add video assembly pipeline to processor"
```

---

### Task 6: app.py — Flask App, Job State & Background Thread

**Files:**
- Create: `app.py`

**Step 1: Write app.py skeleton with job state and /upload route**

```python
import os
import uuid
import threading
import tempfile
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_file, render_template, stream_with_context
import json

from processor import assemble_video

app = Flask(__name__)

# ── Job state ────────────────────────────────────────────────────────────────
jobs: dict[str, dict] = {}
# Each job:
# {
#   "status": "running" | "done" | "error" | "cancelled",
#   "progress": 0-100,
#   "message": str,
#   "output_path": str | None,
#   "cancel_flag": [False],   # mutable list so thread can check it
#   "thread": Thread,
#   "workdir": str,
#   "events": list[str],      # SSE event queue
#   "stats": dict | None,
# }

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    job_id = str(uuid.uuid4())
    workdir = tempfile.mkdtemp(prefix=f"beatcut_{job_id}_")

    # Save images
    images = request.files.getlist("images")
    image_paths = []
    for f in images:
        ext = Path(f.filename).suffix.lower()
        if ext in ALLOWED_IMAGE_EXTS:
            dest = os.path.join(workdir, f"img_{len(image_paths)}{ext}")
            f.save(dest)
            image_paths.append(dest)

    if not image_paths:
        shutil.rmtree(workdir, ignore_errors=True)
        return jsonify({"error": "No valid images uploaded"}), 400

    # Save music
    music_file = request.files.get("music")
    if not music_file:
        shutil.rmtree(workdir, ignore_errors=True)
        return jsonify({"error": "No music file uploaded"}), 400
    audio_ext = Path(music_file.filename).suffix.lower()
    audio_path = os.path.join(workdir, f"audio{audio_ext}")
    music_file.save(audio_path)

    # Settings
    resolution = request.form.get("resolution", "1080x1080")
    w, h = (int(x) for x in resolution.lower().split("x"))
    fps = int(request.form.get("fps", 24))
    sensitivity = float(request.form.get("sensitivity", 1.0))
    transition = request.form.get("transition", "flash")
    zoom = float(request.form.get("zoom", 1.5))

    output_path = os.path.join(workdir, "output.mp4")
    cancel_flag = [False]

    job = {
        "status": "running",
        "progress": 0,
        "message": "Starting…",
        "output_path": output_path,
        "cancel_flag": cancel_flag,
        "thread": None,
        "workdir": workdir,
        "events": [],
        "stats": None,
    }
    jobs[job_id] = job

    def run():
        def on_progress(pct, msg):
            job["progress"] = pct
            job["message"] = msg
            job["events"].append(json.dumps({"pct": pct, "msg": msg}))

        try:
            stats = assemble_video(
                image_paths=image_paths,
                audio_path=audio_path,
                output_path=output_path,
                output_w=w,
                output_h=h,
                fps=fps,
                sensitivity=sensitivity,
                transition=transition,
                zoom=zoom,
                progress_callback=on_progress,
                cancel_flag=cancel_flag,
            )
            job["stats"] = stats
            job["status"] = "done"
            job["events"].append(json.dumps({
                "pct": 100,
                "msg": "done",
                "file": f"/download/{job_id}",
                "stats": stats,
            }))
        except InterruptedError:
            job["status"] = "cancelled"
            job["events"].append(json.dumps({"pct": 0, "msg": "cancelled"}))
        except Exception as e:
            job["status"] = "error"
            job["events"].append(json.dumps({"error": str(e)}))

    t = threading.Thread(target=run, daemon=True)
    job["thread"] = t
    t.start()

    return jsonify({"job_id": job_id})
```

**Step 2: Verify Flask starts without import errors**

```bash
python -c "import app; print('Flask app imports OK')"
```

Expected: `Flask app imports OK`

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Flask app skeleton with upload route and job management"
```

---

### Task 7: app.py — SSE Progress, Download & Cancel Routes

**Files:**
- Modify: `app.py` (append routes before `if __name__ == "__main__":`)

**Step 1: Add SSE, download, and cancel routes**

```python
@app.route("/progress/<job_id>")
def progress(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        last = 0
        import time
        while True:
            events = job["events"]
            while last < len(events):
                yield f"data: {events[last]}\n\n"
                last += 1
            if job["status"] in ("done", "error", "cancelled"):
                break
            time.sleep(0.2)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404

    path = job["output_path"]
    if not os.path.exists(path):
        return jsonify({"error": "Output file missing"}), 500

    def cleanup():
        import time
        time.sleep(5)
        shutil.rmtree(job.get("workdir", ""), ignore_errors=True)
        jobs.pop(job_id, None)

    threading.Thread(target=cleanup, daemon=True).start()

    return send_file(path, as_attachment=True, download_name="beatcut_output.mp4")


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    job["cancel_flag"][0] = True
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
```

**Step 2: Test routes exist**

```bash
python -c "
from app import app
rules = [str(r) for r in app.url_map.iter_rules()]
for r in ['/progress/<job_id>', '/download/<job_id>', '/cancel/<job_id>']:
    assert any(r in rule for rule in rules), f'Missing route: {r}'
print('All routes present — PASS')
"
```

Expected: `All routes present — PASS`

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add SSE progress, download and cancel routes"
```

---

### Task 8: templates/index.html — HTML Structure

**Files:**
- Create: `templates/index.html`

**Step 1: Write full HTML**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>BeatCut</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>

  <!-- ── Stage 1: Upload & Configure ─────────────────────────── -->
  <section id="stage-upload" class="stage active">
    <h1 class="logo">BEATCUT</h1>
    <p class="tagline">beat-synced · face-centered · export-ready</p>

    <div class="upload-row">
      <!-- Photos -->
      <div class="upload-zone" id="zone-photos">
        <label for="input-photos" class="upload-label">
          <span class="upload-icon">🖼</span>
          <span class="upload-title">Drop Photos</span>
          <span class="upload-hint">click or drag</span>
        </label>
        <input type="file" id="input-photos" multiple accept="image/*" style="opacity:0;position:absolute;width:0;height:0;" />
        <p class="upload-status" id="photos-status">No photos selected</p>
      </div>

      <!-- Music -->
      <div class="upload-zone" id="zone-music">
        <label for="input-music" class="upload-label">
          <span class="upload-icon">🎵</span>
          <span class="upload-title">Drop Music</span>
          <span class="upload-hint">click or drag</span>
        </label>
        <input type="file" id="input-music" accept="audio/*" style="opacity:0;position:absolute;width:0;height:0;" />
        <p class="upload-status" id="music-status">No file selected</p>
      </div>
    </div>

    <!-- Settings -->
    <div class="settings-grid">
      <div class="setting-item">
        <label for="sel-transition">Transition</label>
        <select id="sel-transition">
          <option value="flash">Flash</option>
          <option value="cut">Hard Cut</option>
          <option value="fade">Fade</option>
          <option value="zoom_in">Zoom In</option>
        </select>
      </div>

      <div class="setting-item">
        <label for="range-sensitivity">Beat Sensitivity <span id="val-sensitivity">1.0</span></label>
        <input type="range" id="range-sensitivity" min="0.3" max="2" step="0.1" value="1.0" />
      </div>

      <div class="setting-item">
        <label for="sel-resolution">Resolution</label>
        <select id="sel-resolution">
          <option value="1080x1080">1080×1080 (Square)</option>
          <option value="1920x1080">1920×1080 (Wide)</option>
          <option value="1080x1920">1080×1920 (Vertical)</option>
          <option value="720x720">720×720 (Small)</option>
        </select>
      </div>

      <div class="setting-item">
        <label for="range-zoom">Face Zoom <span id="val-zoom">1.50</span></label>
        <input type="range" id="range-zoom" min="1.0" max="2.0" step="0.05" value="1.5" />
      </div>
    </div>

    <button id="btn-process" class="btn-primary" disabled>Process Video</button>
  </section>

  <!-- ── Stage 2: Processing ───────────────────────────────── -->
  <section id="stage-processing" class="stage">
    <h2>Processing…</h2>
    <div class="progress-track">
      <div class="progress-bar" id="progress-bar" style="width:0%"></div>
    </div>
    <p class="progress-label" id="progress-label">Starting…</p>
    <div class="log-box" id="log-box"></div>
    <button id="btn-cancel" class="btn-danger">Cancel</button>
  </section>

  <!-- ── Stage 3: Done ─────────────────────────────────────── -->
  <section id="stage-done" class="stage">
    <h2>Done!</h2>
    <div class="stats-grid" id="stats-grid">
      <div class="stat-card"><span class="stat-val" id="stat-photos">—</span><span class="stat-lbl">Photos Used</span></div>
      <div class="stat-card"><span class="stat-val" id="stat-beats">—</span><span class="stat-lbl">Beats</span></div>
      <div class="stat-card"><span class="stat-val" id="stat-bpm">—</span><span class="stat-lbl">BPM</span></div>
      <div class="stat-card"><span class="stat-val" id="stat-dur">—</span><span class="stat-lbl">Duration</span></div>
    </div>
    <a id="btn-download" class="btn-primary" href="#">Download MP4</a>
    <button id="btn-again" class="btn-secondary">Make Another</button>
  </section>

  <!-- ── Stage 4: Error ────────────────────────────────────── -->
  <section id="stage-error" class="stage">
    <h2>Something went wrong</h2>
    <p class="error-msg" id="error-msg"></p>
    <button id="btn-retry" class="btn-danger">Try Again</button>
  </section>

  <script src="/static/app.js"></script>
</body>
</html>
```

**Step 2: Start Flask and check the page loads**

```bash
python app.py &
sleep 2
curl -s http://localhost:5000 | grep -c "BeatCut"
kill %1
```

Expected: `1` (the title tag appears).

**Step 3: Commit**

```bash
git add templates/index.html
git commit -m "feat: add HTML structure for all 4 stages"
```

---

### Task 9: static/style.css — Dark Theme Styles

**Files:**
- Create: `static/style.css`

**Step 1: Write full CSS**

```css
/* ── Variables ─────────────────────────────────────── */
:root {
  --bg: #060608;
  --surface: #0e0e12;
  --surface2: #16161e;
  --accent: #c8ff00;
  --danger: #ff3c6e;
  --text: #e8e8f0;
  --muted: #6b6b80;
  --radius: 12px;
}

/* ── Reset & base ──────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Space Mono', monospace;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}

/* ── Stage system ──────────────────────────────────── */
.stage {
  display: none;
  flex-direction: column;
  align-items: center;
  gap: 24px;
  width: 100%;
  max-width: 860px;
  animation: fadeIn 0.3s ease;
}
.stage.active { display: flex; }

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Logo & tagline ────────────────────────────────── */
.logo {
  font-family: 'Bebas Neue', sans-serif;
  font-size: clamp(56px, 10vw, 96px);
  letter-spacing: 0.08em;
  color: var(--accent);
  line-height: 1;
}
.tagline {
  font-size: 13px;
  color: var(--muted);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-top: -16px;
}

/* ── Upload zones ──────────────────────────────────── */
.upload-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  width: 100%;
}

.upload-zone {
  position: relative;
  background: var(--surface);
  border: 2px dashed var(--surface2);
  border-radius: var(--radius);
  padding: 32px 24px;
  text-align: center;
  transition: border-color 0.2s, background 0.2s;
  cursor: pointer;
}
.upload-zone:hover,
.upload-zone.drag-over {
  border-color: var(--accent);
  background: #0e0e1a;
}

.upload-label {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}
.upload-icon { font-size: 32px; }
.upload-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 22px;
  letter-spacing: 0.05em;
}
.upload-hint { font-size: 12px; color: var(--muted); }
.upload-status {
  margin-top: 12px;
  font-size: 12px;
  color: var(--accent);
  min-height: 16px;
}

/* ── Settings grid ─────────────────────────────────── */
.settings-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  width: 100%;
}

.setting-item {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.setting-item label {
  font-size: 12px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

select {
  background: var(--surface2);
  color: var(--text);
  border: 1px solid #2a2a38;
  border-radius: 8px;
  padding: 10px 14px;
  font-family: 'Space Mono', monospace;
  font-size: 14px;
  cursor: pointer;
  appearance: none;
  -webkit-appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%236b6b80' stroke-width='2' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 14px center;
  padding-right: 36px;
}
select:focus { outline: none; border-color: var(--accent); }

input[type="range"] {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: var(--surface2);
  cursor: pointer;
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
  box-shadow: 0 0 0 3px rgba(200,255,0,0.15);
}
input[type="range"]::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border: none;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
}

/* ── Buttons ───────────────────────────────────────── */
.btn-primary, .btn-secondary, .btn-danger {
  font-family: 'Bebas Neue', sans-serif;
  letter-spacing: 0.08em;
  font-size: 20px;
  padding: 14px 48px;
  border-radius: var(--radius);
  border: none;
  cursor: pointer;
  text-decoration: none;
  text-align: center;
  transition: opacity 0.15s, transform 0.1s;
}
.btn-primary:hover:not(:disabled), .btn-secondary:hover, .btn-danger:hover {
  opacity: 0.85;
  transform: translateY(-1px);
}
.btn-primary {
  background: var(--accent);
  color: #000;
}
.btn-primary:disabled {
  opacity: 0.3;
  cursor: not-allowed;
  transform: none;
}
.btn-secondary {
  background: var(--surface);
  color: var(--text);
  border: 1px solid #2a2a38;
}
.btn-danger {
  background: var(--danger);
  color: #fff;
}

/* ── Progress ──────────────────────────────────────── */
.progress-track {
  width: 100%;
  height: 12px;
  background: var(--surface);
  border-radius: 6px;
  overflow: hidden;
}
.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, var(--accent) 0%, #a0cc00 100%);
  border-radius: 6px;
  transition: width 0.4s ease;
}
.progress-label {
  font-size: 13px;
  color: var(--muted);
  align-self: flex-start;
}

/* ── Log box ───────────────────────────────────────── */
.log-box {
  width: 100%;
  height: 180px;
  overflow-y: auto;
  background: var(--surface);
  border-radius: var(--radius);
  padding: 14px 18px;
  font-size: 12px;
  color: var(--muted);
  line-height: 1.8;
  scroll-behavior: smooth;
}
.log-box .log-line { display: block; }

/* ── Stats grid ────────────────────────────────────── */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  width: 100%;
}
.stat-card {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 20px 16px;
  text-align: center;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.stat-val {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 36px;
  color: var(--accent);
  letter-spacing: 0.04em;
}
.stat-lbl {
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

/* ── Error ─────────────────────────────────────────── */
.error-msg {
  color: var(--danger);
  font-size: 14px;
  text-align: center;
  max-width: 520px;
}

/* ── h2 in stages ──────────────────────────────────── */
h2 {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 48px;
  letter-spacing: 0.06em;
  color: var(--text);
}

/* ── Responsive ────────────────────────────────────── */
@media (max-width: 600px) {
  .upload-row, .settings-grid { grid-template-columns: 1fr; }
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
}
```

**Step 2: Commit**

```bash
git add static/style.css
git commit -m "feat: add dark theme CSS for all stages"
```

---

### Task 10: static/app.js — Vanilla JS Logic

**Files:**
- Create: `static/app.js`

**Step 1: Write full app.js**

```javascript
/* ── State ──────────────────────────────────────────── */
let currentJobId = null;
let eventSource  = null;

/* ── DOM refs ───────────────────────────────────────── */
const stageUpload     = document.getElementById('stage-upload');
const stageProcessing = document.getElementById('stage-processing');
const stageDone       = document.getElementById('stage-done');
const stageError      = document.getElementById('stage-error');

const inputPhotos  = document.getElementById('input-photos');
const inputMusic   = document.getElementById('input-music');
const photosStatus = document.getElementById('photos-status');
const musicStatus  = document.getElementById('music-status');
const btnProcess   = document.getElementById('btn-process');
const btnCancel    = document.getElementById('btn-cancel');
const btnAgain     = document.getElementById('btn-again');
const btnRetry     = document.getElementById('btn-retry');
const btnDownload  = document.getElementById('btn-download');

const progressBar   = document.getElementById('progress-bar');
const progressLabel = document.getElementById('progress-label');
const logBox        = document.getElementById('log-box');
const errorMsg      = document.getElementById('error-msg');

const rangeSensitivity = document.getElementById('range-sensitivity');
const valSensitivity   = document.getElementById('val-sensitivity');
const rangeZoom        = document.getElementById('range-zoom');
const valZoom          = document.getElementById('val-zoom');

const statPhotos = document.getElementById('stat-photos');
const statBeats  = document.getElementById('stat-beats');
const statBpm    = document.getElementById('stat-bpm');
const statDur    = document.getElementById('stat-dur');

/* ── Stage helpers ──────────────────────────────────── */
function showStage(stage) {
  [stageUpload, stageProcessing, stageDone, stageError].forEach(s => s.classList.remove('active'));
  stage.classList.add('active');
}

/* ── File inputs ─────────────────────────────────────── */
function updateProcessButton() {
  const hasPhotos = inputPhotos.files && inputPhotos.files.length > 0;
  const hasMusic  = inputMusic.files  && inputMusic.files.length  > 0;
  btnProcess.disabled = !(hasPhotos && hasMusic);
}

inputPhotos.addEventListener('change', () => {
  const n = inputPhotos.files.length;
  photosStatus.textContent = n > 0 ? `${n} photo${n > 1 ? 's' : ''} selected` : 'No photos selected';
  updateProcessButton();
});

inputMusic.addEventListener('change', () => {
  const f = inputMusic.files[0];
  musicStatus.textContent = f ? f.name : 'No file selected';
  updateProcessButton();
});

/* ── Drag-and-drop for upload zones ─────────────────── */
function setupDrop(zone, input) {
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const dt = e.dataTransfer;
    if (dt.files.length) {
      // Assign dropped files to the hidden input via DataTransfer trick
      try {
        input.files = dt.files;
        input.dispatchEvent(new Event('change'));
      } catch (_) {
        // Fallback: Chrome/Safari don't allow assigning .files directly always
        console.warn('Could not assign dropped files to input');
      }
    }
  });
  // Make clicking anywhere in the zone trigger the input
  zone.addEventListener('click', e => {
    if (e.target !== input) input.click();
  });
}

setupDrop(document.getElementById('zone-photos'), inputPhotos);
setupDrop(document.getElementById('zone-music'),  inputMusic);

/* ── Live range value display ───────────────────────── */
rangeSensitivity.addEventListener('input', () => {
  valSensitivity.textContent = parseFloat(rangeSensitivity.value).toFixed(1);
});
rangeZoom.addEventListener('input', () => {
  valZoom.textContent = parseFloat(rangeZoom.value).toFixed(2);
});

/* ── Process ─────────────────────────────────────────── */
btnProcess.addEventListener('click', async () => {
  const fd = new FormData();
  Array.from(inputPhotos.files).forEach(f => fd.append('images', f));
  fd.append('music',       inputMusic.files[0]);
  fd.append('resolution',  document.getElementById('sel-resolution').value);
  fd.append('transition',  document.getElementById('sel-transition').value);
  fd.append('sensitivity', rangeSensitivity.value);
  fd.append('zoom',        rangeZoom.value);
  fd.append('fps',         '24');

  showStage(stageProcessing);
  logBox.innerHTML = '';
  progressBar.style.width = '0%';
  progressLabel.textContent = 'Starting…';

  let res;
  try {
    res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Upload failed');
    currentJobId = data.job_id;
  } catch (err) {
    showError(err.message);
    return;
  }

  startSSE(currentJobId);
});

/* ── SSE ─────────────────────────────────────────────── */
function startSSE(jobId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/progress/${jobId}`);

  eventSource.onmessage = (e) => {
    const data = JSON.parse(e.data);

    if (data.error) {
      eventSource.close();
      showError(data.error);
      return;
    }

    if (data.msg === 'cancelled') {
      eventSource.close();
      resetToUpload();
      return;
    }

    progressBar.style.width = data.pct + '%';
    progressLabel.textContent = data.msg;
    appendLog(data.msg);

    if (data.pct >= 100 && data.msg === 'done') {
      eventSource.close();
      showDone(data);
    }
  };

  eventSource.onerror = () => {
    eventSource.close();
    if (stageProcessing.classList.contains('active')) {
      showError('Lost connection to server. Check the terminal for errors.');
    }
  };
}

function appendLog(msg) {
  const line = document.createElement('span');
  line.className = 'log-line';
  line.textContent = `› ${msg}`;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

/* ── Cancel ─────────────────────────────────────────── */
btnCancel.addEventListener('click', async () => {
  if (!currentJobId) return;
  await fetch(`/cancel/${currentJobId}`, { method: 'POST' });
  if (eventSource) eventSource.close();
  resetToUpload();
});

/* ── Done ────────────────────────────────────────────── */
function showDone(data) {
  const stats = data.stats || {};
  statPhotos.textContent = stats.photos_used ?? '—';
  statBeats.textContent  = stats.beats       ?? '—';
  statBpm.textContent    = stats.bpm ? Math.round(stats.bpm) : '—';
  statDur.textContent    = stats.duration ? `${stats.duration.toFixed(1)}s` : '—';
  btnDownload.href = data.file;
  showStage(stageDone);
}

btnAgain.addEventListener('click', resetToUpload);

/* ── Error ───────────────────────────────────────────── */
function showError(msg) {
  errorMsg.textContent = msg;
  showStage(stageError);
}

btnRetry.addEventListener('click', resetToUpload);

/* ── Reset ───────────────────────────────────────────── */
function resetToUpload() {
  currentJobId = null;
  if (eventSource) { eventSource.close(); eventSource = null; }

  // Clear file inputs (create new DataTransfer)
  try {
    const dt = new DataTransfer();
    inputPhotos.files = dt.files;
    inputMusic.files  = dt.files;
  } catch (_) {}

  photosStatus.textContent = 'No photos selected';
  musicStatus.textContent  = 'No file selected';
  btnProcess.disabled      = true;

  progressBar.style.width     = '0%';
  progressLabel.textContent   = 'Starting…';
  logBox.innerHTML            = '';
  errorMsg.textContent        = '';
  btnDownload.href            = '#';

  showStage(stageUpload);
}
```

**Step 2: Verify no syntax errors**

```bash
node --check static/app.js && echo "JS syntax OK"
```

Expected: `JS syntax OK`

**Step 3: Commit**

```bash
git add static/app.js
git commit -m "feat: add vanilla JS frontend logic"
```

---

### Task 11: End-to-End Smoke Test

**Step 1: Start the server**

```bash
python app.py
```

Expected: `* Running on http://127.0.0.1:5000`

**Step 2: Open browser**

Navigate to `http://localhost:5000` — you should see the BeatCut logo and upload stage.

**Step 3: Quick manual test**

1. Select 3–5 photos (any JPEG/PNG).
2. Select any MP3 or WAV file (≥10 seconds).
3. Leave settings at defaults, click **Process Video**.
4. Watch the progress bar fill and SSE log lines appear.
5. On completion, click **Download MP4** and confirm the file plays.

**Step 4: Test cancel**

Repeat steps 1–3 but click **Cancel** halfway — confirm it returns to Stage 1.

**Step 5: Commit final state**

```bash
git add -A
git commit -m "feat: complete BeatCut scaffold ready for use"
```

---

## Gotchas Reference

| Issue | Fix built into code |
|---|---|
| moviepy API breaks on >1.0.3 | `requirements.txt` pins `moviepy==1.0.3` |
| OpenCV BGR vs RGB | All frames converted with `cv2.cvtColor(BGR→RGB)` before `ImageClip` |
| Flask SSE with threads | `threaded=True` + `stream_with_context` in `/progress` route |
| librosa audioread warnings | `warnings.filterwarnings("ignore")` at top of processor |
| Temp file cleanup | `/download` route spawns 5-second cleanup thread post-send |
| Non-image files slipping through | `ALLOWED_IMAGE_EXTS` set check on `.suffix.lower()` |
| Zero or one beat | Beat times synthesized from duration when `len(beat_times) < 2` |
| Audio longer than video | `AudioFileClip.subclip(0, final.duration)` trims audio to match |
