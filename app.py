import os
import uuid
import threading
import tempfile
import shutil
import json
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_file, render_template, stream_with_context

from processor import assemble_video

app = Flask(__name__)

# ── Job state ────────────────────────────────────────────────────────────────
# Each job:
# {
#   "status": "running" | "done" | "error" | "cancelled",
#   "progress": 0-100,
#   "message": str,
#   "output_path": str | None,
#   "cancel_flag": [False],
#   "thread": Thread,
#   "workdir": str,
#   "events": list[str],   # SSE event queue (JSON strings)
#   "stats": dict | None,
# }
jobs: dict = {}

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
        "message": "Starting...",
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
