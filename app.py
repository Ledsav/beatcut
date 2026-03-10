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
    zoom       = float(request.form.get("zoom", 1.5))
    start_time = float(request.form.get("start_time", 0))
    end_time   = float(request.form.get("end_time", 0))
    sync_mode  = request.form.get("sync_mode", "beat")
    if sync_mode not in {"beat", "drums", "melody", "energy"}:
        shutil.rmtree(workdir, ignore_errors=True)
        return jsonify({"error": f"Invalid sync_mode: {sync_mode!r}"}), 400
    polaroid_entry  = request.form.get("polaroid_entry", "drop")
    photo_filter = request.form.get("photo_filter", "none")
    if photo_filter not in {"none", "vignette", "grain", "vintage", "faded"}:
        photo_filter = "none"

    # keyframes: "0:first,2:last" → {0: "first", 2: "last"}
    keyframes_raw = request.form.get("keyframes", "")
    keyframes = {}
    for part in keyframes_raw.split(","):
        part = part.strip()
        if ":" in part:
            try:
                idx_str, pin = part.split(":", 1)
                idx = int(idx_str)
                if pin in ("first", "last"):
                    keyframes[idx] = pin
            except ValueError:
                pass

    stack_size = max(2, min(8, int(request.form.get("stack_size", 4))))

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
                start_time=start_time,
                end_time=end_time,
                sync_mode=sync_mode,
                polaroid_entry=polaroid_entry,
                photo_filter=photo_filter,
                keyframes=keyframes,
                stack_size=stack_size,
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


@app.route("/progress/<job_id>")
def progress(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        import time
        last = 0
        while True:
            events = job["events"]
            while last < len(events):
                yield f"data: {events[last]}\n\n"
                last += 1
            if job["status"] in ("done", "error", "cancelled"):
                # Final drain: pick up any events added between last check and status change
                events = job["events"]
                while last < len(events):
                    yield f"data: {events[last]}\n\n"
                    last += 1
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
    # use_reloader=False: Werkzeug's reloader restarts the process on file changes,
    # which wipes the in-memory jobs dict mid-processing and causes 404s on SSE.
    app.run(debug=False, threaded=True, port=5000, use_reloader=False)
