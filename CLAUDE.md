# BeatCut — Claude Context

## What this project is
Local Flask web app: user uploads photos + a song, gets back a beat-synced, face-centered MP4 slideshow. Runs entirely on localhost:5000.

## Running the app
```bash
source .venv/Scripts/activate   # Windows Git Bash
python app.py                   # opens http://localhost:5000
```
**Never use the system Python** — always `.venv/Scripts/python` / `.venv/Scripts/pip`.

## Project layout
```
beatcut/
├── app.py          # Flask server — routes, job state, SSE
├── processor.py    # All video/audio/face logic — no Flask imports
├── requirements.txt
├── templates/
│   └── index.html  # 4-stage UI (upload → processing → done → error)
└── static/
    ├── style.css   # Dark theme, CSS vars, no frameworks
    └── app.js      # Vanilla JS only — no frameworks
```

## Key architecture decisions

### Job lifecycle (`app.py`)
- `jobs = {}` dict keyed by `uuid4` job_id — **in-memory only**, lost on server restart
- Each job: `{ status, progress, message, output_path, cancel_flag: [False], thread, workdir, events: [], stats }`
- `/upload` saves files to `tempfile.mkdtemp()`, spawns daemon thread, returns `{job_id}` immediately
- `/progress/<job_id>` streams SSE via `stream_with_context(generate())` — polls `job["events"]` every 0.2s
- `/download/<job_id>` serves the MP4 then spawns a 5s cleanup thread deleting the workdir
- `/cancel/<job_id>` sets `cancel_flag[0] = True`; processor checks this between images
- `app.run(debug=False, threaded=True, use_reloader=False)` — reloader MUST stay off or in-memory jobs are wiped mid-job

### SSE gotcha (critical)
`assemble_video` emits `_progress(99, "Export complete...")` at the end — NOT `(100, "done")`.
The `run()` function in `app.py` emits the real `pct=100, msg="done"` event with `file` and `stats`.
The frontend only calls `showDone(data)` when `data.file` is present — prevents triggering on the intermediate 99% event.

### processor.py pipeline
```
assemble_video(image_paths, audio_path, output_path, output_w, output_h,
               fps, sensitivity, transition, zoom, max_duration=0,
               polaroid_entry="drop", progress_callback, cancel_flag)
```
1. `detect_and_crop()` — haarcascade frontal → profile → center fallback, crops to output aspect/zoom, LANCZOS4 resize. Returns `(bgr, face_found)`.
2. `detect_beats()` — librosa beat_track, subsample by `step = max(1, round(1/sensitivity))`. Returns `(beat_times, duration, bpm)`.
3. Clip building — loops frames via `frames[i % len(frames)]` to fill all beats. Two code paths:
   - Standard: flash / fade / zoom_in / cut transitions
   - Polaroid: `_build_polaroid_clips()` — see below
4. `concatenate_videoclips(method="compose")` + `AudioFileClip.subclip` + `write_videofile(libx264/aac)`

**OpenCV → moviepy colour**: all BGR frames must be `cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)` before passing to `ImageClip`.

### Polaroid Stack effect (`_build_polaroid_clips`)
- Black canvas. Each beat: new polaroid lands on a sliding-window pile of max **10** polaroids (oldest dropped when full, pile rerendered from scratch at O(10) cost).
- `_make_polaroid_pil()` — warm off-white RGBA frame, thick bottom border, photo pasted inside.
- Shadow precomputed once per slot at startup: `GaussianBlur(alpha_channel * 0.55)` → stored as `shadow_mask`. Zero blur cost per animation frame.
- Paste order: shadow (offset +SHADOW_DX/DY) → polaroid on top.
- Deterministic random angles/offsets: `np.random.default_rng(seed=7)`, so same inputs → same layout every run.
- Landing styles controlled by `polaroid_entry`: `drop` (ease-out quad, 18% frame height), `flash` (white decay), `fade` (alpha ramp), `cut` (static).

### max_duration
`0` = full song (default). Positive float = cap beat_times at that many seconds. Sent as `0` from frontend when slider is at 600 (sentinel for "full song").

## Frontend (`app.js`)
- Stage transitions: toggle `.active` class — only one `<section>` visible at a time.
- File inputs are `opacity:0; position:absolute` inside a `<label>` — the label is the visible button. **Do not add a second click handler on the zone that would call `input.click()`** if the click originated from within the label (causes double file dialog).
- SSE via `EventSource`, `.close()` called on done/error/cancel.
- `resetToUpload()` resets all inputs including `DataTransfer()` trick for file inputs, resets sliders, resets `selTransition` to `flash`, hides polaroid entry dropdown.
- Polaroid entry dropdown `#polaroid-entry-item` is shown/hidden via `updatePolaroidVisibility()` based on `#sel-transition` value.

## Versions (pinned — do not change)
| Package | Version | Why pinned |
|---|---|---|
| moviepy | **1.0.3 exactly** | `>=2.0` breaks the entire API (`concatenate_videoclips`, `ImageClip`, etc.) |
| Python | 3.12.10 (in .venv) | — |

## Allowed image extensions
`.jpg .jpeg .png .webp .bmp` — checked on `Path(filename).suffix.lower()`.

## Transition values (what the frontend sends → what processor receives)
| UI label | `transition` value |
|---|---|
| Flash | `flash` |
| Hard Cut | `cut` |
| Fade | `fade` |
| Zoom In | `zoom_in` |
| Polaroid Stack | `polaroid` |

## CSS design tokens
```css
--bg: #060608        /* page background */
--surface: #0e0e12   /* cards */
--accent: #c8ff00    /* lime green — buttons, progress, stats */
--danger: #ff3c6e    /* cancel, error */
--text: #e8e8f0
--muted: #6b6b80
```
Fonts: **Bebas Neue** (headings/logo) + **Space Mono** (body) via Google Fonts.

## Common failure modes
| Symptom | Root cause |
|---|---|
| `GET /undefined 404` | Frontend triggered `showDone` before the event with `data.file` arrived — check the `data.file &&` guard in `app.js` |
| Jobs disappear mid-processing | `use_reloader=True` — Werkzeug restarted the process, wiping `jobs` dict |
| Double file picker dialog | Zone click handler calling `input.click()` when click came from within `<label>` |
| No video, no error shown | SSE generator exited before emitting the final "done" event — check the post-status drain loop in `/progress` route |
| `moviepy` import error or wrong API | Wrong version installed — must be `==1.0.3` |
| Audio decode error | Corrupt or unsupported audio — re-export as WAV/MP3 |
