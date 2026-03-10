# BeatCut Todo Features — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement export speed fix, photo filters (global), key frames (per-photo beat pins), and Quad Stack template in priority order.

**Architecture:** All video logic lives in `processor.py` (no Flask imports). `app.py` parses form fields and passes them to `assemble_video`. Frontend is vanilla JS + CSS in `static/`. Tests live in `tests/test_processor.py`.

**Tech Stack:** Python 3.12, Flask, moviepy 1.0.3 (exact — do NOT upgrade), OpenCV, Pillow, librosa, numpy, pytest.

**Run tests with:** `.venv/Scripts/python -m pytest tests/test_processor.py -v`

---

## Task 1: Export Speed — Add Profiling Flag

**Files:**
- Modify: `processor.py` (top of file + `assemble_video`)

**Step 1: Add profiling constants at top of `processor.py`**

After `warnings.filterwarnings("ignore")`, add:

```python
# ── Export tuning ────────────────────────────────────────────────────────────
EXPORT_PRESET  = "veryfast"   # libx264 preset: ultrafast/superfast/veryfast/faster/fast/medium
EXPORT_THREADS = 4            # ffmpeg thread count
PROFILE_EXPORT = False        # set True once to profile write_videofile, then revert
```

**Step 2: Wrap `write_videofile` call in `assemble_video` with optional profiling**

Find the `write_videofile` call (currently at line ~488) and replace it with:

```python
    if PROFILE_EXPORT:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()

    final.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        preset=EXPORT_PRESET,
        threads=EXPORT_THREADS,
        logger=None,
    )

    if PROFILE_EXPORT:
        pr.disable()
        buf = io.StringIO()
        ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
        ps.print_stats(20)
        print("\n=== EXPORT PROFILE ===\n" + buf.getvalue())
```

**Step 3: Write test for the constants**

Add to `tests/test_processor.py`:

```python
def test_export_constants_exist():
    import processor
    assert hasattr(processor, "EXPORT_PRESET")
    assert hasattr(processor, "EXPORT_THREADS")
    assert hasattr(processor, "PROFILE_EXPORT")
    assert isinstance(processor.EXPORT_PRESET, str)
    assert isinstance(processor.EXPORT_THREADS, int)
    assert processor.EXPORT_THREADS >= 1
    assert processor.PROFILE_EXPORT is False  # must be off in committed code
```

**Step 4: Run test**

```
.venv/Scripts/python -m pytest tests/test_processor.py::test_export_constants_exist -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add processor.py tests/test_processor.py
git commit -m "perf: add veryfast preset and thread count to write_videofile"
```

---

## Task 2: Photo Filters — Backend `apply_filter`

**Files:**
- Modify: `processor.py`
- Modify: `tests/test_processor.py`

**Step 1: Write failing tests**

Add to `tests/test_processor.py`:

```python
def test_apply_filter_none_returns_unchanged():
    from processor import apply_filter
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:] = (128, 64, 32)
    result = apply_filter(frame, "none")
    np.testing.assert_array_equal(result, frame)

def test_apply_filter_returns_same_shape():
    from processor import apply_filter
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    for name in ("vignette", "grain", "vintage", "faded"):
        out = apply_filter(frame, name)
        assert out.shape == frame.shape, f"filter={name} changed shape"
        assert out.dtype == np.uint8, f"filter={name} wrong dtype"

def test_apply_filter_vignette_darkens_corners():
    from processor import apply_filter
    frame = np.full((200, 200, 3), 200, dtype=np.uint8)
    out = apply_filter(frame, "vignette")
    center = int(out[100, 100, 0])
    corner = int(out[0, 0, 0])
    assert corner < center, "vignette should darken corners vs center"

def test_apply_filter_grain_changes_pixels():
    from processor import apply_filter
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    out = apply_filter(frame, "grain")
    assert not np.array_equal(out, frame), "grain should change pixel values"

def test_apply_filter_unknown_raises():
    from processor import apply_filter
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown filter"):
        apply_filter(frame, "nonexistent")
```

**Step 2: Run to verify they fail**

```
.venv/Scripts/python -m pytest tests/test_processor.py::test_apply_filter_none_returns_unchanged tests/test_processor.py::test_apply_filter_returns_same_shape -v
```

Expected: FAIL with `ImportError: cannot import name 'apply_filter'`

**Step 3: Implement `apply_filter` in `processor.py`**

Add after the cascade classifiers block (around line 14), before `detect_and_crop`:

```python
def apply_filter(rgb: np.ndarray, filter_name: str) -> np.ndarray:
    """
    Apply a stylised photo filter to an RGB numpy frame (H×W×3, uint8).
    Returns a new uint8 array of the same shape.

    filter_name: "none" | "vignette" | "grain" | "vintage" | "faded"
    """
    if filter_name == "none":
        return rgb

    h, w = rgb.shape[:2]
    out = rgb.astype(np.float32)

    if filter_name == "vignette":
        # Radial gaussian mask darkens corners
        ys = np.linspace(-1, 1, h)
        xs = np.linspace(-1, 1, w)
        xv, yv = np.meshgrid(xs, ys)
        dist = np.sqrt(xv ** 2 + yv ** 2)
        sigma = 0.85
        mask = np.exp(-(dist ** 2) / (2 * sigma ** 2))  # 1.0 at center, falls off
        mask = mask[:, :, np.newaxis]
        out = out * mask

    elif filter_name == "grain":
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0, 12, out.shape)
        out = out + noise

    elif filter_name == "vintage":
        # Warm tone shift
        out[:, :, 0] = out[:, :, 0] + 15   # R +15
        out[:, :, 2] = out[:, :, 2] - 10   # B -10
        # Mild desaturation (blend toward luminance)
        lum = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
        lum = lum[:, :, np.newaxis]
        out = out * 0.75 + lum * 0.25
        # Light vignette
        ys = np.linspace(-1, 1, h)
        xs = np.linspace(-1, 1, w)
        xv, yv = np.meshgrid(xs, ys)
        dist = np.sqrt(xv ** 2 + yv ** 2)
        mask = np.exp(-(dist ** 2) / (2 * 1.2 ** 2))
        out = out * mask[:, :, np.newaxis]

    elif filter_name == "faded":
        # Lift blacks to ~25, reduce contrast to 0.85, slight cool shift
        out = out * 0.85 + 25
        out[:, :, 0] = out[:, :, 0] - 5   # R -5
        out[:, :, 2] = out[:, :, 2] + 5   # B +5

    else:
        raise ValueError(f"Unknown filter: {filter_name!r}. Expected: none, vignette, grain, vintage, faded")

    return np.clip(out, 0, 255).astype(np.uint8)
```

**Step 4: Run all filter tests**

```
.venv/Scripts/python -m pytest tests/test_processor.py -k "filter" -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add processor.py tests/test_processor.py
git commit -m "feat: add apply_filter function (vignette, grain, vintage, faded)"
```

---

## Task 3: Photo Filters — Wire into Pipeline + UI

**Files:**
- Modify: `processor.py` (`assemble_video` signature + crop loop)
- Modify: `app.py` (parse `photo_filter` form field)
- Modify: `templates/index.html` (add filter select)
- Modify: `static/app.js` (send field + reset)

**Step 1: Add `photo_filter` param to `assemble_video`**

Change the signature (around line 336):

```python
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
    photo_filter: str = "none",
    progress_callback=None,
    cancel_flag=None,
) -> dict:
```

In the crop loop inside `assemble_video`, after building each `rgb` frame, add the filter call:

```python
        bgr, face_found = detect_and_crop(path, output_w, output_h, zoom)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = apply_filter(rgb, photo_filter)   # ← add this line
        frames.append(rgb)
```

**Step 2: Parse and pass in `app.py`**

In the `upload()` route, after `polaroid_entry = ...` (around line 80), add:

```python
    photo_filter = request.form.get("photo_filter", "none")
    if photo_filter not in {"none", "vignette", "grain", "vintage", "faded"}:
        photo_filter = "none"
```

In the `assemble_video(...)` call, add:

```python
                photo_filter=photo_filter,
```

**Step 3: Add filter select to `templates/index.html`**

In the `settings-grid`, add a new `setting-item` after the `sel-resolution` item (after line 89):

```html
      <div class="setting-item">
        <label for="sel-filter">Photo Filter</label>
        <select id="sel-filter">
          <option value="none">None</option>
          <option value="vignette">Vignette</option>
          <option value="grain">Grain</option>
          <option value="vintage">Vintage</option>
          <option value="faded">Faded</option>
        </select>
      </div>
```

**Step 4: Wire in `static/app.js`**

In `btnProcess` click handler, after `fd.append('zoom', ...)`, add:

```js
  fd.append('photo_filter', document.getElementById('sel-filter').value);
```

In `resetToUpload()`, after `selSyncMode.value = 'beat';`, add:

```js
  document.getElementById('sel-filter').value = 'none';
```

**Step 5: Write a wiring test**

Add to `tests/test_processor.py`:

```python
def test_assemble_video_accepts_photo_filter_param():
    """assemble_video signature must include photo_filter kwarg."""
    import inspect
    from processor import assemble_video
    sig = inspect.signature(assemble_video)
    assert "photo_filter" in sig.parameters
    assert sig.parameters["photo_filter"].default == "none"
```

**Step 6: Run test**

```
.venv/Scripts/python -m pytest tests/test_processor.py::test_assemble_video_accepts_photo_filter_param -v
```

Expected: PASS

**Step 7: Commit**

```bash
git add processor.py app.py templates/index.html static/app.js tests/test_processor.py
git commit -m "feat: add global photo filter UI and pipeline (vignette, grain, vintage, faded)"
```

---

## Task 4: Key Frames — Backend `_apply_keyframes`

**Files:**
- Modify: `processor.py`
- Modify: `tests/test_processor.py`

**Step 1: Write failing tests**

Add to `tests/test_processor.py`:

```python
def test_apply_keyframes_no_pins_returns_cycling_sequence():
    from processor import _apply_keyframes
    frames = ["a", "b", "c"]
    result = _apply_keyframes(frames, {}, total_clips=7)
    assert result == ["a", "b", "c", "a", "b", "c", "a"]

def test_apply_keyframes_first_pin_goes_to_slot_zero():
    from processor import _apply_keyframes
    frames = ["a", "b", "c"]
    result = _apply_keyframes(frames, {2: "first"}, total_clips=5)
    assert result[0] == "c", "first-pinned frame must be at index 0"

def test_apply_keyframes_last_pin_goes_to_last_slot():
    from processor import _apply_keyframes
    frames = ["a", "b", "c"]
    result = _apply_keyframes(frames, {1: "last"}, total_clips=5)
    assert result[-1] == "b", "last-pinned frame must be at last index"

def test_apply_keyframes_first_and_last_pins():
    from processor import _apply_keyframes
    frames = ["a", "b", "c", "d"]
    result = _apply_keyframes(frames, {0: "first", 3: "last"}, total_clips=6)
    assert result[0] == "a"
    assert result[-1] == "d"
    assert len(result) == 6

def test_apply_keyframes_returns_exact_length():
    from processor import _apply_keyframes
    frames = ["a", "b"]
    for n in (1, 3, 5, 10):
        result = _apply_keyframes(frames, {}, total_clips=n)
        assert len(result) == n, f"expected length {n}, got {len(result)}"
```

**Step 2: Run to verify they fail**

```
.venv/Scripts/python -m pytest tests/test_processor.py -k "keyframe" -v
```

Expected: FAIL with ImportError

**Step 3: Implement `_apply_keyframes` in `processor.py`**

Add after `apply_filter`, before `detect_and_crop`:

```python
def _apply_keyframes(frames: list, keyframes: dict, total_clips: int) -> list:
    """
    Build a sequence of exactly `total_clips` frames honouring beat-pin constraints.

    keyframes: {frame_index: "first" | "last"}
      - "first" pins: placed at the start of the sequence (slot 0, 1, … in ascending index order)
      - "last"  pins: placed at the end   (slots -N … -1 in ascending index order)
      - remaining frames fill the middle via cycling

    Returns a list of length total_clips.
    """
    if not keyframes or not frames:
        return [frames[i % len(frames)] for i in range(total_clips)]

    first_idxs = sorted(i for i, v in keyframes.items() if v == "first" and 0 <= i < len(frames))
    last_idxs  = sorted(i for i, v in keyframes.items() if v == "last"  and 0 <= i < len(frames))
    pinned     = set(first_idxs) | set(last_idxs)
    other_idxs = [i for i in range(len(frames)) if i not in pinned]

    first_frames = [frames[i] for i in first_idxs]
    last_frames  = [frames[i] for i in last_idxs]
    other_frames = [frames[i] for i in other_idxs] if other_idxs else frames

    n_middle = total_clips - len(first_frames) - len(last_frames)
    middle   = [other_frames[j % len(other_frames)] for j in range(max(0, n_middle))]

    sequence = first_frames + middle + last_frames

    # Guarantee exact length (edge case: more pins than clips)
    while len(sequence) < total_clips:
        sequence.append(other_frames[len(sequence) % len(other_frames)])
    return sequence[:total_clips]
```

**Step 4: Run all keyframe tests**

```
.venv/Scripts/python -m pytest tests/test_processor.py -k "keyframe" -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add processor.py tests/test_processor.py
git commit -m "feat: add _apply_keyframes for beat-pinned photo ordering"
```

---

## Task 5: Key Frames — Wire into `assemble_video` + `app.py`

**Files:**
- Modify: `processor.py` (`assemble_video`)
- Modify: `app.py`

**Step 1: Add `keyframes` param to `assemble_video` signature**

```python
def assemble_video(
    ...
    photo_filter: str = "none",
    keyframes: dict = None,          # ← add this
    progress_callback=None,
    cancel_flag=None,
) -> dict:
```

At the start of `assemble_video`, after `def _cancelled()`:

```python
    _keyframes = keyframes or {}
```

**Step 2: Replace frame access in clip-building loop**

After the crop loop (after `_progress(30, "Detecting beats...")`), add:

```python
    # Apply keyframe ordering before building clips
    total_clips_count = len(beat_times) - 1  # will be recalculated after beat detection
```

Then *after* beat detection and beat_times finalisation (after the `if len(beat_times) < 2` block), replace the `total_clips = len(beat_times) - 1` line with:

```python
    total_clips = len(beat_times) - 1
    ordered_frames = _apply_keyframes(frames, _keyframes, total_clips)
```

In the non-polaroid clip loop, replace every `frames[i % len(frames)]` with `ordered_frames[i]`.

In the polaroid branch, pass `ordered_frames` instead of `frames` to `_build_polaroid_clips`:

```python
    if transition == "polaroid":
        clips = _build_polaroid_clips(
            ordered_frames, beat_times, output_w, output_h, fps,
            polaroid_entry, progress_callback, cancel_flag,
        )
```

**Step 3: Add `keyframes` parsing in `app.py`**

In `upload()`, after `photo_filter = ...`, add:

```python
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
```

Pass `keyframes=keyframes` to `assemble_video`.

**Step 4: Write wiring test**

Add to `tests/test_processor.py`:

```python
def test_assemble_video_accepts_keyframes_param():
    import inspect
    from processor import assemble_video
    sig = inspect.signature(assemble_video)
    assert "keyframes" in sig.parameters
    assert sig.parameters["keyframes"].default is None
```

**Step 5: Run test**

```
.venv/Scripts/python -m pytest tests/test_processor.py::test_assemble_video_accepts_keyframes_param -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add processor.py app.py tests/test_processor.py
git commit -m "feat: wire keyframes param into assemble_video pipeline"
```

---

## Task 6: Key Frames — Frontend Badge UI

**Files:**
- Modify: `templates/index.html`
- Modify: `static/app.js`
- Modify: `static/style.css`

**Step 1: Add keyframe badge container to `index.html`**

Replace the `<p class="upload-status" id="photos-status">` line in `#zone-photos`:

```html
        <p class="upload-status" id="photos-status">No photos selected</p>
        <div class="kf-list" id="kf-list" style="display:none;"></div>
```

**Step 2: Add CSS for badges in `style.css`**

Append to `style.css`:

```css
/* ── Key frame badges ──────────────────────────────── */
.kf-list {
  width: 100%;
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 120px;
  overflow-y: auto;
}

.kf-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.kf-name {
  font-size: 11px;
  color: var(--muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 140px;
}

.kf-badge {
  font-family: 'Space Mono', monospace;
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 20px;
  border: 1px solid #2a2a38;
  background: var(--surface2);
  color: var(--muted);
  cursor: pointer;
  white-space: nowrap;
  flex-shrink: 0;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
}
.kf-badge[data-pin="first"] {
  background: rgba(200,255,0,0.12);
  color: var(--accent);
  border-color: var(--accent);
}
.kf-badge[data-pin="last"] {
  background: rgba(255,60,110,0.12);
  color: var(--danger);
  border-color: var(--danger);
}
```

**Step 3: Add keyframe JS logic to `app.js`**

Add after the `selSyncMode` DOM ref block:

```js
/* ── Key frames ──────────────────────────────────────── */
let keyframeState = {};   // { fileIndex: "first" | "last" }

const kfList = document.getElementById('kf-list');

function renderKfList(files) {
  kfList.innerHTML = '';
  if (!files || files.length === 0) { kfList.style.display = 'none'; return; }
  kfList.style.display = '';
  Array.from(files).forEach((file, idx) => {
    const row = document.createElement('div');
    row.className = 'kf-row';

    const name = document.createElement('span');
    name.className = 'kf-name';
    name.textContent = file.name;
    name.title = file.name;

    const badge = document.createElement('button');
    badge.className = 'kf-badge';
    const pin = keyframeState[idx] || 'auto';
    badge.dataset.pin = pin;
    badge.textContent = pin === 'auto' ? 'Auto' : pin === 'first' ? 'First' : 'Last';

    badge.addEventListener('click', () => {
      const cur = keyframeState[idx] || 'auto';
      const next = cur === 'auto' ? 'first' : cur === 'first' ? 'last' : 'auto';
      if (next === 'auto') delete keyframeState[idx];
      else keyframeState[idx] = next;
      badge.dataset.pin = next;
      badge.textContent = next === 'auto' ? 'Auto' : next === 'first' ? 'First' : 'Last';
    });

    row.appendChild(name);
    row.appendChild(badge);
    kfList.appendChild(row);
  });
}

function serializeKeyframes() {
  return Object.entries(keyframeState)
    .map(([idx, pin]) => `${idx}:${pin}`)
    .join(',');
}
```

Update the `inputPhotos` change handler to call `renderKfList`:

```js
inputPhotos.addEventListener('change', () => {
  const n = inputPhotos.files.length;
  photosStatus.textContent = n > 0 ? `${n} photo${n > 1 ? 's' : ''} selected` : 'No photos selected';
  keyframeState = {};
  renderKfList(inputPhotos.files);
  updateProcessButton();
});
```

In `btnProcess` click handler, after `fd.append('photo_filter', ...)`:

```js
  fd.append('keyframes', serializeKeyframes());
```

In `resetToUpload()`, after `selSyncMode.value = 'beat';`:

```js
  keyframeState = {};
  renderKfList(null);
```

**Step 4: Manual test**

Start the app (`python app.py`), upload some photos, verify badges appear, click them to cycle Auto → First → Last → Auto, confirm correct colors.

**Step 5: Commit**

```bash
git add templates/index.html static/app.js static/style.css
git commit -m "feat: add per-photo key frame badge UI (First/Last beat pins)"
```

---

## Task 7: Quad Stack — Backend `_build_quad_stack_clips`

**Files:**
- Modify: `processor.py`
- Modify: `tests/test_processor.py`

**Step 1: Write a smoke test**

Add to `tests/test_processor.py`:

```python
def test_build_quad_stack_clips_returns_clips():
    from processor import _build_quad_stack_clips
    frames = [np.zeros((540, 540, 3), dtype=np.uint8) for _ in range(12)]
    # Provide enough beat times for 2 full cycles (4 + 4 + 1 = 9 per cycle → 18+ beats)
    beat_times = [i * 0.5 for i in range(20)]
    clips = _build_quad_stack_clips(
        frames, beat_times,
        output_w=540, output_h=540,
        fps=24, stack_size=4,
        progress_callback=None, cancel_flag=None,
    )
    assert len(clips) == len(beat_times) - 1
    assert all(hasattr(c, "duration") for c in clips), "each item must be a moviepy clip"
```

**Step 2: Run to verify it fails**

```
.venv/Scripts/python -m pytest tests/test_processor.py::test_build_quad_stack_clips_returns_clips -v
```

Expected: FAIL with ImportError

**Step 3: Implement `_build_quad_stack_clips` in `processor.py`**

Add after `_build_polaroid_clips`, before `assemble_video`:

```python
def _build_quad_stack_clips(frames, beat_times, output_w, output_h, fps,
                             stack_size, progress_callback, cancel_flag):
    """
    Quad Stack template: repeating 3-phase cycle.

    Phase 1 (4 beats): quarters fill top-left → top-right → bottom-left → bottom-right,
                        each with a flash on land.
    Phase 2 (stack_size beats): polaroids stack in the center on top of the background.
    Phase 3 (1 beat): hero photo zooms from 0.5× to full screen (ease-out).

    After Phase 3 the background and pile reset for the next cycle.
    """
    total = len(beat_times) - 1
    if total == 0:
        return []

    cycle_len = 4 + stack_size + 1

    # Quarter rects: (x1, y1, x2, y2)
    hw, hh = output_w // 2, output_h // 2
    quarter_rects = [
        (0,  0,  hw,       hh),        # top-left
        (hw, 0,  output_w, hh),        # top-right
        (0,  hh, hw,       output_h),  # bottom-left
        (hw, hh, output_w, output_h),  # bottom-right
    ]

    # ── Pre-render centre polaroids (smaller than full-screen version) ──
    pol_size  = int(min(output_w, output_h) * 0.35)
    b_side    = max(int(pol_size * 0.07), 6)
    b_top     = max(int(pol_size * 0.07), 6)
    b_bottom  = max(int(pol_size * 0.22), 18)
    rng       = np.random.default_rng(seed=13)
    angles    = rng.uniform(-12, 12,           total + 1).tolist()
    offsets_x = rng.integers(-output_w // 18, output_w // 18, total + 1).tolist()
    offsets_y = rng.integers(-output_h // 18, output_h // 18, total + 1).tolist()
    SHADOW_DX   = max(int(pol_size * 0.012), 4)
    SHADOW_DY   = max(int(pol_size * 0.015), 5)
    SHADOW_BLUR = max(int(pol_size * 0.018), 6)

    pol_data = []
    for i in range(total + 1):
        rgb   = frames[i % len(frames)]
        pil   = _make_polaroid_pil(rgb, pol_size, b_side, b_top, b_bottom)
        rot   = pil.rotate(-angles[i], expand=True, resample=PILImage.BICUBIC)
        alpha = rot.getchannel("A")
        shad  = alpha.point(lambda p: int(p * 0.55))
        shad  = shad.filter(PILImageFilter.GaussianBlur(radius=SHADOW_BLUR))
        pol_data.append((rot, shad))

    def _paste_pol_center(canvas, idx, px, py):
        pol, shad = pol_data[idx]
        black = PILImage.new("RGB", pol.size, (0, 0, 0))
        canvas.paste(black, (px + SHADOW_DX, py + SHADOW_DY), shad)
        canvas.paste(pol.convert("RGB"), (px, py), pol.getchannel("A"))

    # ── Mutable cycle state ──
    quad_canvas      = PILImage.new("RGB", (output_w, output_h), (0, 0, 0))
    pol_pile_canvas  = PILImage.new("RGB", (output_w, output_h), (0, 0, 0))
    pol_pile_slots   = []   # (idx, px, py)

    clips = []
    frame_counter = 0

    for i in range(total):
        if cancel_flag is not None and cancel_flag[0]:
            raise InterruptedError("Job cancelled")

        clip_dur  = max(beat_times[i + 1] - beat_times[i], 1.0 / fps)
        phase     = i % cycle_len
        rgb       = frames[frame_counter % len(frames)]
        frame_counter += 1

        # ── Phase 1: fill a quarter ──────────────────────────────────
        if phase < 4:
            q_x1, q_y1, q_x2, q_y2 = quarter_rects[phase]
            q_w, q_h = q_x2 - q_x1, q_y2 - q_y1
            # Resize photo to quarter size
            pil_q = PILImage.fromarray(rgb).resize((q_w, q_h), PILImage.LANCZOS)
            quad_canvas = quad_canvas.copy()
            quad_canvas.paste(pil_q, (q_x1, q_y1))
            pol_pile_canvas = PILImage.new("RGB", (output_w, output_h), (0, 0, 0))
            pol_pile_slots  = []
            base_rgb = np.array(quad_canvas)

            _fps2 = fps
            def _fl_qflash(gf, t, _base=base_rgb.copy(), _fps=_fps2):
                bgr = cv2.cvtColor(_base, cv2.COLOR_RGB2BGR)
                return cv2.cvtColor(_apply_flash(bgr, t, _fps), cv2.COLOR_BGR2RGB)

            clip = ImageClip(base_rgb, duration=clip_dur).fl(_fl_qflash)

        # ── Phase 2: stack a polaroid in the centre ──────────────────
        elif phase < 4 + stack_size:
            pol, _ = pol_data[i]
            rx, ry = pol.size
            px = output_w // 2 - rx // 2 + int(offsets_x[i])
            py = output_h // 2 - ry // 2 + int(offsets_y[i])

            # Composite canvas = quad background + pile so far
            composite = quad_canvas.copy()
            for (si, sx, sy) in pol_pile_slots:
                _paste_pol_center(composite, si, sx, sy)
            _paste_pol_center(composite, i, px, py)
            full_rgb = np.array(composite)

            _b_base   = quad_canvas.copy()
            _b_slots  = list(pol_pile_slots)
            _i2, _px2, _py2 = i, px, py

            def _fl_drop(gf, t, _b=_b_base, _slots=_b_slots,
                         _ii=_i2, _pxi=_px2, _pyi=_py2, _d=clip_dur,
                         _dp=int(min(output_w, output_h) * 0.14)):
                p    = min(t / max(_d, 1e-6), 1.0)
                ease = 1.0 - (1.0 - p) ** 2
                dy   = int(_dp * (1.0 - ease))
                c    = _b.copy()
                for (si, sx, sy) in _slots:
                    _paste_pol_center(c, si, sx, sy)
                pol_r, shad_r = pol_data[_ii]
                black = PILImage.new("RGB", pol_r.size, (0, 0, 0))
                c.paste(black, (_pxi + SHADOW_DX, _pyi - dy + SHADOW_DY), shad_r)
                c.paste(pol_r.convert("RGB"), (_pxi, _pyi - dy), pol_r.getchannel("A"))
                return np.array(c)

            clip = ImageClip(full_rgb, duration=clip_dur).fl(_fl_drop)

            pol_pile_slots.append((i, px, py))
            pol_pile_canvas = composite.copy()

        # ── Phase 3: hero zoom fills the screen ─────────────────────
        else:
            hero_rgb = rgb.copy()
            _h, _w   = hero_rgb.shape[:2]
            _d3      = clip_dur

            def _fl_hero(gf, t, _hr=hero_rgb, _d=_d3):
                p     = min(t / max(_d, 1e-6), 1.0)
                ease  = 1.0 - (1.0 - p) ** 2       # ease-out quad 0→1
                scale = 0.5 + 0.5 * ease            # 0.5 → 1.0
                hh2, ww2 = _hr.shape[:2]
                crop_w = max(1, int(ww2 / scale))
                crop_h = max(1, int(hh2 / scale))
                x1     = max(0, (ww2 - crop_w) // 2)
                y1     = max(0, (hh2 - crop_h) // 2)
                x2     = min(ww2, x1 + crop_w)
                y2     = min(hh2, y1 + crop_h)
                crop   = _hr[y1:y2, x1:x2]
                bgr    = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2BGR)
                out    = cv2.resize(bgr, (ww2, hh2), interpolation=cv2.INTER_LINEAR)
                return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

            clip = ImageClip(hero_rgb, duration=clip_dur).fl(_fl_hero)

            # Reset background and pile for next cycle
            quad_canvas     = PILImage.new("RGB", (output_w, output_h), (0, 0, 0))
            pol_pile_canvas = PILImage.new("RGB", (output_w, output_h), (0, 0, 0))
            pol_pile_slots  = []

        clips.append(clip)

        if i % max(1, total // 10) == 0:
            pct = 35 + int(55 * (i + 1) / total)
            if progress_callback:
                progress_callback(pct, f"Building quad stack clip {i + 1}/{total}")

    return clips
```

**Step 4: Run the smoke test**

```
.venv/Scripts/python -m pytest tests/test_processor.py::test_build_quad_stack_clips_returns_clips -v
```

Expected: PASS

**Step 5: Run full test suite**

```
.venv/Scripts/python -m pytest tests/test_processor.py -v
```

Expected: all PASS

**Step 6: Commit**

```bash
git add processor.py tests/test_processor.py
git commit -m "feat: add _build_quad_stack_clips (quad stack template backend)"
```

---

## Task 8: Quad Stack — Wire into `assemble_video` + `app.py`

**Files:**
- Modify: `processor.py`
- Modify: `app.py`

**Step 1: Add `stack_size` param to `assemble_video`**

```python
def assemble_video(
    ...
    polaroid_entry: str = "drop",
    photo_filter: str = "none",
    keyframes: dict = None,
    stack_size: int = 4,             # ← add this
    progress_callback=None,
    cancel_flag=None,
) -> dict:
```

**Step 2: Add routing for `quad_stack` in clip-building section**

After the `if transition == "polaroid":` branch, add an `elif`:

```python
    elif transition == "quad_stack":
        clips = _build_quad_stack_clips(
            ordered_frames, beat_times, output_w, output_h, fps,
            stack_size, progress_callback, cancel_flag,
        )
```

**Step 3: Parse `stack_size` in `app.py`**

After `polaroid_entry = ...`:

```python
    stack_size = max(2, min(8, int(request.form.get("stack_size", 4))))
```

Pass `stack_size=stack_size` to `assemble_video`.

**Step 4: Write wiring test**

Add to `tests/test_processor.py`:

```python
def test_assemble_video_accepts_stack_size_param():
    import inspect
    from processor import assemble_video
    sig = inspect.signature(assemble_video)
    assert "stack_size" in sig.parameters
    assert sig.parameters["stack_size"].default == 4
```

**Step 5: Run test**

```
.venv/Scripts/python -m pytest tests/test_processor.py::test_assemble_video_accepts_stack_size_param -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add processor.py app.py tests/test_processor.py
git commit -m "feat: wire quad_stack into assemble_video and app.py upload route"
```

---

## Task 9: Quad Stack — Frontend UI

**Files:**
- Modify: `templates/index.html`
- Modify: `static/app.js`

**Step 1: Add `quad_stack` option to transition select in `index.html`**

After `<option value="polaroid">Polaroid Stack</option>`:

```html
          <option value="quad_stack">Quad Stack</option>
```

**Step 2: Add Stack Size setting row in `index.html`**

After the `#polaroid-entry-item` div:

```html
      <div class="setting-item" id="quad-stack-item" style="display:none;">
        <label for="range-stack-size">Stack Size <span id="val-stack-size">4</span></label>
        <input type="range" id="range-stack-size" min="2" max="8" step="1" value="4" />
      </div>
```

**Step 3: Wire Stack Size show/hide + live display in `app.js`**

Update `updatePolaroidVisibility` to also handle `quad_stack`:

```js
function updatePolaroidVisibility() {
  polaroidEntryItem.style.display = selTransition.value === 'polaroid'    ? '' : 'none';
  document.getElementById('quad-stack-item').style.display =
    selTransition.value === 'quad_stack' ? '' : 'none';
}
```

Add range listener after `rangeZoom` listener:

```js
document.getElementById('range-stack-size').addEventListener('input', () => {
  document.getElementById('val-stack-size').textContent =
    document.getElementById('range-stack-size').value;
});
```

In `btnProcess` click handler, after `fd.append('keyframes', ...)`:

```js
  fd.append('stack_size', document.getElementById('range-stack-size').value);
```

In `resetToUpload()`, after keyframeState reset:

```js
  document.getElementById('range-stack-size').value = '4';
  document.getElementById('val-stack-size').textContent = '4';
  document.getElementById('quad-stack-item').style.display = 'none';
```

**Step 4: Manual test**

Start app, select "Quad Stack" transition, verify Stack Size slider appears. Upload photos + music and run a short export. Confirm video shows quarter-fill → polaroid pile → hero zoom cycle.

**Step 5: Run full test suite one last time**

```
.venv/Scripts/python -m pytest tests/test_processor.py -v
```

Expected: all PASS

**Step 6: Final commit**

```bash
git add templates/index.html static/app.js
git commit -m "feat: add Quad Stack UI (transition option + stack size slider)"
```
