# BeatCut — Todo Features Design
**Date:** 2026-03-10
**Scope:** Quad Stack template, Key Frames, Export Speed, Photo Filters

---

## 1. Quad Stack Template

### Summary
New standalone transition (`transition = "quad_stack"`) with a three-phase repeating cycle.

### Phases (per cycle)
1. **Phase 1 — 4 beats:** Quarters fill one-by-one. Beat 1 → top-left, beat 2 → top-right, beat 3 → bottom-left, beat 4 → bottom-right. Each quarter shows the next photo with a flash on land. Previous quarters remain visible.
2. **Phase 2 — n beats (configurable, default 4):** Polaroid-style photos stack in the center, on top of the four background quarters. Uses same pre-render + shadow approach as the existing polaroid code. `n` is controlled by a "Stack Size" slider (range 2–8).
3. **Phase 3 — 1 beat:** Next photo in sequence expands from center (scale 0.5→1.0 ease-out) to fill the entire screen. Then cycle repeats from Phase 1.

### Backend (`processor.py`)
- New `_build_quad_stack_clips(frames, beat_times, output_w, output_h, fps, stack_size, progress_callback, cancel_flag)` function.
- `assemble_video` receives `stack_size: int = 4` parameter and routes to this function when `transition == "quad_stack"`.

### Frontend
- Add `<option value="quad_stack">Quad Stack</option>` to `#sel-transition`.
- New `#quad-stack-item` settings row (hidden unless `quad_stack` selected), containing a "Stack Size" range input (`min=2, max=8, step=1, default=4`), following same show/hide pattern as `#polaroid-entry-item`.
- Send `stack_size` in FormData on submit.
- `resetToUpload()` resets stack size slider to 4.

---

## 2. Key Frames

### Summary
Per-photo beat pins. Any photo can be marked `First` (appears at beat slot 0) or `Last` (appears at the final beat slot). Independent of transition type.

### UI
- After photos are loaded, `#photos-status` is replaced by a compact per-photo tag list.
- Each photo shows its filename (truncated) + a pill badge that cycles `Auto → First → Last → Auto` on click.
- JS state: `keyframes = {}` — maps file index → `"first" | "last"`.
- Sent as a single `keyframes` form field, e.g. `"0:first,2:last"`. Empty string if no pins.

### Backend (`processor.py`)
- `assemble_video` receives `keyframes: dict = {}` (parsed from the string before being passed in from `app.py`).
- Before the clip-building loop, extract pinned frames:
  - `first_frames`: photos pinned to `"first"`, placed at the start of the sequence.
  - `last_frames`: photos pinned to `"last"`, placed at the end.
  - Remaining photos fill middle slots via existing looping logic (`frames[i % len(remaining)]`).
- If multiple photos are pinned `"first"`, they appear at slots 0, 1, 2… in file-index order. Same for `"last"` in reverse.

---

## 3. Export Speed

### Summary
Profile `assemble_video` to confirm bottleneck, then apply targeted fix.

### Profiling
- Add `PROFILE_EXPORT = False` flag at top of `processor.py`.
- When `True`, wrap the `write_videofile` call (and optionally the whole function) with `cProfile.Profile`, print `pstats` report (sorted by cumulative time) to stdout after export.
- Run once on a representative job, capture output, identify top offenders.

### Expected finding & fix
- `write_videofile` defaults to libx264 with no `preset`, which uses `medium` (slow). Fix: `preset="veryfast"`, `threads=4`.
- Expose `preset` as a dev-only constant at the top of `processor.py` (not a user-facing UI option).
- If profiling reveals clip-building lambdas are the bottleneck instead, address that separately.

---

## 4. Photo Filters (global)

### Summary
One global filter applied to all photos. Options: None, Vignette, Grain, Vintage, Faded. Applied per-frame in `processor.py` after `detect_and_crop()` using PIL/numpy — no new dependencies.

### Filter definitions
| Filter | Implementation |
|--------|---------------|
| **None** | No-op |
| **Vignette** | Radial gaussian mask darkens corners (σ = 0.5 × min dimension) |
| **Grain** | Add gaussian noise (σ ≈ 12) to all channels, clip to [0, 255] |
| **Vintage** | +15R, −10B channel shift + 0.75 saturation blend + light vignette |
| **Faded** | Lift blacks to ~25, contrast scale 0.85, −5R +5B highlight cool |

### Backend (`processor.py`)
- New `apply_filter(rgb: np.ndarray, filter_name: str) -> np.ndarray` function.
- Called on each frame immediately after `detect_and_crop()` in the crop loop inside `assemble_video`.
- `assemble_video` receives `photo_filter: str = "none"`.

### Frontend
- New `<select id="sel-filter">` in settings grid: `None / Vignette / Grain / Vintage / Faded`.
- Sent as `photo_filter` in FormData.
- `resetToUpload()` resets to `none`.
