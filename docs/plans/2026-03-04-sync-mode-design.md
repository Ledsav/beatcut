# Sync Mode Feature — Design

**Date:** 2026-03-04
**Status:** Approved

## Problem

The slideshow currently cuts photos only on rhythmic drum beats (`librosa.beat.beat_track()`). Users cannot make cuts follow the melody, instrumental onsets, or the audio's energy envelope — limiting expressiveness for non-percussive music.

## Goal

Add a **Sync Mode** selector that lets users choose what audio event drives photo cuts:

| Mode | What it follows |
|------|----------------|
| Beat | Drum beat (current behaviour) |
| Melody | Harmonic/melodic note onsets |
| Drums | Percussive component beat (true drum isolation) |
| Energy | Loud/energetic moments in the full audio |

## Approach: HPSS Source Separation

Use `librosa.effects.hpss()` to split audio into harmonic and percussive components. Apply appropriate detection per mode. HPSS is only computed when mode is `drums` or `melody`.

## Design

### 1. UI — new Sync Mode dropdown (`index.html` / `style.css`)

- New `<select id="sel-sync-mode">` in the upload form, next to the existing Transition dropdown.
- Options: `beat` (Beat), `melody` (Melody), `drums` (Drums), `energy` (Energy).
- Styled consistently with existing dropdowns using project CSS vars.
- Use `ui-ux-pro-max` skill for all HTML/CSS work.

### 2. Processor — `detect_sync_points()` replaces `detect_beats()`

Signature:
```python
def detect_sync_points(audio_path: str, sensitivity: float, sync_mode: str) -> tuple[list, float, float]:
    """Returns (sync_times, duration, bpm)."""
```

Code paths:
- **`beat`**: unchanged — `librosa.beat.beat_track(y, sr)`, subsample by step
- **`drums`**: `y_harm, y_perc = librosa.effects.hpss(y)` → `beat_track(y_perc, sr)`, subsample by step
- **`melody`**: HPSS → `librosa.onset.onset_detect(y=y_harm, sr=sr)` → frames to times → subsample by step
- **`energy`**: `librosa.feature.rms(y=y)` → flatten → `scipy.signal.find_peaks(rms, distance=distance)` where `distance = max(1, round(sr / (sensitivity * sr / hop_length)))`

`bpm` is derived from beat_track for `beat`/`drums` modes; estimated from inter-onset intervals for `melody`; estimated from peak spacing for `energy`.

Sensitivity mapping (consistent across all modes):
- `beat`/`drums`/`melody`: `step = max(1, round(1/sensitivity))`, subsample sync_times[::step]
- `energy`: `distance = max(1, round(hop_length / sensitivity))` — higher sensitivity → smaller minimum distance between peaks

### 3. Backend — `app.py`

- `/upload` reads `sync_mode = request.form.get("sync_mode", "beat")`
- Passes `sync_mode` to `assemble_video()`
- `assemble_video()` passes it through to `detect_sync_points()`
- Progress message at 35% becomes: `f"Detected {len(sync_times)} sync points ({sync_mode} mode) at {bpm:.0f} BPM"`

### 4. Frontend — `app.js`

- `FormData` append: `formData.append("sync_mode", document.getElementById("sel-sync-mode").value)`
- `resetToUpload()` resets `sel-sync-mode` to `"beat"`
- No other JS logic needed

## Constraints

- `scipy` is already available (numpy dependency); `find_peaks` requires no new packages
- moviepy 1.0.3 constraint unchanged — no changes to clip assembly
- HPSS adds ~1–2s processing time for melody/drums modes (acceptable)
- `beat` mode is the default — existing behaviour preserved for all current users

## Out of Scope

- Per-frequency-band modes (bass/treble/hi-hat) — not in this iteration
- Animating visual effects based on audio energy — separate feature
- Changing the Polaroid stack logic — no changes
