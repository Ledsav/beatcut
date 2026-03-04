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
