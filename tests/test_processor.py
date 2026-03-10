import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_audio(duration_s=3.0, sr=22050):
    """Generate a short sine wave and save to a temp wav file."""
    import tempfile, wave
    samples = (np.sin(2 * np.pi * 440 * np.arange(int(duration_s * sr)) / sr) * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path


def test_detect_sync_points_beat_returns_list():
    from processor import detect_sync_points
    path = _make_audio()
    times, duration, bpm = detect_sync_points(path, sensitivity=1.0, sync_mode="beat")
    assert isinstance(times, list)
    assert duration > 0
    assert bpm >= 0


def test_detect_sync_points_all_modes():
    from processor import detect_sync_points
    path = _make_audio()
    for mode in ("beat", "drums", "melody", "energy"):
        times, duration, bpm = detect_sync_points(path, sensitivity=1.0, sync_mode=mode)
        assert isinstance(times, list), f"mode={mode} did not return list"
        assert duration > 0, f"mode={mode} returned zero duration"


def test_export_constants_exist():
    import processor
    assert hasattr(processor, "EXPORT_PRESET")
    assert hasattr(processor, "EXPORT_THREADS")
    assert hasattr(processor, "PROFILE_EXPORT")
    assert isinstance(processor.EXPORT_PRESET, str)
    assert isinstance(processor.EXPORT_THREADS, int)
    assert processor.EXPORT_THREADS >= 1
    assert processor.PROFILE_EXPORT is False, "Revert PROFILE_EXPORT to False before committing"


def test_detect_sync_points_sensitivity_subsamples():
    from processor import detect_sync_points
    path = _make_audio(duration_s=10.0)
    times_dense, _, _ = detect_sync_points(path, sensitivity=2.0, sync_mode="beat")
    times_sparse, _, _ = detect_sync_points(path, sensitivity=0.5, sync_mode="beat")
    # Higher sensitivity should produce more or equal sync points
    assert len(times_dense) >= len(times_sparse)
