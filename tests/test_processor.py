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
