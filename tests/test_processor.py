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
    assert result is not frame, "none filter must return a copy, not the original array"

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

def test_apply_filter_vintage_warms_tone():
    from processor import apply_filter
    # Neutral grey input — vintage should increase R channel relative to B at center
    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    out = apply_filter(frame, "vintage")
    center_r = int(out[100, 100, 0])
    center_b = int(out[100, 100, 2])
    assert center_r > center_b, "vintage should produce warmer tone (higher R than B)"

def test_apply_filter_faded_lifts_blacks():
    from processor import apply_filter
    # Pure black input — faded should lift it above zero
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = apply_filter(frame, "faded")
    assert out[50, 50, 0] > 0, "faded should lift black pixels above zero"

def test_assemble_video_accepts_photo_filter_param():
    import inspect
    from processor import assemble_video
    sig = inspect.signature(assemble_video)
    assert "photo_filter" in sig.parameters
    assert sig.parameters["photo_filter"].default == "none"


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


def test_assemble_video_accepts_keyframes_param():
    import inspect
    from processor import assemble_video
    sig = inspect.signature(assemble_video)
    assert "keyframes" in sig.parameters
    assert sig.parameters["keyframes"].default is None
