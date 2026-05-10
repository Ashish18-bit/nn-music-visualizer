"""tests/test_preprocessor.py — Preprocessor and segmentation tests."""

import numpy as np
import pytest
import soundfile as sf

from audio.loader import AudioBuffer
from audio.preprocessor import (
    AudioSegment,
    PreprocessedAudio,
    Preprocessor,
    apply_noise_gate,
    compute_rms_db,
    is_silent,
    normalise_rms,
    remove_dc_offset,
    segment_audio,
)


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

SR = 22050

def _sine_buf(duration=3.0, freq=440.0, amplitude=0.5, sr=SR) -> AudioBuffer:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    s = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return AudioBuffer(samples=s, sample_rate=sr, duration_sec=duration, source="test")


def _silent_buf(duration=3.0, sr=SR) -> AudioBuffer:
    s = np.zeros(int(sr * duration), dtype=np.float32)
    return AudioBuffer(samples=s, sample_rate=sr, duration_sec=duration, source="silent")


def _dc_offset_buf(duration=3.0, dc=0.3, sr=SR) -> AudioBuffer:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    s = (0.3 * np.sin(2 * np.pi * 440 * t) + dc).astype(np.float32)
    s = np.clip(s, -1.0, 1.0)
    return AudioBuffer(samples=s, sample_rate=sr, duration_sec=duration, source="dc")


# ─────────────────────────────────────────────────────────────
#  Pure function tests
# ─────────────────────────────────────────────────────────────

class TestPureFunctions:
    def test_remove_dc_offset_zeroes_mean(self):
        s = np.ones(1000, dtype=np.float32) * 0.5
        result = remove_dc_offset(s)
        assert abs(result.mean()) < 1e-6

    def test_compute_rms_db_sine(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        s = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        # RMS of full-scale sine = 1/sqrt(2) → ~-3 dBFS
        rms = compute_rms_db(s)
        assert rms == pytest.approx(-3.01, abs=0.5)

    def test_compute_rms_db_silence(self):
        s = np.zeros(1000, dtype=np.float32)
        rms = compute_rms_db(s)
        assert rms <= -100     # essentially -inf

    def test_is_silent_returns_true_for_silence(self):
        s = np.zeros(1000, dtype=np.float32)
        assert is_silent(s, threshold_db=-60.0) is True

    def test_is_silent_returns_false_for_tone(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        s = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        assert is_silent(s, threshold_db=-60.0) is False

    def test_normalise_rms_target(self):
        buf = _sine_buf()
        normalised = normalise_rms(buf.samples, target_db=-20.0)
        rms_db = compute_rms_db(normalised)
        assert rms_db == pytest.approx(-20.0, abs=1.0)

    def test_normalise_rms_clips_to_one(self):
        very_loud = np.ones(1000, dtype=np.float32) * 2.0
        result = normalise_rms(very_loud, target_db=-1.0)
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_apply_noise_gate_kills_silence(self):
        silent = np.zeros(SR * 3, dtype=np.float32)
        result = apply_noise_gate(silent, threshold_db=-60.0)
        assert np.all(result == 0.0)

    def test_apply_noise_gate_preserves_tone(self):
        t = np.linspace(0, 3, SR * 3, endpoint=False)
        s = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = apply_noise_gate(s, threshold_db=-60.0)
        # Most samples should remain non-zero
        assert np.count_nonzero(result) > len(result) * 0.8


# ─────────────────────────────────────────────────────────────
#  Segmentation tests
# ─────────────────────────────────────────────────────────────

class TestSegmentation:
    def test_number_of_segments(self):
        samples = np.zeros(SR * 9, dtype=np.float32)  # 9 seconds
        # window=3s, hop=1.5s: starts at 0,1.5,3,4.5,6,7.5 → 6 full windows
        segs = segment_audio(samples, SR, window_sec=3.0, overlap=0.5)
        assert len(segs) >= 5

    def test_segment_shape(self):
        samples = np.random.rand(SR * 6).astype(np.float32)
        segs = segment_audio(samples, SR, window_sec=3.0, overlap=0.5)
        window_len = int(SR * 3.0)
        for seg in segs:
            assert seg.samples.shape == (window_len,)

    def test_segment_dtype(self):
        samples = np.random.rand(SR * 6).astype(np.float32)
        segs = segment_audio(samples, SR)
        for seg in segs:
            assert seg.samples.dtype == np.float32

    def test_segment_indices_sequential(self):
        samples = np.zeros(SR * 6, dtype=np.float32)
        segs = segment_audio(samples, SR)
        for i, seg in enumerate(segs):
            assert seg.segment_idx == i

    def test_start_end_times(self):
        samples = np.zeros(SR * 6, dtype=np.float32)
        segs = segment_audio(samples, SR, window_sec=3.0, overlap=0.5)
        assert segs[0].start_sec == pytest.approx(0.0)
        assert segs[0].end_sec == pytest.approx(3.0, abs=0.01)

    def test_overlap_zero(self):
        samples = np.zeros(SR * 6, dtype=np.float32)
        segs = segment_audio(samples, SR, window_sec=3.0, overlap=0.0)
        assert len(segs) == 2

    def test_audio_too_short_raises(self):
        short = np.zeros(int(SR * 1.0), dtype=np.float32)
        with pytest.raises(ValueError, match="exceeds audio length"):
            segment_audio(short, SR, window_sec=3.0)

    def test_invalid_overlap_raises(self):
        samples = np.zeros(SR * 6, dtype=np.float32)
        with pytest.raises(ValueError, match="hop would be"):
            segment_audio(samples, SR, window_sec=3.0, overlap=1.0)

    def test_source_label_propagated(self):
        samples = np.zeros(SR * 6, dtype=np.float32)
        segs = segment_audio(samples, SR, source="my_source")
        assert all(seg.source == "my_source" for seg in segs)

    def test_pad_last_segment(self):
        # 5.5 seconds → last window needs padding
        samples = np.zeros(int(SR * 5.5), dtype=np.float32)
        segs = segment_audio(samples, SR, window_sec=3.0, overlap=0.5, pad_last=True)
        window_len = int(SR * 3.0)
        assert segs[-1].samples.shape == (window_len,)

    def test_no_pad_skips_tail(self):
        samples = np.zeros(int(SR * 5.5), dtype=np.float32)
        segs_pad = segment_audio(samples, SR, window_sec=3.0, overlap=0.5, pad_last=True)
        segs_nopad = segment_audio(samples, SR, window_sec=3.0, overlap=0.5, pad_last=False)
        assert len(segs_pad) > len(segs_nopad)


# ─────────────────────────────────────────────────────────────
#  Preprocessor class tests
# ─────────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_process_returns_preprocessed_audio(self):
        prep = Preprocessor(verbose=False)
        buf = _sine_buf()
        result = prep.process(buf)
        assert isinstance(result, PreprocessedAudio)

    def test_process_normalises_rms(self):
        prep = Preprocessor(norm_target_db=-20.0, verbose=False)
        buf = _sine_buf(amplitude=0.1)   # quiet signal
        result = prep.process(buf)
        rms = compute_rms_db(result.samples)
        assert rms == pytest.approx(-20.0, abs=2.0)

    def test_process_silent_buffer(self):
        prep = Preprocessor(verbose=False)
        buf = _silent_buf()
        result = prep.process(buf)
        assert result.is_silent is True

    def test_process_removes_dc(self):
        prep = Preprocessor(verbose=False)
        buf = _dc_offset_buf(dc=0.3)
        result = prep.process(buf)
        assert abs(result.samples.mean()) < 0.01

    def test_samples_in_range_after_process(self):
        prep = Preprocessor(verbose=False)
        buf = _sine_buf(amplitude=0.9)
        result = prep.process(buf)
        assert result.samples.min() >= -1.0
        assert result.samples.max() <= 1.0

    def test_segment_produces_list(self):
        prep = Preprocessor(window_sec=3.0, overlap=0.5, verbose=False)
        buf = _sine_buf(duration=9.0)
        processed = prep.process(buf)
        segs = prep.segment(processed)
        assert isinstance(segs, list)
        assert len(segs) > 0

    def test_segment_silent_returns_empty(self):
        prep = Preprocessor(verbose=False)
        buf = _silent_buf()
        processed = prep.process(buf)
        segs = prep.segment(processed)
        assert segs == []

    def test_process_and_segment_shortcut(self):
        prep = Preprocessor(window_sec=3.0, verbose=False)
        buf = _sine_buf(duration=9.0)
        segs = prep.process_and_segment(buf)
        assert len(segs) > 0
        assert all(isinstance(s, AudioSegment) for s in segs)
