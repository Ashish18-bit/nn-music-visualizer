"""tests/test_loader.py — AudioFileLoader and AudioBuffer tests."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio.loader import AudioBuffer, AudioFileLoader, load_audio


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

def _make_wav(path: str, duration: float = 3.0, sr: int = 22050, channels: int = 1) -> str:
    """Write a simple sine-wave WAV file and return its path."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    if channels == 2:
        tone = np.stack([tone, tone], axis=-1)
    sf.write(path, tone, sr)
    return path


@pytest.fixture
def mono_wav(tmp_path):
    path = tmp_path / "mono.wav"
    _make_wav(str(path), duration=3.0, sr=22050, channels=1)
    return path


@pytest.fixture
def stereo_wav(tmp_path):
    path = tmp_path / "stereo.wav"
    _make_wav(str(path), duration=3.0, sr=44100, channels=2)
    return path


@pytest.fixture
def short_wav(tmp_path):
    path = tmp_path / "short.wav"
    _make_wav(str(path), duration=0.3, sr=22050, channels=1)
    return path


# ─────────────────────────────────────────────────────────────
#  AudioBuffer
# ─────────────────────────────────────────────────────────────

class TestAudioBuffer:
    def test_basic_construction(self):
        s = np.zeros(22050, dtype=np.float32)
        buf = AudioBuffer(samples=s, sample_rate=22050, duration_sec=1.0, source="test")
        assert buf.samples.shape == (22050,)
        assert buf.sample_rate == 22050

    def test_requires_1d(self):
        with pytest.raises(AssertionError):
            AudioBuffer(
                samples=np.zeros((2, 22050), dtype=np.float32),
                sample_rate=22050, duration_sec=1.0, source="test",
            )

    def test_requires_float32(self):
        with pytest.raises(AssertionError):
            AudioBuffer(
                samples=np.zeros(100, dtype=np.int16),
                sample_rate=22050, duration_sec=0.1, source="test",
            )

    def test_repr(self):
        s = np.zeros(22050, dtype=np.float32)
        buf = AudioBuffer(samples=s, sample_rate=22050, duration_sec=1.0, source="x")
        assert "AudioBuffer" in repr(buf)


# ─────────────────────────────────────────────────────────────
#  AudioFileLoader
# ─────────────────────────────────────────────────────────────

class TestAudioFileLoader:
    def test_load_mono_wav(self, mono_wav):
        loader = AudioFileLoader(target_sr=22050, verbose=False)
        buf = loader.load(mono_wav)

        assert isinstance(buf, AudioBuffer)
        assert buf.samples.ndim == 1
        assert buf.sample_rate == 22050
        assert buf.samples.dtype == np.float32
        assert buf.duration_sec == pytest.approx(3.0, abs=0.05)
        assert "file:" in buf.source

    def test_load_stereo_resamples_to_mono(self, stereo_wav):
        loader = AudioFileLoader(target_sr=22050, verbose=False)
        buf = loader.load(stereo_wav)

        assert buf.samples.ndim == 1
        assert buf.sample_rate == 22050

    def test_samples_in_range(self, mono_wav):
        loader = AudioFileLoader(verbose=False)
        buf = loader.load(mono_wav)
        assert buf.samples.min() >= -1.0
        assert buf.samples.max() <= 1.0

    def test_file_not_found(self):
        loader = AudioFileLoader(verbose=False)
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/song.wav")

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "audio.xyz"
        path.write_text("junk")
        loader = AudioFileLoader(verbose=False)
        with pytest.raises(ValueError, match="Unsupported extension"):
            loader.load(path)

    def test_too_short_raises(self, short_wav):
        loader = AudioFileLoader(min_duration_sec=1.0, verbose=False)
        with pytest.raises(ValueError, match="minimum required"):
            loader.load(short_wav)

    def test_load_segment(self, mono_wav):
        loader = AudioFileLoader(verbose=False)
        buf = loader.load_segment(mono_wav, offset_sec=0.5, duration_sec=1.0)
        assert buf.duration_sec == pytest.approx(1.0, abs=0.05)

    def test_different_target_sr(self, mono_wav):
        loader = AudioFileLoader(target_sr=16000, verbose=False)
        buf = loader.load(mono_wav)
        assert buf.sample_rate == 16000

    def test_metadata_populated(self, mono_wav):
        loader = AudioFileLoader(verbose=False)
        buf = loader.load(mono_wav)
        assert "path" in buf.metadata
        assert "filename" in buf.metadata
        assert "native_sr" in buf.metadata


# ─────────────────────────────────────────────────────────────
#  Convenience function
# ─────────────────────────────────────────────────────────────

class TestLoadAudioHelper:
    def test_basic(self, mono_wav):
        buf = load_audio(mono_wav, verbose=False)
        assert isinstance(buf, AudioBuffer)

    def test_segment_load(self, mono_wav):
        buf = load_audio(mono_wav, offset_sec=1.0, duration_sec=1.5, verbose=False)
        assert buf.duration_sec == pytest.approx(1.5, abs=0.1)
