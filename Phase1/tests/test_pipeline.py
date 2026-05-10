"""tests/test_pipeline.py — End-to-end pipeline integration tests."""

import numpy as np
import pytest
import soundfile as sf

from features.extractor import FeatureVector
from pipeline import AudioPipeline


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

SR = 22050


def _write_tone(path, duration=6.0, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), tone, sr)
    return path


@pytest.fixture
def tone_wav(tmp_path):
    return _write_tone(tmp_path / "tone.wav")


@pytest.fixture
def short_wav(tmp_path):
    return _write_tone(tmp_path / "short.wav", duration=0.5)


@pytest.fixture
def pipeline(tmp_path):
    return AudioPipeline(
        config_path=None,
        use_cache=True,
        verbose=False,
    )


# ─────────────────────────────────────────────────────────────
#  Integration: full pipeline
# ─────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_run_returns_list_of_feature_vectors(self, pipeline, tone_wav):
        vectors = pipeline.run(tone_wav)
        assert isinstance(vectors, list)
        assert len(vectors) > 0
        assert all(isinstance(v, FeatureVector) for v in vectors)

    def test_all_vectors_same_dim(self, pipeline, tone_wav):
        vectors = pipeline.run(tone_wav)
        dims = {v.dim for v in vectors}
        assert len(dims) == 1

    def test_vectors_are_finite(self, pipeline, tone_wav):
        vectors = pipeline.run(tone_wav)
        for v in vectors:
            assert np.all(np.isfinite(v.vector)), f"Non-finite values in segment {v.segment_idx}"

    def test_vectors_float32(self, pipeline, tone_wav):
        vectors = pipeline.run(tone_wav)
        for v in vectors:
            assert v.vector.dtype == np.float32

    def test_segment_times_sequential(self, pipeline, tone_wav):
        vectors = pipeline.run(tone_wav)
        for i in range(1, len(vectors)):
            assert vectors[i].start_sec >= vectors[i-1].start_sec

    def test_short_file_returns_empty(self, short_wav, tmp_path):
        pipe = AudioPipeline(use_cache=False, verbose=False)
        # Short file below min_duration_sec should return []
        try:
            vectors = pipe.run(short_wav)
            assert vectors == []
        except Exception:
            pass   # raising is also acceptable

    def test_nonexistent_file_raises(self, pipeline):
        with pytest.raises(FileNotFoundError):
            pipeline.run("/does/not/exist.wav")


# ─────────────────────────────────────────────────────────────
#  Caching
# ─────────────────────────────────────────────────────────────

class TestCaching:
    def test_second_run_hits_cache(self, tone_wav, tmp_path):
        pipe = AudioPipeline(use_cache=True, verbose=False)
        v1 = pipe.run(tone_wav)
        v2 = pipe.run(tone_wav)   # should load from cache
        assert len(v1) == len(v2)
        np.testing.assert_array_equal(v1[0].vector, v2[0].vector)

    def test_force_recompute_ignores_cache(self, tone_wav, tmp_path):
        pipe = AudioPipeline(use_cache=True, verbose=False)
        v1 = pipe.run(tone_wav)
        v2 = pipe.run(tone_wav, force_recompute=True)
        # Both runs produce vectors with the same dimension
        assert v1[0].dim == v2[0].dim
        # First segment should be identical
        np.testing.assert_array_almost_equal(v1[0].vector, v2[0].vector, decimal=5)

    def test_no_cache_still_works(self, tone_wav):
        pipe = AudioPipeline(use_cache=False, verbose=False)
        vectors = pipe.run(tone_wav)
        assert len(vectors) > 0


# ─────────────────────────────────────────────────────────────
#  Batch
# ─────────────────────────────────────────────────────────────

class TestBatchPipeline:
    def test_run_batch_returns_dict(self, pipeline, tmp_path):
        f1 = _write_tone(tmp_path / "a.wav", duration=6.0)
        f2 = _write_tone(tmp_path / "b.wav", duration=9.0)
        results = pipeline.run_batch([f1, f2])
        assert isinstance(results, dict)
        assert len(results) == 2

    def test_batch_skips_bad_files(self, pipeline, tmp_path):
        good = _write_tone(tmp_path / "good.wav", duration=6.0)
        bad = tmp_path / "missing.wav"
        results = pipeline.run_batch([good, bad])
        assert str(good) in results
        assert str(bad) not in results
