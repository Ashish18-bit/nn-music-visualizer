"""tests/test_extractor.py — FeatureExtractor tests."""

import numpy as np
import pytest

from audio.preprocessor import AudioSegment
from features.extractor import FeatureExtractor, FeatureVector


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

SR = 22050
WINDOW_SEC = 3.0
WINDOW_LEN = int(SR * WINDOW_SEC)


def _make_segment(
    freq: float = 440.0,
    amplitude: float = 0.5,
    idx: int = 0,
) -> AudioSegment:
    t = np.linspace(0, WINDOW_SEC, WINDOW_LEN, endpoint=False)
    s = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return AudioSegment(
        samples=s,
        sample_rate=SR,
        start_sec=idx * WINDOW_SEC * 0.5,
        end_sec=idx * WINDOW_SEC * 0.5 + WINDOW_SEC,
        segment_idx=idx,
        source="test",
    )


@pytest.fixture
def extractor():
    return FeatureExtractor(verbose=False)


@pytest.fixture
def segment():
    return _make_segment()


# ─────────────────────────────────────────────────────────────
#  FeatureVector shape & type
# ─────────────────────────────────────────────────────────────

class TestFeatureVectorShape:
    def test_vector_is_1d_float32(self, extractor, segment):
        fv = extractor.extract(segment)
        assert fv.vector.ndim == 1
        assert fv.vector.dtype == np.float32

    def test_dim_matches_output_dim(self, extractor, segment):
        fv = extractor.extract(segment)
        assert fv.dim == extractor.output_dim

    def test_output_dim_default(self):
        ext = FeatureExtractor(verbose=False)
        # MFCC 40*3*2=240, chroma 12*2=24, contrast (6+1)*2=14, spectral 3*2=6, te 1+2+2=5
        assert ext.output_dim == 289

    def test_feature_map_covers_full_vector(self, extractor, segment):
        fv = extractor.extract(segment)
        total_from_map = sum(end - start for start, end in fv.feature_map.values())
        assert total_from_map == fv.dim

    def test_feature_map_non_overlapping(self, extractor, segment):
        fv = extractor.extract(segment)
        positions = sorted(fv.feature_map.values())
        for (_, e1), (s2, _) in zip(positions, positions[1:]):
            assert e1 == s2, "Feature map slices overlap or have gaps"


# ─────────────────────────────────────────────────────────────
#  Named sub-vector access
# ─────────────────────────────────────────────────────────────

class TestSubVectorAccess:
    def test_mfcc_group_exists(self, extractor, segment):
        fv = extractor.extract(segment)
        assert "mfcc" in fv.feature_map

    def test_mfcc_dim_with_deltas(self):
        ext = FeatureExtractor(n_mfcc=40, include_delta=True, include_delta2=True, verbose=False)
        fv = ext.extract(_make_segment())
        mfcc_vec = fv.get("mfcc")
        # 40 coeffs * 3 (base+delta+delta2) * 2 (mean+std)
        assert mfcc_vec.shape[0] == 40 * 3 * 2

    def test_mfcc_dim_no_deltas(self):
        ext = FeatureExtractor(n_mfcc=20, include_delta=False, include_delta2=False, verbose=False)
        fv = ext.extract(_make_segment())
        assert fv.get("mfcc").shape[0] == 20 * 2

    def test_chroma_dim(self, extractor, segment):
        fv = extractor.extract(segment)
        assert fv.get("chroma").shape[0] == 12 * 2

    def test_spectral_contrast_dim(self, extractor, segment):
        fv = extractor.extract(segment)
        # (6 bands + 1) * 2
        assert fv.get("spectral_contrast").shape[0] == 14

    def test_get_invalid_key_raises(self, extractor, segment):
        fv = extractor.extract(segment)
        with pytest.raises(KeyError):
            fv.get("nonexistent_feature")


# ─────────────────────────────────────────────────────────────
#  Normalisation
# ─────────────────────────────────────────────────────────────

class TestNormalisation:
    def test_normalised_vector_finite(self, extractor, segment):
        fv = extractor.extract(segment)
        assert np.all(np.isfinite(fv.vector))

    def test_normalised_approx_zero_mean(self, extractor, segment):
        fv = extractor.extract(segment)
        assert abs(fv.vector.mean()) < 1.0   # z-score, so mean near 0

    def test_no_normalisation_option(self, segment):
        ext = FeatureExtractor(normalize=False, verbose=False)
        fv = ext.extract(segment)
        assert np.all(np.isfinite(fv.vector))


# ─────────────────────────────────────────────────────────────
#  Batch extraction
# ─────────────────────────────────────────────────────────────

class TestBatchExtraction:
    def test_batch_returns_same_count(self, extractor):
        segs = [_make_segment(idx=i) for i in range(5)]
        fvs = extractor.extract_batch(segs)
        assert len(fvs) == 5

    def test_batch_all_same_dim(self, extractor):
        segs = [_make_segment(freq=f, idx=i) for i, f in enumerate([220, 440, 880])]
        fvs = extractor.extract_batch(segs)
        dims = {fv.dim for fv in fvs}
        assert len(dims) == 1

    def test_batch_indices_preserved(self, extractor):
        segs = [_make_segment(idx=i) for i in range(4)]
        fvs = extractor.extract_batch(segs)
        for i, fv in enumerate(fvs):
            assert fv.segment_idx == i


# ─────────────────────────────────────────────────────────────
#  Edge cases
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_silence_still_produces_vector(self, extractor):
        silent = AudioSegment(
            samples=np.zeros(WINDOW_LEN, dtype=np.float32),
            sample_rate=SR,
            start_sec=0.0, end_sec=WINDOW_SEC,
            segment_idx=0, source="silent",
        )
        fv = extractor.extract(silent)
        assert fv.dim > 0

    def test_very_loud_signal(self, extractor):
        loud = AudioSegment(
            samples=np.ones(WINDOW_LEN, dtype=np.float32),
            sample_rate=SR,
            start_sec=0.0, end_sec=WINDOW_SEC,
            segment_idx=0, source="loud",
        )
        fv = extractor.extract(loud)
        assert np.all(np.isfinite(fv.vector))

    def test_noise_signal(self, extractor):
        rng = np.random.default_rng(42)
        noise = AudioSegment(
            samples=rng.uniform(-0.5, 0.5, WINDOW_LEN).astype(np.float32),
            sample_rate=SR,
            start_sec=0.0, end_sec=WINDOW_SEC,
            segment_idx=0, source="noise",
        )
        fv = extractor.extract(noise)
        assert fv.dim == extractor.output_dim

    def test_different_frequencies_give_different_vectors(self, extractor):
        fv_low = extractor.extract(_make_segment(freq=110.0))
        fv_high = extractor.extract(_make_segment(freq=2000.0))
        # chroma and spectral features should differ
        assert not np.allclose(fv_low.vector, fv_high.vector, atol=0.1)

    def test_repr_contains_dim(self, extractor, segment):
        fv = extractor.extract(segment)
        assert str(fv.dim) in repr(fv)
