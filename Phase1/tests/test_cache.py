"""tests/test_cache.py — FeatureCache save/load/delete tests."""

import numpy as np
import pytest

from features.cache import FeatureCache
from features.extractor import FeatureVector


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

FMAP = {"mfcc": (0, 80), "chroma": (80, 104), "spectral_contrast": (104, 118)}
DIM = 118


def _make_fv(idx: int = 0) -> FeatureVector:
    rng = np.random.default_rng(idx)
    return FeatureVector(
        vector=rng.random(DIM).astype(np.float32),
        feature_map=FMAP,
        segment_idx=idx,
        start_sec=float(idx * 1.5),
        end_sec=float(idx * 1.5 + 3.0),
        source="test_source",
    )


@pytest.fixture
def cache(tmp_path):
    return FeatureCache(save_dir=tmp_path / "cache")


@pytest.fixture
def batch():
    return [_make_fv(i) for i in range(5)]


# ─────────────────────────────────────────────────────────────
#  Save / load round-trip
# ─────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_returns_path(self, cache, batch):
        path = cache.save(batch, key="test_batch")
        assert path.exists()
        assert path.suffix == ".npz"

    def test_load_returns_same_count(self, cache, batch):
        cache.save(batch, key="test")
        loaded = cache.load("test")
        assert loaded is not None
        assert len(loaded) == len(batch)

    def test_load_vectors_equal(self, cache, batch):
        cache.save(batch, key="test")
        loaded = cache.load("test")
        for orig, restored in zip(batch, loaded):
            np.testing.assert_array_almost_equal(orig.vector, restored.vector, decimal=5)

    def test_load_metadata_preserved(self, cache, batch):
        cache.save(batch, key="test")
        loaded = cache.load("test")
        for orig, restored in zip(batch, loaded):
            assert restored.segment_idx == orig.segment_idx
            assert restored.start_sec == pytest.approx(orig.start_sec)
            assert restored.end_sec == pytest.approx(orig.end_sec)
            assert restored.source == orig.source

    def test_load_feature_map_preserved(self, cache, batch):
        cache.save(batch, key="test")
        loaded = cache.load("test")
        assert loaded[0].feature_map == FMAP

    def test_load_nonexistent_returns_none(self, cache):
        result = cache.load("this_does_not_exist")
        assert result is None

    def test_empty_batch_raises(self, cache):
        with pytest.raises(ValueError):
            cache.save([], key="empty")

    def test_save_single_vector(self, cache):
        fv = _make_fv(0)
        cache.save([fv], key="single")
        loaded = cache.load("single")
        assert len(loaded) == 1
        np.testing.assert_array_almost_equal(fv.vector, loaded[0].vector, decimal=5)


# ─────────────────────────────────────────────────────────────
#  Cache management
# ─────────────────────────────────────────────────────────────

class TestCacheManagement:
    def test_exists_true_after_save(self, cache, batch):
        cache.save(batch, key="mykey")
        assert cache.exists("mykey") is True

    def test_exists_false_before_save(self, cache):
        assert cache.exists("nope") is False

    def test_delete_removes_entry(self, cache, batch):
        cache.save(batch, key="del_me")
        deleted = cache.delete("del_me")
        assert deleted is True
        assert cache.exists("del_me") is False

    def test_delete_nonexistent_returns_false(self, cache):
        assert cache.delete("ghost") is False

    def test_list_keys(self, cache, batch):
        cache.save(batch, key="alpha")
        cache.save(batch, key="beta")
        keys = cache.list_keys()
        assert "alpha" in keys
        assert "beta" in keys

    def test_clear_all(self, cache, batch):
        cache.save(batch, key="a")
        cache.save(batch, key="b")
        n = cache.clear_all()
        assert n == 2
        assert cache.list_keys() == []

    def test_key_sanitisation(self, cache, batch):
        # Keys with special chars should not raise
        cache.save(batch, key="my key/with:special chars!")
        assert len(cache.list_keys()) == 1

    def test_make_key_deterministic(self):
        k1 = FeatureCache.make_key("/path/to/song.mp3", {"n_mfcc": 40})
        k2 = FeatureCache.make_key("/path/to/song.mp3", {"n_mfcc": 40})
        assert k1 == k2

    def test_make_key_changes_with_config(self):
        k1 = FeatureCache.make_key("song.mp3", {"n_mfcc": 40})
        k2 = FeatureCache.make_key("song.mp3", {"n_mfcc": 20})
        assert k1 != k2
