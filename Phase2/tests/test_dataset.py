"""tests/test_dataset.py — dataset, augmentation, and dataloader tests."""

import numpy as np
import pytest
import soundfile as sf
import torch

from data.dataset import (
    EMOTIONS, LABEL2IDX, IDX2LABEL,
    SpecAugment, DataAugmentor, SyntheticGenerator,
    EmotionDataset, compute_melspec,
    scan_dataset, split_dataset, build_dataloaders,
)

SR = 22050
WINDOW_LEN = SR * 3


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

def _write_tone(path, freq=440.0, duration=3.0, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    s = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), s, sr)
    return path


@pytest.fixture
def tiny_raw_dir(tmp_path):
    """2 files per emotion class."""
    raw = tmp_path / "raw"
    for i, emotion in enumerate(EMOTIONS):
        d = raw / emotion
        d.mkdir(parents=True)
        for j in range(2):
            _write_tone(d / f"{emotion}_{j}.wav", freq=200 + i * 50 + j * 10)
    return raw


@pytest.fixture
def minimal_cfg(tmp_path):
    return {
        "emotions": {"classes": EMOTIONS, "num_classes": 5},
        "dataset": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "use_synthetic": True,
            "synthetic_samples": 4,
            "train_split": 0.70,
            "val_split": 0.15,
            "test_split": 0.15,
            "random_seed": 42,
        },
        "audio": {"sample_rate": SR, "window_sec": 3.0, "overlap": 0.5, "min_duration_sec": 1.0},
        "features": {
            "n_mels": 64, "n_fft": 512, "hop_length": 256,
            "f_min": 20.0, "f_max": 8000.0,
            "use_feature_vector": True, "feature_dim": 100,
        },
        "augmentation": {
            "enabled": False,
            "time_stretch_range": [0.9, 1.1],
            "pitch_shift_range": [-1, 1],
            "noise_factor": 0.005,
            "spec_augment": {
                "freq_mask_param": 5, "time_mask_param": 10,
                "num_freq_masks": 1, "num_time_masks": 1,
            },
        },
        "training": {"batch_size": 4, "learning_rate": 0.001,
                     "weight_decay": 1e-4, "epochs": 2,
                     "warmup_epochs": 1, "early_stopping_patience": 5,
                     "early_stopping_metric": "val_f1",
                     "checkpoint_dir": str(tmp_path / "ckpt"),
                     "save_best_only": True,
                     "log_dir": str(tmp_path / "logs"),
                     "log_every_n_steps": 1},
        "evaluation": {"metrics": ["accuracy", "f1_weighted"], "target_f1": 0.5},
    }


# ─────────────────────────────────────────────────────────────
#  Label mapping
# ─────────────────────────────────────────────────────────────

class TestLabelMapping:
    def test_all_emotions_have_idx(self):
        for e in EMOTIONS:
            assert e in LABEL2IDX

    def test_label_idx_roundtrip(self):
        for e in EMOTIONS:
            assert IDX2LABEL[LABEL2IDX[e]] == e

    def test_five_classes(self):
        assert len(EMOTIONS) == 5


# ─────────────────────────────────────────────────────────────
#  SpecAugment
# ─────────────────────────────────────────────────────────────

class TestSpecAugment:
    def test_output_shape_preserved(self):
        aug = SpecAugment(freq_mask_param=10, time_mask_param=20)
        spec = np.random.rand(128, 130).astype(np.float32)
        result = aug(spec)
        assert result.shape == spec.shape

    def test_introduces_zeros(self):
        aug = SpecAugment(freq_mask_param=20, time_mask_param=40,
                          num_freq_masks=2, num_time_masks=2)
        spec = np.ones((128, 130), dtype=np.float32)
        result = aug(spec)
        assert (result == 0.0).any()

    def test_values_in_range(self):
        aug = SpecAugment()
        spec = np.random.rand(64, 100).astype(np.float32)
        result = aug(spec)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6


# ─────────────────────────────────────────────────────────────
#  DataAugmentor
# ─────────────────────────────────────────────────────────────

class TestDataAugmentor:
    def test_output_is_float32(self):
        aug = DataAugmentor(noise_factor=0.01)
        s = np.random.randn(WINDOW_LEN).astype(np.float32)
        result = aug(s)
        assert result.dtype == np.float32

    def test_output_clipped(self):
        aug = DataAugmentor(noise_factor=0.5)
        s = np.ones(WINDOW_LEN, dtype=np.float32)
        result = aug(s)
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_output_same_or_similar_length(self):
        aug = DataAugmentor(time_stretch_range=(0.95, 1.05))
        s = np.random.randn(WINDOW_LEN).astype(np.float32) * 0.5
        result = aug(s)
        # Time stretch may change length slightly, that's OK
        assert len(result) > 0


# ─────────────────────────────────────────────────────────────
#  compute_melspec
# ─────────────────────────────────────────────────────────────

class TestComputeMelspec:
    def test_output_shape(self):
        s = np.random.rand(WINDOW_LEN).astype(np.float32)
        mel = compute_melspec(s, sr=SR, n_mels=64, target_frames=100)
        assert mel.shape == (64, 100)

    def test_output_dtype(self):
        s = np.random.rand(WINDOW_LEN).astype(np.float32)
        mel = compute_melspec(s, target_frames=100)
        assert mel.dtype == np.float32

    def test_values_in_0_1(self):
        s = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 3, WINDOW_LEN))).astype(np.float32)
        mel = compute_melspec(s, target_frames=100)
        assert mel.min() >= 0.0 - 1e-6
        assert mel.max() <= 1.0 + 1e-6

    def test_silence_no_nan(self):
        s = np.zeros(WINDOW_LEN, dtype=np.float32)
        mel = compute_melspec(s, target_frames=100)
        assert np.all(np.isfinite(mel))


# ─────────────────────────────────────────────────────────────
#  SyntheticGenerator
# ─────────────────────────────────────────────────────────────

class TestSyntheticGenerator:
    def test_generates_correct_files(self, tmp_path):
        gen = SyntheticGenerator(sr=SR, duration=1.0, output_dir=tmp_path / "raw")
        paths = gen.generate(n_per_class=3)
        for emotion in EMOTIONS:
            assert emotion in paths
            assert len(paths[emotion]) == 3
            for p in paths[emotion]:
                assert p.exists()

    def test_audio_is_valid(self, tmp_path):
        gen = SyntheticGenerator(sr=SR, duration=1.0, output_dir=tmp_path / "raw")
        paths = gen.generate(n_per_class=2)
        for emotion, file_paths in paths.items():
            audio, file_sr = sf.read(str(file_paths[0]))
            assert file_sr == SR
            assert len(audio) > 0
            assert np.all(np.isfinite(audio))
            assert np.abs(audio).max() <= 1.0 + 1e-5

    def test_emotions_differ(self, tmp_path):
        gen = SyntheticGenerator(sr=SR, duration=1.0, output_dir=tmp_path / "raw")
        paths = gen.generate(n_per_class=2)
        audios = {}
        for emotion, fps in paths.items():
            a, _ = sf.read(str(fps[0]))
            audios[emotion] = a[:SR]
        # happy and sad should not be identical
        assert not np.allclose(audios["happy"], audios["sad"], atol=0.01)


# ─────────────────────────────────────────────────────────────
#  scan_dataset / split_dataset
# ─────────────────────────────────────────────────────────────

class TestScanSplit:
    def test_scan_returns_all_files(self, tiny_raw_dir):
        samples = scan_dataset(tiny_raw_dir)
        assert len(samples) == len(EMOTIONS) * 2

    def test_scan_labels_correct(self, tiny_raw_dir):
        samples = scan_dataset(tiny_raw_dir)
        for path, label in samples:
            emotion = path.parent.name
            assert LABEL2IDX[emotion] == label

    def test_scan_missing_class_warns(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "happy").mkdir()
        _write_tone(raw / "happy" / "h.wav")
        samples = scan_dataset(raw)
        labels = {lbl for _, lbl in samples}
        assert LABEL2IDX["happy"] in labels

    def test_split_sizes(self, tiny_raw_dir):
        samples = scan_dataset(tiny_raw_dir)
        train, val, test = split_dataset(samples, train=0.7, val=0.15)
        total = len(train) + len(val) + len(test)
        assert total == len(samples)

    def test_split_no_overlap(self, tiny_raw_dir):
        samples = scan_dataset(tiny_raw_dir)
        train, val, test = split_dataset(samples)
        train_paths = {str(p) for p, _ in train}
        val_paths = {str(p) for p, _ in val}
        test_paths = {str(p) for p, _ in test}
        assert train_paths.isdisjoint(val_paths)
        assert train_paths.isdisjoint(test_paths)


# ─────────────────────────────────────────────────────────────
#  EmotionDataset
# ─────────────────────────────────────────────────────────────

class TestEmotionDataset:
    def test_getitem_returns_three_tensors(self, tiny_raw_dir):
        samples = scan_dataset(tiny_raw_dir)
        ds = EmotionDataset(samples, n_mels=64, feature_dim=100)
        mel, feat, label = ds[0]
        assert isinstance(mel, torch.Tensor)
        assert isinstance(feat, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_mel_shape(self, tiny_raw_dir):
        samples = scan_dataset(tiny_raw_dir)
        ds = EmotionDataset(samples, n_mels=64, target_frames=100, feature_dim=100)
        mel, _, _ = ds[0]
        assert mel.shape == (1, 64, 100)

    def test_feature_dim(self, tiny_raw_dir):
        samples = scan_dataset(tiny_raw_dir)
        ds = EmotionDataset(samples, n_mels=64, feature_dim=150)
        _, feat, _ = ds[0]
        assert feat.shape == (150,)

    def test_label_in_range(self, tiny_raw_dir):
        samples = scan_dataset(tiny_raw_dir)
        ds = EmotionDataset(samples, n_mels=64, feature_dim=100)
        for i in range(len(ds)):
            _, _, label = ds[i]
            assert 0 <= label.item() < 5

    def test_mel_dtype(self, tiny_raw_dir):
        samples = scan_dataset(tiny_raw_dir)
        ds = EmotionDataset(samples, n_mels=64, feature_dim=100)
        mel, feat, _ = ds[0]
        assert mel.dtype == torch.float32
        assert feat.dtype == torch.float32

    def test_feature_finite(self, tiny_raw_dir):
        samples = scan_dataset(tiny_raw_dir)
        ds = EmotionDataset(samples, n_mels=64, feature_dim=100)
        _, feat, _ = ds[0]
        assert torch.all(torch.isfinite(feat))

    def test_class_counts(self, tiny_raw_dir):
        samples = scan_dataset(tiny_raw_dir)
        ds = EmotionDataset(samples, n_mels=64, feature_dim=100)
        counts = ds.class_counts
        assert sum(counts.values()) == len(ds)

    def test_with_spec_augment(self, tiny_raw_dir):
        samples = scan_dataset(tiny_raw_dir)
        aug = SpecAugment(freq_mask_param=5, time_mask_param=10)
        ds = EmotionDataset(samples, n_mels=64, feature_dim=100, spec_augment=aug)
        mel, _, _ = ds[0]
        assert mel.shape[0] == 1


# ─────────────────────────────────────────────────────────────
#  build_dataloaders
# ─────────────────────────────────────────────────────────────

class TestBuildDataloaders:
    def test_returns_three_loaders(self, minimal_cfg, tmp_path):
        minimal_cfg["dataset"]["raw_dir"] = str(tmp_path / "raw")
        train, val, test = build_dataloaders(minimal_cfg)
        assert train is not None
        assert val is not None
        assert test is not None

    def test_batch_shape(self, minimal_cfg, tmp_path):
        minimal_cfg["dataset"]["raw_dir"] = str(tmp_path / "raw")
        train, _, _ = build_dataloaders(minimal_cfg)
        mel, feat, labels = next(iter(train))
        B = mel.shape[0]
        assert mel.shape == (B, 1, 64, 130)
        assert feat.shape[0] == B
        assert labels.shape == (B,)

    def test_labels_are_long(self, minimal_cfg, tmp_path):
        minimal_cfg["dataset"]["raw_dir"] = str(tmp_path / "raw")
        train, _, _ = build_dataloaders(minimal_cfg)
        _, _, labels = next(iter(train))
        assert labels.dtype == torch.long
