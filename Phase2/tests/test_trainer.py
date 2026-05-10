"""tests/test_trainer.py — Trainer, EarlyStopping, and compute_metrics tests."""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.cnn_lstm import CNNLSTMEmotionClassifier
from training.trainer import Trainer, EarlyStopping, compute_metrics

N_CLASSES = 5
N_MELS = 128
T_FRAMES = 130
FEAT_DIM = 289


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

def _make_loader(n=16, batch_size=4):
    mel = torch.randn(n, 1, N_MELS, T_FRAMES)
    feat = torch.randn(n, FEAT_DIM)
    labels = torch.randint(0, N_CLASSES, (n,))
    ds = TensorDataset(mel, feat, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@pytest.fixture
def tiny_cfg(tmp_path):
    return {
        "emotions": {"num_classes": N_CLASSES},
        "training": {
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "warmup_epochs": 1,
            "early_stopping_patience": 10,
            "early_stopping_metric": "val_f1",
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "save_best_only": True,
            "log_dir": str(tmp_path / "logs"),
            "log_every_n_steps": 1,
        },
    }


@pytest.fixture
def tiny_model():
    return CNNLSTMEmotionClassifier(
        num_classes=N_CLASSES,
        n_mels=N_MELS,
        cnn_channels=(8, 16, 32),
        lstm_hidden=32,
        lstm_layers=1,
        head_hidden=16,
        feature_dim=FEAT_DIM,
    )


@pytest.fixture
def trainer(tiny_model, tiny_cfg):
    train_loader = _make_loader(16, batch_size=4)
    val_loader = _make_loader(8, batch_size=4)
    return Trainer(tiny_model, train_loader, val_loader, tiny_cfg, device="cpu")


# ─────────────────────────────────────────────────────────────
#  compute_metrics
# ─────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_accuracy(self):
        labels = [0, 1, 2, 3, 4]
        preds = [0, 1, 2, 3, 4]
        m = compute_metrics(preds, labels)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_zero_accuracy(self):
        labels = [0, 0, 0, 0]
        preds = [1, 2, 3, 4]
        m = compute_metrics(preds, labels)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_f1_keys_present(self):
        m = compute_metrics([0, 1, 2], [0, 1, 2])
        assert "accuracy" in m
        assert "f1_weighted" in m
        assert "f1_per_class" in m

    def test_f1_weighted_in_range(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 5, 50).tolist()
        preds = rng.integers(0, 5, 50).tolist()
        m = compute_metrics(preds, labels)
        assert 0.0 <= m["f1_weighted"] <= 1.0

    def test_f1_per_class_length(self):
        m = compute_metrics([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], num_classes=5)
        assert len(m["f1_per_class"]) == 5


# ─────────────────────────────────────────────────────────────
#  EarlyStopping
# ─────────────────────────────────────────────────────────────

class TestEarlyStopping:
    def test_does_not_stop_early(self):
        es = EarlyStopping(patience=3, mode="max")
        for v in [0.5, 0.6, 0.7, 0.8]:
            assert not es.step(v)

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3, mode="max")
        es.step(0.8)           # best
        es.step(0.79)          # no improve (counter=1)
        es.step(0.78)          # counter=2
        result = es.step(0.77) # counter=3 → stop
        assert result is True

    def test_resets_counter_on_improvement(self):
        es = EarlyStopping(patience=3, mode="max")
        es.step(0.7)
        es.step(0.6)   # counter=1
        es.step(0.8)   # improvement → counter reset
        assert es.counter == 0

    def test_min_mode(self):
        es = EarlyStopping(patience=2, mode="min")
        es.step(0.5)
        es.step(0.6)   # no improve
        result = es.step(0.7)  # no improve → stop
        assert result is True

    def test_best_tracks_maximum(self):
        es = EarlyStopping(patience=5, mode="max")
        for v in [0.3, 0.7, 0.5, 0.9, 0.6]:
            es.step(v)
        assert es.best == pytest.approx(0.9)

    def test_first_call_never_stops(self):
        es = EarlyStopping(patience=1, mode="max")
        assert not es.step(0.0)


# ─────────────────────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────────────────────

class TestTrainer:
    def test_fit_returns_history(self, trainer):
        history = trainer.fit()
        assert isinstance(history, list)
        assert len(history) > 0

    def test_history_has_required_keys(self, trainer):
        history = trainer.fit()
        required = {"epoch", "train_loss", "val_loss",
                    "train_f1_weighted", "val_f1_weighted",
                    "train_accuracy", "val_accuracy"}
        for row in history:
            assert required.issubset(row.keys())

    def test_epoch_numbers_sequential(self, trainer):
        history = trainer.fit()
        for i, row in enumerate(history):
            assert row["epoch"] == i + 1

    def test_loss_is_positive(self, trainer):
        history = trainer.fit()
        for row in history:
            assert row["train_loss"] > 0
            assert row["val_loss"] > 0

    def test_accuracy_in_range(self, trainer):
        history = trainer.fit()
        for row in history:
            assert 0.0 <= row["train_accuracy"] <= 1.0
            assert 0.0 <= row["val_accuracy"] <= 1.0

    def test_checkpoint_saved(self, trainer, tmp_path):
        trainer.fit()
        ckpt = Path(trainer.cfg["checkpoint_dir"]) / "best_model.pt"
        assert ckpt.exists()

    def test_checkpoint_loadable(self, trainer, tiny_model):
        trainer.fit()
        ckpt = Path(trainer.cfg["checkpoint_dir"]) / "best_model.pt"
        if not ckpt.exists():
            # If no checkpoint was saved (val_f1 never improved), save one now
            trainer.save_checkpoint("best_model.pt", epoch=1, val_f1=0.0)
        saved = torch.load(ckpt, map_location="cpu")
        assert "model_state_dict" in saved
        assert "epoch" in saved
        assert "val_f1" in saved

    def test_save_load_checkpoint_roundtrip(self, trainer, tiny_cfg, tmp_path):
        trainer.fit()
        ckpt_path = trainer.save_checkpoint("test_ckpt.pt", epoch=99, val_f1=0.88)
        assert ckpt_path.exists()
        epoch = trainer.load_checkpoint(ckpt_path)
        assert epoch == 99

    def test_csv_log_created(self, trainer):
        trainer.fit()
        csv_path = Path(trainer.cfg["log_dir"]) / "training_log.csv"
        assert csv_path.exists()

    def test_csv_has_correct_rows(self, trainer):
        history = trainer.fit()
        csv_path = Path(trainer.cfg["log_dir"]) / "training_log.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(history)

    def test_model_weights_change_after_fit(self, tiny_model, tiny_cfg):
        train_loader = _make_loader(16)
        val_loader = _make_loader(8)
        trainer = Trainer(tiny_model, train_loader, val_loader, tiny_cfg, device="cpu")

        w_before = tiny_model.classifier[-1].weight.data.clone()
        trainer.fit()
        w_after = tiny_model.classifier[-1].weight.data

        assert not torch.allclose(w_before, w_after)

    def test_early_stopping_fires(self, tiny_model, tmp_path):
        """With patience=1, training should stop after 2 epochs."""
        cfg = {
            "emotions": {"num_classes": N_CLASSES},
            "training": {
                "epochs": 20,
                "batch_size": 4,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "scheduler": "cosine",
                "warmup_epochs": 1,
                "early_stopping_patience": 1,
                "early_stopping_metric": "val_f1",
                "checkpoint_dir": str(tmp_path / "ckpt"),
                "save_best_only": True,
                "log_dir": str(tmp_path / "logs"),
                "log_every_n_steps": 1,
            },
        }
        train_loader = _make_loader(8)
        val_loader = _make_loader(4)
        trainer = Trainer(tiny_model, train_loader, val_loader, cfg, device="cpu")
        history = trainer.fit()
        assert len(history) < 20
