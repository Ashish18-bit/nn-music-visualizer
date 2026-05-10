"""tests/test_evaluator.py — Evaluator metrics and output tests."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.cnn_lstm import CNNLSTMEmotionClassifier
from evaluation.evaluator import Evaluator, EMOTIONS

N_CLASSES = 5
N_MELS = 128
T_FRAMES = 130
FEAT_DIM = 289


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

def _make_loader(n=20, batch_size=4):
    mel = torch.randn(n, 1, N_MELS, T_FRAMES)
    feat = torch.randn(n, FEAT_DIM)
    labels = torch.randint(0, N_CLASSES, (n,))
    ds = TensorDataset(mel, feat, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@pytest.fixture
def tiny_model():
    m = CNNLSTMEmotionClassifier(
        num_classes=N_CLASSES,
        n_mels=N_MELS,
        cnn_channels=(8, 16, 32),
        lstm_hidden=32,
        lstm_layers=1,
        head_hidden=16,
        feature_dim=FEAT_DIM,
    )
    m.eval()
    return m


@pytest.fixture
def evaluator(tiny_model):
    loader = _make_loader()
    return Evaluator(tiny_model, loader, device="cpu", class_names=EMOTIONS)


# ─────────────────────────────────────────────────────────────
#  run() — metric correctness
# ─────────────────────────────────────────────────────────────

class TestEvaluatorRun:
    def test_run_returns_dict(self, evaluator):
        metrics = evaluator.run()
        assert isinstance(metrics, dict)

    def test_required_keys_present(self, evaluator):
        metrics = evaluator.run()
        assert "accuracy" in metrics
        assert "f1_weighted" in metrics
        assert "f1_per_class" in metrics
        assert "n_samples" in metrics

    def test_accuracy_in_range(self, evaluator):
        metrics = evaluator.run()
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_f1_in_range(self, evaluator):
        metrics = evaluator.run()
        assert 0.0 <= metrics["f1_weighted"] <= 1.0

    def test_n_samples_correct(self, evaluator):
        metrics = evaluator.run()
        assert metrics["n_samples"] == 20

    def test_per_class_f1_all_emotions(self, evaluator):
        metrics = evaluator.run()
        for emotion in EMOTIONS:
            assert emotion in metrics["f1_per_class"]

    def test_per_class_f1_in_range(self, evaluator):
        metrics = evaluator.run()
        for v in metrics["f1_per_class"].values():
            assert 0.0 <= v <= 1.0

    def test_perfect_classifier_accuracy(self):
        """A model that always predicts class 0 should have 100% acc on all-0 labels."""
        import torch.nn.functional as F_func

        local_model = CNNLSTMEmotionClassifier(
            num_classes=N_CLASSES, n_mels=N_MELS,
            cnn_channels=(8, 16, 32), lstm_hidden=32,
            lstm_layers=1, head_hidden=16, feature_dim=FEAT_DIM,
        )
        local_model.eval()

        mel = torch.randn(8, 1, N_MELS, T_FRAMES)
        feat = torch.randn(8, FEAT_DIM)
        labels = torch.zeros(8, dtype=torch.long)
        ds = TensorDataset(mel, feat, labels)
        loader = DataLoader(ds, batch_size=4)

        # monkey-patch forward to always return class 0
        def always_zero(m, f):
            logits = torch.zeros(m.shape[0], N_CLASSES)
            logits[:, 0] = 10.0
            return F_func.log_softmax(logits, dim=1)

        local_model.forward = always_zero
        ev = Evaluator(local_model, loader, device="cpu")
        metrics = ev.run()
        assert metrics["accuracy"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────
#  Internal state after run()
# ─────────────────────────────────────────────────────────────

class TestInternalState:
    def test_preds_populated(self, evaluator):
        evaluator.run()
        assert evaluator._preds is not None
        assert len(evaluator._preds) == 20

    def test_labels_populated(self, evaluator):
        evaluator.run()
        assert evaluator._labels is not None
        assert len(evaluator._labels) == 20

    def test_probs_shape(self, evaluator):
        evaluator.run()
        assert evaluator._probs is not None
        assert evaluator._probs.shape == (20, N_CLASSES)

    def test_probs_sum_to_one(self, evaluator):
        evaluator.run()
        row_sums = evaluator._probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(20), atol=1e-5)

    def test_preds_valid_class_range(self, evaluator):
        evaluator.run()
        for p in evaluator._preds:
            assert 0 <= p < N_CLASSES


# ─────────────────────────────────────────────────────────────
#  class_report()
# ─────────────────────────────────────────────────────────────

class TestClassReport:
    def test_returns_string(self, evaluator):
        evaluator.run()
        report = evaluator.class_report()
        assert isinstance(report, str)

    def test_contains_emotion_names(self, evaluator):
        evaluator.run()
        report = evaluator.class_report()
        for emotion in EMOTIONS:
            assert emotion in report

    def test_requires_run_first(self, tiny_model):
        loader = _make_loader(8)
        ev = Evaluator(tiny_model, loader, device="cpu")
        with pytest.raises(RuntimeError, match="run\\(\\)"):
            ev.class_report()


# ─────────────────────────────────────────────────────────────
#  check_target()
# ─────────────────────────────────────────────────────────────

class TestCheckTarget:
    def test_returns_bool(self, evaluator):
        evaluator.run()
        result = evaluator.check_target(target_f1=0.0)
        assert isinstance(result, bool)

    def test_passes_with_zero_target(self, evaluator):
        evaluator.run()
        assert evaluator.check_target(target_f1=0.0) is True

    def test_fails_with_impossible_target(self, evaluator):
        evaluator.run()
        assert evaluator.check_target(target_f1=1.01) is False

    def test_requires_run_first(self, tiny_model):
        loader = _make_loader(8)
        ev = Evaluator(tiny_model, loader, device="cpu")
        with pytest.raises(RuntimeError):
            ev.check_target()


# ─────────────────────────────────────────────────────────────
#  plot_confusion_matrix()
# ─────────────────────────────────────────────────────────────

class TestPlotConfusionMatrix:
    def test_saves_file(self, evaluator, tmp_path):
        evaluator.run()
        out = tmp_path / "cm.png"
        evaluator.plot_confusion_matrix(save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_requires_run_first(self, tiny_model, tmp_path):
        loader = _make_loader(8)
        ev = Evaluator(tiny_model, loader, device="cpu")
        with pytest.raises(RuntimeError):
            ev.plot_confusion_matrix(save_path=tmp_path / "cm.png")


# ─────────────────────────────────────────────────────────────
#  plot_training_history()
# ─────────────────────────────────────────────────────────────

class TestPlotTrainingHistory:
    def _write_fake_csv(self, path: Path, n_epochs: int = 5) -> None:
        rows = []
        for e in range(1, n_epochs + 1):
            rows.append({
                "epoch": e,
                "lr": 1e-3,
                "train_loss": 1.5 - e * 0.1,
                "val_loss": 1.6 - e * 0.09,
                "train_f1_weighted": 0.2 + e * 0.05,
                "val_f1_weighted": 0.18 + e * 0.05,
                "train_accuracy": 0.3 + e * 0.04,
                "val_accuracy": 0.28 + e * 0.04,
                "elapsed_sec": 2.0,
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_saves_png(self, evaluator, tmp_path):
        csv_path = tmp_path / "logs" / "training_log.csv"
        self._write_fake_csv(csv_path)
        out = tmp_path / "curves.png"
        evaluator.plot_training_history(csv_path=csv_path, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_missing_csv_does_not_crash(self, evaluator, tmp_path):
        # Should log a warning, not raise
        evaluator.plot_training_history(
            csv_path=tmp_path / "nonexistent.csv",
            save_path=tmp_path / "out.png",
        )
