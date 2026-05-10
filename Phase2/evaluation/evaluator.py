"""
evaluation/evaluator.py
───────────────────────
Post-training evaluation utilities:

  - Evaluator.run()          — full test-set pass, returns metric dict
  - Evaluator.confusion_matrix() — pretty confusion matrix PNG
  - Evaluator.class_report()     — per-class precision/recall/F1
  - Evaluator.plot_training()    — loss/F1 curves from history CSV
  - Evaluator.check_target()     — asserts weighted F1 ≥ target
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

logger = logging.getLogger(__name__)

EMOTIONS = ["happy", "sad", "calm", "angry", "energetic"]


class Evaluator:
    """
    Evaluate a trained CNNLSTMEmotionClassifier on a DataLoader.

    Parameters
    ----------
    model      : trained model
    loader     : test DataLoader
    device     : 'cpu' / 'cuda' / 'mps'
    class_names: list of class label strings
    """

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: Optional[str] = None,
        class_names: List[str] = EMOTIONS,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loader = loader
        self.class_names = class_names
        self._preds: Optional[List[int]] = None
        self._labels: Optional[List[int]] = None
        self._probs: Optional[np.ndarray] = None

    # ── public ──────────────────────────────────────────────

    def run(self) -> Dict:
        """
        Run inference over the full DataLoader.

        Returns
        -------
        metrics dict with keys:
          accuracy, f1_weighted, f1_per_class, loss (if criterion given)
        Also populates internal ._preds, ._labels, ._probs for plots.
        """
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for mel, feats, labels in self.loader:
                mel = mel.to(self.device)
                feats = feats.to(self.device)

                log_probs = self.model(mel, feats)
                probs = log_probs.exp().cpu().numpy()
                preds = log_probs.argmax(dim=1).cpu().tolist()

                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
                all_probs.append(probs)

        self._preds = all_preds
        self._labels = all_labels
        self._probs = np.vstack(all_probs)

        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)

        accuracy = float((preds_arr == labels_arr).mean())
        f1 = float(f1_score(labels_arr, preds_arr, average="weighted", zero_division=0))
        f1_per = f1_score(
            labels_arr, preds_arr,
            labels=list(range(len(self.class_names))),
            average=None, zero_division=0,
        ).tolist()

        metrics = {
            "accuracy": accuracy,
            "f1_weighted": f1,
            "f1_per_class": {
                self.class_names[i]: round(f1_per[i], 4)
                for i in range(len(self.class_names))
            },
            "n_samples": len(all_labels),
        }

        self._print_summary(metrics)
        return metrics

    def class_report(self) -> str:
        """Return sklearn classification_report string."""
        self._require_run()
        return classification_report(
            self._labels, self._preds,
            target_names=self.class_names,
            zero_division=0,
        )

    def plot_confusion_matrix(
        self,
        save_path: str | Path = "evaluation/confusion_matrix.png",
        title: str = "Emotion Confusion Matrix",
    ) -> None:
        """Save a heatmap confusion matrix image."""
        self._require_run()
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not installed — skipping confusion matrix plot")
            return

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(self._labels, self._preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.class_names, yticklabels=self.class_names,
            ax=axes[0],
        )
        axes[0].set_title("Raw counts")
        axes[0].set_ylabel("True label")
        axes[0].set_xlabel("Predicted label")

        # Normalised
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=self.class_names, yticklabels=self.class_names,
            ax=axes[1], vmin=0, vmax=1,
        )
        axes[1].set_title("Normalised (row)")
        axes[1].set_ylabel("True label")
        axes[1].set_xlabel("Predicted label")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Confusion matrix saved → %s", save_path)

    def plot_training_history(
        self,
        csv_path: str | Path = "logs/training_log.csv",
        save_path: str | Path = "evaluation/training_curves.png",
    ) -> None:
        """Plot loss and F1 curves from the training log CSV."""
        try:
            import csv as csv_mod
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available — skipping training curves plot")
            return

        csv_path = Path(csv_path)
        if not csv_path.exists():
            logger.warning("Training log not found: %s", csv_path)
            return

        epochs, train_loss, val_loss = [], [], []
        train_f1, val_f1 = [], []

        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                train_f1.append(float(row["train_f1_weighted"]))
                val_f1.append(float(row["val_f1_weighted"]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Training History", fontsize=13, fontweight="bold")

        # Loss
        axes[0].plot(epochs, train_loss, label="Train", linewidth=2)
        axes[0].plot(epochs, val_loss, label="Val", linewidth=2, linestyle="--")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("NLL Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # F1
        axes[1].plot(epochs, train_f1, label="Train", linewidth=2)
        axes[1].plot(epochs, val_f1, label="Val", linewidth=2, linestyle="--")
        best_ep = epochs[int(np.argmax(val_f1))]
        best_f1 = max(val_f1)
        axes[1].axvline(best_ep, color="red", linestyle=":", alpha=0.6,
                        label=f"Best val_f1={best_f1:.3f}")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Weighted F1")
        axes[1].set_title("F1 Score")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Training curves saved → %s", save_path)

    def check_target(self, target_f1: float = 0.75) -> bool:
        """
        Assert weighted F1 meets the project target.

        Returns True if target met, False otherwise (does not raise).
        """
        self._require_run()
        actual = float(f1_score(
            self._labels, self._preds, average="weighted", zero_division=0
        ))
        passed = actual >= target_f1
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"\n  Target F1 check: {status}")
        print(f"  Required : {target_f1:.4f}")
        print(f"  Achieved : {actual:.4f}\n")
        return passed

    # ── private ─────────────────────────────────────────────

    def _require_run(self) -> None:
        if self._preds is None:
            raise RuntimeError("Call evaluator.run() before plotting or reporting.")

    def _print_summary(self, metrics: Dict) -> None:
        print(f"\n{'─'*50}")
        print(f"  Evaluation Results ({metrics['n_samples']} samples)")
        print(f"{'─'*50}")
        print(f"  Accuracy        : {metrics['accuracy']:.4f}")
        print(f"  Weighted F1     : {metrics['f1_weighted']:.4f}")
        print(f"\n  Per-class F1:")
        for cls, f1 in metrics["f1_per_class"].items():
            bar = "█" * int(f1 * 20)
            print(f"    {cls:<12} {f1:.4f}  {bar}")
        print(f"{'─'*50}\n")
