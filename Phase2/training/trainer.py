"""
training/trainer.py
───────────────────
Full training loop for the CNN-LSTM emotion classifier.

Features
────────
- Train / val / test split handling
- Cosine LR schedule with linear warmup
- Early stopping on val_f1 (configurable)
- Best-model checkpointing (.pt files)
- Per-epoch logging to CSV + console
- Class-weighted NLLLoss to handle imbalanced datasets
- Gradient clipping for LSTM stability
- Device-agnostic (CPU / CUDA / MPS)
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Metric helpers
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    all_preds: List[int],
    all_labels: List[int],
    num_classes: int = 5,
) -> Dict[str, float]:
    """Compute accuracy and weighted F1 from prediction lists."""
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    accuracy = float((preds == labels).mean())
    f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    f1_per_class = f1_score(
        labels, preds,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    ).tolist()

    return {
        "accuracy": accuracy,
        "f1_weighted": f1,
        "f1_per_class": f1_per_class,
    }


# ─────────────────────────────────────────────────────────────
#  EarlyStopping
# ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience  : epochs to wait before stopping
    min_delta : minimum change to count as improvement
    mode      : 'max' (higher=better) or 'min' (lower=better)
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """
        Call once per epoch. Returns True if training should stop.
        """
        if self.best is None:
            self.best = value
            return False

        if self.mode == "max":
            improved = value >= self.best + self.min_delta
        else:
            improved = value <= self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ─────────────────────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the full training lifecycle of CNNLSTMEmotionClassifier.

    Parameters
    ----------
    model           : the model to train
    train_loader    : training DataLoader
    val_loader      : validation DataLoader
    cfg             : full config dict (training section used)
    device          : 'cpu', 'cuda', or 'mps'
    class_weights   : optional per-class loss weights tensor

    Usage
    -----
    trainer = Trainer(model, train_loader, val_loader, cfg)
    history = trainer.fit()
    trainer.save_checkpoint("checkpoints/best.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: Optional[str] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.cfg = cfg["training"]
        self.num_classes = cfg["emotions"]["num_classes"]

        # ── device ──────────────────────────────────────────
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        logger.info("Training on device: %s", self.device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ── loss function ────────────────────────────────────
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.NLLLoss(weight=class_weights)

        # ── optimiser ────────────────────────────────────────
        self.optimiser = Adam(
            self.model.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=self.cfg["weight_decay"],
        )

        # ── LR scheduler: linear warmup → cosine decay ───────
        warmup = self.cfg["warmup_epochs"]
        total = self.cfg["epochs"]
        warmup_sched = LinearLR(
            self.optimiser,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup,
        )
        cosine_sched = CosineAnnealingLR(
            self.optimiser,
            T_max=total - warmup,
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimiser,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup],
        )

        # ── early stopping ────────────────────────────────────
        self.early_stopping = EarlyStopping(
            patience=self.cfg["early_stopping_patience"],
            mode="max",
        )

        # ── checkpointing ─────────────────────────────────────
        self.checkpoint_dir = Path(self.cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_f1 = 0.0
        self.best_epoch = 0

        # ── logging ───────────────────────────────────────────
        self.log_dir = Path(self.cfg["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict] = []
        self._csv_path = self.log_dir / "training_log.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = None   # initialised on first write

    # ── public ──────────────────────────────────────────────

    def fit(self) -> List[Dict]:
        """
        Run the full training loop.

        Returns
        -------
        history : list of per-epoch metric dicts
        """
        epochs = self.cfg["epochs"]
        logger.info(
            "Starting training: %d epochs, batch_size=%d",
            epochs, self.train_loader.batch_size,
        )

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._val_epoch()

            elapsed = time.perf_counter() - t0
            lr = self.optimiser.param_groups[0]["lr"]
            self.scheduler.step()

            row = {
                "epoch": epoch,
                "lr": lr,
                **{f"train_{k}": v for k, v in train_metrics.items()
                   if not isinstance(v, list)},
                **{f"val_{k}": v for k, v in val_metrics.items()
                   if not isinstance(v, list)},
                "elapsed_sec": elapsed,
            }
            self.history.append(row)
            self._log_epoch(row)
            self._write_csv(row)

            # ── checkpoint best model ────────────────────────
            val_f1 = val_metrics["f1_weighted"]
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                self.save_checkpoint("best_model.pt", epoch, val_f1)

            # ── early stopping ───────────────────────────────
            if self.early_stopping.step(val_f1):
                logger.info(
                    "Early stopping at epoch %d (best val_f1=%.4f @ epoch %d)",
                    epoch, self.best_val_f1, self.best_epoch,
                )
                break

        self._csv_file.close()
        logger.info(
            "Training complete. Best val_f1=%.4f at epoch %d",
            self.best_val_f1, self.best_epoch,
        )
        return self.history

    def save_checkpoint(
        self,
        filename: str,
        epoch: int = 0,
        val_f1: float = 0.0,
    ) -> Path:
        """Save model weights + training state to a .pt file."""
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "val_f1": val_f1,
                "model_state_dict": self.model.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved → %s  (val_f1=%.4f)", path, val_f1)
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        """Load model weights from checkpoint. Returns saved epoch number."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimiser.load_state_dict(ckpt["optimiser_state_dict"])
        epoch = ckpt.get("epoch", 0)
        val_f1 = ckpt.get("val_f1", 0.0)
        logger.info(
            "Loaded checkpoint from %s  (epoch=%d, val_f1=%.4f)",
            path, epoch, val_f1,
        )
        return epoch

    # ── private: epoch loops ─────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        for batch_idx, (mel, feats, labels) in enumerate(self.train_loader):
            mel = mel.to(self.device)
            feats = feats.to(self.device)
            labels = labels.to(self.device)

            self.optimiser.zero_grad()
            log_probs = self.model(mel, feats)
            loss = self.criterion(log_probs, labels)
            loss.backward()

            # Gradient clipping — important for LSTM stability
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()

            total_loss += loss.item()
            preds = log_probs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

            if (batch_idx + 1) % self.cfg["log_every_n_steps"] == 0:
                logger.debug(
                    "Epoch %d  step %d/%d  loss=%.4f",
                    epoch, batch_idx + 1, len(self.train_loader), loss.item(),
                )

        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_metrics(all_preds, all_labels, self.num_classes)
        metrics["loss"] = avg_loss
        return metrics

    @torch.no_grad()
    def _val_epoch(self) -> Dict:
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        for mel, feats, labels in self.val_loader:
            mel = mel.to(self.device)
            feats = feats.to(self.device)
            labels = labels.to(self.device)

            log_probs = self.model(mel, feats)
            loss = self.criterion(log_probs, labels)
            total_loss += loss.item()

            preds = log_probs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(all_preds, all_labels, self.num_classes)
        metrics["loss"] = avg_loss
        return metrics

    # ── private: logging ─────────────────────────────────────

    def _log_epoch(self, row: Dict) -> None:
        logger.info(
            "Epoch %3d | lr=%.6f | "
            "train loss=%.4f  acc=%.3f  f1=%.3f | "
            "val   loss=%.4f  acc=%.3f  f1=%.3f | "
            "%.1fs",
            row["epoch"], row["lr"],
            row["train_loss"], row["train_accuracy"], row["train_f1_weighted"],
            row["val_loss"], row["val_accuracy"], row["val_f1_weighted"],
            row["elapsed_sec"],
        )
        # Visual progress bar
        val_f1 = row["val_f1_weighted"]
        bar_len = 40
        filled = int(bar_len * val_f1)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"  Epoch {row['epoch']:3d}  "
            f"val_f1={val_f1:.4f}  [{bar}]  "
            f"lr={row['lr']:.2e}",
            flush=True,
        )

    def _write_csv(self, row: Dict) -> None:
        if self._csv_writer is None:
            fieldnames = [k for k in row if not isinstance(row[k], list)]
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=fieldnames
            )
            self._csv_writer.writeheader()
        clean = {k: v for k, v in row.items() if not isinstance(v, list)}
        self._csv_writer.writerow(clean)
        self._csv_file.flush()
