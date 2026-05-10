"""
train.py
────────
Main entry point for Phase 2 training.

Usage
─────
  # Quickstart with synthetic data (no dataset needed):
  python train.py

  # With real audio dataset:
  python train.py --data-dir data/raw

  # With a custom config:
  python train.py --config config.yaml --epochs 30

  # Evaluate a saved checkpoint:
  python train.py --eval-only --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

from data.dataset import build_dataloaders, EMOTIONS
from model.cnn_lstm import CNNLSTMEmotionClassifier
from training.trainer import Trainer
from evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def override_config(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides onto the loaded config."""
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.data_dir:
        cfg["dataset"]["raw_dir"] = args.data_dir
    if args.synthetic:
        cfg["dataset"]["use_synthetic"] = True
    return cfg


def build_model(cfg: dict) -> CNNLSTMEmotionClassifier:
    model = CNNLSTMEmotionClassifier.from_config(cfg)
    model.summary()
    logger.info("Model parameters: %d", model.count_parameters())
    return model


def run_training(cfg: dict, args: argparse.Namespace) -> None:
    logger.info("─── Phase 2: CNN-LSTM Emotion Classifier ───")

    # ── dataloaders ──────────────────────────────────────────
    logger.info("Building dataloaders…")
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, raw_dir=cfg["dataset"]["raw_dir"]
    )

    # ── model ────────────────────────────────────────────────
    model = build_model(cfg)

    # ── trainer ──────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, cfg)

    # Resume from checkpoint if given
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── fit ──────────────────────────────────────────────────
    history = trainer.fit()

    # ── test evaluation ──────────────────────────────────────
    logger.info("Running final evaluation on test set…")
    best_ckpt = Path(cfg["training"]["checkpoint_dir"]) / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded best checkpoint (epoch %d)", ckpt["epoch"])

    evaluator = Evaluator(model, test_loader, class_names=EMOTIONS)
    metrics = evaluator.run()

    print("\nClassification Report:\n")
    print(evaluator.class_report())

    evaluator.plot_confusion_matrix()
    evaluator.plot_training_history()
    evaluator.check_target(cfg["evaluation"]["target_f1"])

    logger.info("Done. Checkpoints in: %s", cfg["training"]["checkpoint_dir"])
    logger.info("Training log   in: %s", cfg["training"]["log_dir"])


def run_eval_only(cfg: dict, args: argparse.Namespace) -> None:
    """Load a checkpoint and evaluate on the test set only."""
    logger.info("─── Evaluation only mode ───")

    if not args.checkpoint:
        logger.error("--checkpoint is required for --eval-only")
        sys.exit(1)

    _, _, test_loader = build_dataloaders(cfg)

    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    evaluator = Evaluator(model, test_loader, class_names=EMOTIONS)
    metrics = evaluator.run()
    print(evaluator.class_report())
    evaluator.plot_confusion_matrix()
    evaluator.check_target(cfg["evaluation"]["target_f1"])


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train",
        description="Phase 2 — CNN-LSTM Emotion Model Training",
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--data-dir", default=None, help="Path to raw audio folder")
    p.add_argument("--synthetic", action="store_true",
                   help="Force synthetic data generation")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", default=None, help="Resume from checkpoint path")
    p.add_argument("--checkpoint", default=None, help="Checkpoint for eval-only mode")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="5-epoch smoke test with minimal synthetic data")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_config(args.config)
    cfg = override_config(cfg, args)

    if args.quick:
        cfg["training"]["epochs"] = 5
        cfg["training"]["batch_size"] = 16
        cfg["dataset"]["synthetic_samples"] = 30
        cfg["dataset"]["use_synthetic"] = True
        cfg["training"]["early_stopping_patience"] = 99
        logger.info("Quick mode: 5 epochs, 30 synthetic samples per class")

    if args.eval_only:
        run_eval_only(cfg, args)
    else:
        run_training(cfg, args)


if __name__ == "__main__":
    main()
