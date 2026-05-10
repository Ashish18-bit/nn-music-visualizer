"""
pipeline.py
───────────
Top-level orchestrator for Phase 1.

Wires together:
  AudioFileLoader → Preprocessor → FeatureExtractor → FeatureCache

Single-file usage
─────────────────
    from pipeline import AudioPipeline

    pipe = AudioPipeline()
    vectors = pipe.run("song.mp3")          # returns List[FeatureVector]

    # Or process many files at once:
    results = pipe.run_batch(["a.mp3", "b.wav"])

CLI usage (see __main__ block at bottom)
─────────────────────────────────────────
    python pipeline.py song.mp3
    python pipeline.py song.mp3 --no-cache --plot
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from audio.loader import AudioFileLoader, load_audio
from audio.preprocessor import Preprocessor
from features.extractor import FeatureExtractor, FeatureVector
from features.cache import FeatureCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────

class AudioPipeline:
    """
    End-to-end Phase 1 pipeline.

    Parameters
    ----------
    config_path  : path to config.yaml  (None = use built-in defaults)
    use_cache    : whether to read/write the feature cache
    verbose      : propagate to sub-components
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        use_cache: bool = True,
        verbose: bool = True,
    ) -> None:
        cfg = self._load_config(config_path)

        audio_cfg = cfg.get("audio", {})
        prep_cfg = cfg.get("preprocessing", {})
        seg_cfg = cfg.get("segmentation", {})
        feat_cfg = cfg.get("features", {})
        out_cfg = cfg.get("output", {})

        self.loader = AudioFileLoader(
            target_sr=audio_cfg.get("sample_rate", 22050),
            min_duration_sec=seg_cfg.get("min_duration_sec", 1.0),
            verbose=verbose,
        )

        self.preprocessor = Preprocessor(
            norm_target_db=prep_cfg.get("norm_target_db", -20.0),
            silence_threshold_db=prep_cfg.get("silence_threshold_db", -60),
            frame_length=prep_cfg.get("frame_length", 2048),
            hop_length=prep_cfg.get("hop_length", 512),
            window_sec=seg_cfg.get("window_sec", 3.0),
            overlap=seg_cfg.get("overlap", 0.5),
            verbose=verbose,
        )

        self.extractor = FeatureExtractor(
            n_mfcc=feat_cfg.get("mfcc", {}).get("n_mfcc", 40),
            include_delta=feat_cfg.get("mfcc", {}).get("include_delta", True),
            include_delta2=feat_cfg.get("mfcc", {}).get("include_delta2", True),
            n_chroma=feat_cfg.get("chroma", {}).get("n_chroma", 12),
            n_contrast_bands=feat_cfg.get("spectral_contrast", {}).get("n_bands", 6),
            fmin_contrast=feat_cfg.get("spectral_contrast", {}).get("fmin", 200.0),
            include_centroid=feat_cfg.get("spectral", {}).get("include_centroid", True),
            include_bandwidth=feat_cfg.get("spectral", {}).get("include_bandwidth", True),
            include_rolloff=feat_cfg.get("spectral", {}).get("include_rolloff", True),
            rolloff_percent=feat_cfg.get("spectral", {}).get("rolloff_percent", 0.85),
            include_tempo=feat_cfg.get("tempo_energy", {}).get("include_tempo", True),
            include_rms=feat_cfg.get("tempo_energy", {}).get("include_rms_energy", True),
            include_zcr=feat_cfg.get("tempo_energy", {}).get("include_zcr", True),
            frame_length=prep_cfg.get("frame_length", 2048),
            hop_length=prep_cfg.get("hop_length", 512),
            verbose=verbose,
        )

        save_dir = out_cfg.get("save_dir", "features/cache")
        self.cache = FeatureCache(save_dir=save_dir) if use_cache else None
        self.use_cache = use_cache
        self.verbose = verbose

    # ── public ──────────────────────────────────────────────

    def run(
        self,
        audio_path: str | Path,
        force_recompute: bool = False,
    ) -> List[FeatureVector]:
        """
        Full pipeline for a single audio file.

        Returns
        -------
        List of FeatureVector (one per 3-second window)
        """
        audio_path = Path(audio_path)
        t0 = time.perf_counter()

        # ── cache lookup ─────────────────────────────────────
        cache_key = None
        if self.use_cache and self.cache:
            cache_key = FeatureCache.make_key(audio_path)
            if not force_recompute and self.cache.exists(cache_key):
                vectors = self.cache.load(cache_key)
                if vectors:
                    logger.info(
                        "Cache hit: %d vectors (%.1f ms)",
                        len(vectors), (time.perf_counter() - t0) * 1000,
                    )
                    return vectors

        # ── load ──────────────────────────────────────────────
        buf = self.loader.load(audio_path)

        # ── preprocess + segment ──────────────────────────────
        segments = self.preprocessor.process_and_segment(buf)
        if not segments:
            logger.warning("No segments produced from %s", audio_path.name)
            return []

        # ── feature extraction ────────────────────────────────
        vectors = self.extractor.extract_batch(segments)

        # ── cache save ────────────────────────────────────────
        if self.use_cache and self.cache and cache_key and vectors:
            self.cache.save(vectors, cache_key)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Pipeline done: %d vectors, dim=%d, %.1f ms",
            len(vectors), vectors[0].dim if vectors else 0, elapsed_ms,
        )
        return vectors

    def run_batch(
        self,
        paths: List[str | Path],
        force_recompute: bool = False,
    ) -> Dict[str, List[FeatureVector]]:
        """
        Process multiple audio files.

        Returns
        -------
        Dict mapping path string → List[FeatureVector]
        Files that fail are skipped with a warning logged.
        """
        results: Dict[str, List[FeatureVector]] = {}
        for path in paths:
            try:
                vecs = self.run(path, force_recompute=force_recompute)
                results[str(path)] = vecs
            except Exception as exc:
                logger.warning("Skipping %s — %s", Path(path).name, exc)
        return results

    def summary(self, vectors: List[FeatureVector]) -> None:
        """Print a human-readable summary of extracted feature vectors."""
        if not vectors:
            print("No vectors.")
            return

        fv0 = vectors[0]
        print("\n" + "─" * 60)
        print(f"  Feature vector summary")
        print("─" * 60)
        print(f"  Segments       : {len(vectors)}")
        print(f"  Vector dim     : {fv0.dim}")
        print(f"  Duration       : {vectors[-1].end_sec:.1f}s")
        print(f"  Source         : {fv0.source}")
        print()
        print("  Feature groups:")
        for name, (start, end) in fv0.feature_map.items():
            sub = np.stack([v.vector[start:end] for v in vectors])
            print(
                f"    {name:<22} dim={end-start:>3}  "
                f"mean={sub.mean():+.3f}  std={sub.std():.3f}"
            )
        print("─" * 60 + "\n")

    # ── private ─────────────────────────────────────────────

    @staticmethod
    def _load_config(config_path: Optional[str | Path]) -> dict:
        if config_path is None:
            # Try default location
            default = Path(__file__).parent / "config.yaml"
            if default.exists():
                config_path = default
            else:
                return {}
        with open(config_path) as f:
            return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="Phase 1 — Neural Network Music Visualizer Audio Pipeline",
    )
    p.add_argument("audio", nargs="?", help="Path to audio file (MP3/WAV/FLAC/…)")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--no-cache", action="store_true", help="Disable feature cache")
    p.add_argument("--force", action="store_true", help="Recompute even if cached")
    p.add_argument("--plot", action="store_true", help="Plot features with matplotlib")
    p.add_argument("--generate-tone", action="store_true",
                   help="Generate a synthetic test tone and run pipeline on it")
    p.add_argument("--summary", action="store_true", default=True,
                   help="Print feature summary table (default: on)")
    return p


def _generate_test_tone(
    duration_sec: float = 5.0,
    sr: int = 22050,
    freq: float = 440.0,
    out_path: str = "sample_data/test_tone.wav",
) -> str:
    """Generate a 440 Hz sine wave and save it to disk."""
    import soundfile as sf
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Mix two frequencies for richer content
    tone = (
        0.5 * np.sin(2 * np.pi * freq * t)
        + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.2 * np.sin(2 * np.pi * freq * 3 * t)
    ).astype(np.float32)
    sf.write(out_path, tone, sr)
    logger.info("Generated test tone → %s", out_path)
    return out_path


def _plot_features(vectors: List[FeatureVector]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")
        return

    matrix = np.stack([v.vector for v in vectors])   # (N, D)
    times = [v.start_sec for v in vectors]

    fv0 = vectors[0]
    n_groups = len(fv0.feature_map)
    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 3 * n_groups), sharex=True)
    if n_groups == 1:
        axes = [axes]

    fig.suptitle("Extracted Feature Groups Over Time", fontsize=14, fontweight="bold")

    for ax, (name, (start, end)) in zip(axes, fv0.feature_map.items()):
        sub = matrix[:, start:end]   # (N, K)
        im = ax.imshow(
            sub.T,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], 0, end - start],
            cmap="magma",
        )
        ax.set_ylabel(name, fontsize=10)
        plt.colorbar(im, ax=ax, pad=0.01)

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig("features_plot.png", dpi=120, bbox_inches="tight")
    logger.info("Saved feature plot → features_plot.png")
    plt.show()


def main() -> None:
    args = _build_parser().parse_args()

    pipe = AudioPipeline(
        config_path=args.config,
        use_cache=not args.no_cache,
    )

    audio_path = args.audio
    if args.generate_tone or audio_path is None:
        audio_path = _generate_test_tone()

    vectors = pipe.run(audio_path, force_recompute=args.force)

    if not vectors:
        logger.error("No features extracted. Exiting.")
        sys.exit(1)

    if args.summary:
        pipe.summary(vectors)

    if args.plot:
        _plot_features(vectors)


if __name__ == "__main__":
    main()
