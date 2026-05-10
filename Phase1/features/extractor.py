"""
features/extractor.py
─────────────────────
Extracts all acoustic features described in the paper from an AudioSegment:

  ┌─────────────────────────────────────────────────────────────┐
  │  Feature group        │  Dim  │  Description                │
  ├─────────────────────────────────────────────────────────────┤
  │  MFCC (+ Δ + ΔΔ)      │  120  │  Timbre / phonetic shape    │
  │  Chroma               │   12  │  Pitch-class distribution   │
  │  Spectral contrast    │    7  │  Texture / brightness       │
  │  Spectral centroid    │    1  │  Tonal centre of mass       │
  │  Spectral bandwidth   │    1  │  Spread around centroid     │
  │  Spectral rolloff     │    1  │  High-frequency energy      │
  │  Tempo                │    1  │  BPM estimate               │
  │  RMS energy           │    1  │  Loudness                   │
  │  Zero-crossing rate   │    1  │  Noisiness / percussiveness │
  ├─────────────────────────────────────────────────────────────┤
  │  Total (default)      │  145  │                             │
  └─────────────────────────────────────────────────────────────┘

Each feature is computed per frame and then summarised to a single
fixed-size vector via mean + std pooling (so segment length doesn't
matter — the output shape is always the same).

The FeatureExtractor is fully configurable via keyword args or a
config dict loaded from config.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

from audio.preprocessor import AudioSegment

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────────────────────

@dataclass
class FeatureVector:
    """
    Output of FeatureExtractor.extract().

    Attributes
    ----------
    vector      : 1-D float32 numpy array, shape (D,)
    feature_map : ordered dict mapping feature name → slice in vector
    segment_idx : which AudioSegment this came from
    start_sec   : segment start time
    end_sec     : segment end time
    source      : inherited audio source label
    """
    vector: np.ndarray
    feature_map: Dict[str, Tuple[int, int]]   # name → (start, end) index
    segment_idx: int
    start_sec: float
    end_sec: float
    source: str
    metadata: dict = field(default_factory=dict)

    # ── convenience ─────────────────────────────────────────

    def get(self, name: str) -> np.ndarray:
        """Return the sub-vector for a named feature group."""
        start, end = self.feature_map[name]
        return self.vector[start:end]

    @property
    def dim(self) -> int:
        return len(self.vector)

    def __repr__(self) -> str:
        groups = ", ".join(
            f"{k}[{e-s}]" for k, (s, e) in self.feature_map.items()
        )
        return (
            f"FeatureVector(dim={self.dim}, "
            f"idx={self.segment_idx}, "
            f"{self.start_sec:.2f}s–{self.end_sec:.2f}s, "
            f"groups=[{groups}])"
        )


# ─────────────────────────────────────────────────────────────
#  Core extraction helpers (pure functions)
# ─────────────────────────────────────────────────────────────

def _pool(frames: np.ndarray) -> np.ndarray:
    """
    Summarise a 2-D frame matrix (features × time) into a 1-D vector
    via mean and standard deviation pooling.

    Shape: (K, T) → (2K,)
    """
    mean = np.mean(frames, axis=1)
    std = np.std(frames, axis=1)
    return np.concatenate([mean, std]).astype(np.float32)


def extract_mfcc(
    samples: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    include_delta: bool = True,
    include_delta2: bool = True,
) -> np.ndarray:
    """
    Mel-Frequency Cepstral Coefficients.

    With delta and delta-delta:
      dim = 2 * (n_mfcc * 3) when include_delta=include_delta2=True
          = 2 * n_mfcc       otherwise
    (factor 2 from mean+std pooling)
    """
    mfcc = librosa.feature.mfcc(
        y=samples, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )  # (n_mfcc, T)

    parts = [mfcc]
    if include_delta:
        parts.append(librosa.feature.delta(mfcc, order=1))
    if include_delta2:
        parts.append(librosa.feature.delta(mfcc, order=2))

    stacked = np.vstack(parts)  # (n_mfcc * [1,2,3], T)
    return _pool(stacked)


def extract_chroma(
    samples: np.ndarray,
    sr: int,
    n_chroma: int = 12,
    n_fft: int = 2048,
    hop_length: int = 512,
    norm=np.inf,
) -> np.ndarray:
    """
    Chroma features — energy distribution across the 12 pitch classes.
    dim = 2 * n_chroma = 24
    """
    chroma = librosa.feature.chroma_stft(
        y=samples, sr=sr, n_chroma=n_chroma,
        n_fft=n_fft, hop_length=hop_length, norm=norm,
    )  # (12, T)
    return _pool(chroma)


def extract_spectral_contrast(
    samples: np.ndarray,
    sr: int,
    n_bands: int = 6,
    fmin: float = 200.0,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Spectral contrast — difference between peaks and valleys in each sub-band.
    Captures textural "brightness" and harmonic structure.
    dim = 2 * (n_bands + 1) = 14
    """
    contrast = librosa.feature.spectral_contrast(
        y=samples, sr=sr, n_bands=n_bands, fmin=fmin,
        n_fft=n_fft, hop_length=hop_length,
    )  # (n_bands+1, T)
    return _pool(contrast)


def extract_spectral_features(
    samples: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    rolloff_percent: float = 0.85,
    include_centroid: bool = True,
    include_bandwidth: bool = True,
    include_rolloff: bool = True,
) -> np.ndarray:
    """
    Three scalar spectral descriptors (mean + std each):
      - Spectral centroid   : frequency centre of mass
      - Spectral bandwidth  : spread around centroid
      - Spectral rolloff    : frequency below which X% of energy lies

    dim = 2 * (number of enabled descriptors) ∈ {2, 4, 6}
    """
    parts: list[np.ndarray] = []

    if include_centroid:
        c = librosa.feature.spectral_centroid(
            y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length
        )  # (1, T)
        parts.append(c)

    if include_bandwidth:
        b = librosa.feature.spectral_bandwidth(
            y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        parts.append(b)

    if include_rolloff:
        r = librosa.feature.spectral_rolloff(
            y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length,
            roll_percent=rolloff_percent,
        )
        parts.append(r)

    if not parts:
        return np.zeros(0, dtype=np.float32)

    stacked = np.vstack(parts)  # (K, T)
    return _pool(stacked)


def extract_tempo_energy(
    samples: np.ndarray,
    sr: int,
    hop_length: int = 512,
    include_tempo: bool = True,
    include_rms: bool = True,
    include_zcr: bool = True,
) -> np.ndarray:
    """
    Rhythm and energy descriptors:
      - Tempo (BPM)         : 1 scalar (no std — single estimate)
      - RMS energy          : mean + std per frame
      - Zero-crossing rate  : mean + std per frame

    dim = 1 (tempo) + 2 (rms) + 2 (zcr)  →  5  (all enabled)
    """
    parts: list[np.ndarray] = []

    if include_tempo:
            try:
                tempo, _ = librosa.beat.beat_track(y=samples, sr=sr, hop_length=hop_length)
                # librosa ≥0.10 returns a 0-d array; flatten to scalar safely
                tempo_val = float(np.atleast_1d(tempo)[0])
            except (AttributeError, Exception):
                # scipy.signal.hann removed in newer scipy — fall back to 0 BPM
                tempo_val = 0.0
            parts.append(np.array([tempo_val], dtype=np.float32))

    if include_rms:
        rms = librosa.feature.rms(y=samples, hop_length=hop_length)  # (1, T)
        parts.append(_pool(rms))

    if include_zcr:
        zcr = librosa.feature.zero_crossing_rate(samples, hop_length=hop_length)
        parts.append(_pool(zcr))

    if not parts:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(parts).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  FeatureExtractor class
# ─────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Orchestrates all feature groups into one fixed-size vector per segment.

    Default output shape: (145,) — see module docstring for breakdown.

    Parameters
    ----------
    n_mfcc           : number of MFCC coefficients
    include_delta    : append MFCC Δ (first-order derivative)
    include_delta2   : append MFCC ΔΔ (second-order derivative)
    n_chroma         : chroma bins
    n_contrast_bands : spectral contrast sub-bands
    fmin_contrast    : lowest spectral contrast band centre (Hz)
    rolloff_percent  : spectral rolloff threshold fraction
    include_centroid : include spectral centroid
    include_bandwidth: include spectral bandwidth
    include_rolloff  : include spectral rolloff
    include_tempo    : include tempo (BPM)
    include_rms      : include RMS energy
    include_zcr      : include zero-crossing rate
    frame_length     : STFT window size (samples)
    hop_length       : STFT hop size (samples)
    normalize        : z-score normalise the final vector
    verbose          : log extraction info

    Example
    -------
    extractor = FeatureExtractor()
    fv = extractor.extract(segment)
    print(fv)          # FeatureVector(dim=145, …)
    print(fv.get("mfcc").shape)   # (120,)
    """

    def __init__(
        self,
        n_mfcc: int = 40,
        include_delta: bool = True,
        include_delta2: bool = True,
        n_chroma: int = 12,
        n_contrast_bands: int = 6,
        fmin_contrast: float = 200.0,
        rolloff_percent: float = 0.85,
        include_centroid: bool = True,
        include_bandwidth: bool = True,
        include_rolloff: bool = True,
        include_tempo: bool = True,
        include_rms: bool = True,
        include_zcr: bool = True,
        frame_length: int = 2048,
        hop_length: int = 512,
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:
        self.n_mfcc = n_mfcc
        self.include_delta = include_delta
        self.include_delta2 = include_delta2
        self.n_chroma = n_chroma
        self.n_contrast_bands = n_contrast_bands
        self.fmin_contrast = fmin_contrast
        self.rolloff_percent = rolloff_percent
        self.include_centroid = include_centroid
        self.include_bandwidth = include_bandwidth
        self.include_rolloff = include_rolloff
        self.include_tempo = include_tempo
        self.include_rms = include_rms
        self.include_zcr = include_zcr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.normalize = normalize
        self.verbose = verbose

    # ── public ──────────────────────────────────────────────

    def extract(self, segment: AudioSegment) -> FeatureVector:
        """
        Extract all enabled features from a single AudioSegment.

        Returns FeatureVector with .vector (float32, shape (D,))
        and .feature_map for named sub-vector lookup.
        """
        s = segment.samples
        sr = segment.sample_rate
        parts: list[np.ndarray] = []
        fmap: dict[str, tuple[int, int]] = {}
        cursor = 0

        def _add(name: str, vec: np.ndarray) -> None:
            nonlocal cursor
            parts.append(vec)
            fmap[name] = (cursor, cursor + len(vec))
            cursor += len(vec)

        # ── MFCC ────────────────────────────────────────────
        mfcc_vec = extract_mfcc(
            s, sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            include_delta=self.include_delta,
            include_delta2=self.include_delta2,
        )
        _add("mfcc", mfcc_vec)

        # ── Chroma ──────────────────────────────────────────
        chroma_vec = extract_chroma(
            s, sr,
            n_chroma=self.n_chroma,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        _add("chroma", chroma_vec)

        # ── Spectral contrast ────────────────────────────────
        contrast_vec = extract_spectral_contrast(
            s, sr,
            n_bands=self.n_contrast_bands,
            fmin=self.fmin_contrast,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        _add("spectral_contrast", contrast_vec)

        # ── Spectral centroid / bandwidth / rolloff ──────────
        spec_vec = extract_spectral_features(
            s, sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            rolloff_percent=self.rolloff_percent,
            include_centroid=self.include_centroid,
            include_bandwidth=self.include_bandwidth,
            include_rolloff=self.include_rolloff,
        )
        if len(spec_vec) > 0:
            _add("spectral", spec_vec)

        # ── Tempo / energy ───────────────────────────────────
        te_vec = extract_tempo_energy(
            s, sr,
            hop_length=self.hop_length,
            include_tempo=self.include_tempo,
            include_rms=self.include_rms,
            include_zcr=self.include_zcr,
        )
        if len(te_vec) > 0:
            _add("tempo_energy", te_vec)

        # ── Assemble ─────────────────────────────────────────
        vector = np.concatenate(parts).astype(np.float32)

        if self.normalize:
            vector = self._zscore(vector)

        fv = FeatureVector(
            vector=vector,
            feature_map=fmap,
            segment_idx=segment.segment_idx,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            source=segment.source,
        )

        if self.verbose:
            logger.debug("Extracted: %s", fv)

        return fv

    def extract_batch(
        self, segments: List[AudioSegment]
    ) -> List[FeatureVector]:
        """Extract features from a list of segments."""
        results = []
        for seg in segments:
            try:
                fv = self.extract(seg)
                results.append(fv)
            except Exception as exc:
                logger.warning(
                    "Skipping segment %d — extraction failed: %s",
                    seg.segment_idx, exc,
                )
        if self.verbose:
            logger.info(
                "Extracted %d / %d segments (dim=%d)",
                len(results), len(segments),
                results[0].dim if results else 0,
            )
        return results

    @property
    def output_dim(self) -> int:
        """
        Return the expected feature vector size without running extraction.
        Useful for pre-allocating model input layers.
        """
        mfcc_coeff = self.n_mfcc * (
            1
            + int(self.include_delta)
            + int(self.include_delta2)
        )
        mfcc_dim = mfcc_coeff * 2   # mean + std

        chroma_dim = self.n_chroma * 2

        contrast_dim = (self.n_contrast_bands + 1) * 2

        spectral_dim = 2 * (
            int(self.include_centroid)
            + int(self.include_bandwidth)
            + int(self.include_rolloff)
        )

        te_dim = (
            int(self.include_tempo)          # 1 (scalar BPM)
            + int(self.include_rms) * 2      # mean + std
            + int(self.include_zcr) * 2      # mean + std
        )

        return mfcc_dim + chroma_dim + contrast_dim + spectral_dim + te_dim

    # ── private ─────────────────────────────────────────────

    @staticmethod
    def _zscore(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Z-score normalise a vector to zero mean, unit variance."""
        mean = vec.mean()
        std = vec.std() + eps
        return ((vec - mean) / std).astype(np.float32)

    @classmethod
    def from_config(cls, cfg: dict) -> "FeatureExtractor":
        """
        Build a FeatureExtractor from the 'features' section of config.yaml.

        Example
        -------
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        extractor = FeatureExtractor.from_config(cfg["features"])
        """
        mfcc = cfg.get("mfcc", {})
        chroma = cfg.get("chroma", {})
        contrast = cfg.get("spectral_contrast", {})
        spectral = cfg.get("spectral", {})
        te = cfg.get("tempo_energy", {})
        pre = cfg.get("preprocessing", {})

        return cls(
            n_mfcc=mfcc.get("n_mfcc", 40),
            include_delta=mfcc.get("include_delta", True),
            include_delta2=mfcc.get("include_delta2", True),
            n_chroma=chroma.get("n_chroma", 12),
            n_contrast_bands=contrast.get("n_bands", 6),
            fmin_contrast=contrast.get("fmin", 200.0),
            rolloff_percent=spectral.get("rolloff_percent", 0.85),
            include_centroid=spectral.get("include_centroid", True),
            include_bandwidth=spectral.get("include_bandwidth", True),
            include_rolloff=spectral.get("include_rolloff", True),
            include_tempo=te.get("include_tempo", True),
            include_rms=te.get("include_rms_energy", True),
            include_zcr=te.get("include_zcr", True),
            frame_length=pre.get("frame_length", 2048),
            hop_length=pre.get("hop_length", 512),
        )
