"""
data/dataset.py
───────────────
Handles everything between raw audio files and PyTorch DataLoaders:

  1. EmotionDataset      – loads mel-spectrograms + feature vectors per segment
  2. SyntheticGenerator  – creates labelled synthetic audio when no real data exists
  3. SpecAugment         – frequency / time masking for training regularisation
  4. DataAugmentor       – time-stretch, pitch-shift, additive noise
  5. build_dataloaders() – returns train/val/test DataLoaders ready for training

Expected raw data layout (real audio):
  data/raw/
    happy/    *.mp3  *.wav …
    sad/      …
    calm/     …
    angry/    …
    energetic/…

Synthetic mode (config.dataset.use_synthetic: true) generates
per-class audio with characteristic tempo/pitch profiles so the
model has something to train on without needing a dataset download.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

# ── Emotion label mapping ────────────────────────────────────
EMOTIONS = ["happy", "sad", "calm", "angry", "energetic"]
LABEL2IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX2LABEL = {i: e for e, i in LABEL2IDX.items()}


# ─────────────────────────────────────────────────────────────
#  SpecAugment
# ─────────────────────────────────────────────────────────────

class SpecAugment:
    """
    SpecAugment: frequency and time masking applied to mel-spectrograms.
    Park et al. (2019) — https://arxiv.org/abs/1904.08779
    """

    def __init__(
        self,
        freq_mask_param: int = 20,
        time_mask_param: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ) -> None:
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply random frequency and time masks.

        Parameters
        ----------
        spec : (freq_bins, time_steps) float32 array

        Returns
        -------
        Augmented spectrogram, same shape
        """
        spec = spec.copy()
        n_freq, n_time = spec.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, n_freq - f))
            spec[f0:f0 + f, :] = 0.0

        # Time masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(0, n_time - t))
            spec[:, t0:t0 + t] = 0.0

        return spec


# ─────────────────────────────────────────────────────────────
#  Audio augmentor
# ─────────────────────────────────────────────────────────────

class DataAugmentor:
    """
    Waveform-level augmentation: time stretch, pitch shift, additive noise.
    Applied randomly during training — each transform fires with p=0.5.
    """

    def __init__(
        self,
        time_stretch_range: Tuple[float, float] = (0.85, 1.15),
        pitch_shift_range: Tuple[int, int] = (-2, 2),
        noise_factor: float = 0.005,
        sr: int = 22050,
    ) -> None:
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.noise_factor = noise_factor
        self.sr = sr

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        # Time stretch
        if random.random() < 0.5:
            rate = random.uniform(*self.time_stretch_range)
            try:
                samples = librosa.effects.time_stretch(samples, rate=rate)
            except Exception:
                pass

        # Pitch shift
        if random.random() < 0.5:
            n_steps = random.randint(*self.pitch_shift_range)
            try:
                samples = librosa.effects.pitch_shift(
                    samples, sr=self.sr, n_steps=n_steps
                )
            except Exception:
                pass

        # Additive white noise
        if random.random() < 0.5:
            noise = np.random.randn(len(samples)).astype(np.float32)
            samples = samples + self.noise_factor * noise

        return np.clip(samples, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  Mel-spectrogram helper
# ─────────────────────────────────────────────────────────────

def compute_melspec(
    samples: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    f_min: float = 20.0,
    f_max: float = 8000.0,
    target_frames: int = 130,
) -> np.ndarray:
    """
    Compute a log-power mel-spectrogram and resize to a fixed width.

    Returns
    -------
    float32 array, shape (n_mels, target_frames), values in [0, 1]
    """
    mel = librosa.feature.melspectrogram(
        y=samples, sr=sr,
        n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
        fmin=f_min, fmax=f_max,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)

    # Resize time axis to target_frames by interpolation
    if mel_db.shape[1] != target_frames:
        from scipy.ndimage import zoom
        factor = target_frames / mel_db.shape[1]
        mel_db = zoom(mel_db, (1.0, factor), order=1)

    # Normalise to [0, 1]
    mel_db = mel_db - mel_db.min()
    denom = mel_db.max()
    if denom > 0:
        mel_db = mel_db / denom

    return mel_db.astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  Synthetic data generator
# ─────────────────────────────────────────────────────────────

class SyntheticGenerator:
    """
    Generate labelled synthetic audio clips with emotion-characteristic profiles.

    Each emotion has a distinct BPM, pitch, harmonic richness, and noise level
    so the model can learn real separating features even without a real dataset.

    Emotion profiles
    ----------------
    happy     : fast tempo (130 BPM), bright harmonics, medium energy
    sad       : slow tempo (60 BPM), low pitch, few harmonics, soft
    calm      : medium-slow (75 BPM), sine tones, very little noise
    angry     : fast (160 BPM), distorted, high noise, lots of harmonics
    energetic : very fast (145 BPM), full spectrum, high energy
    """

    PROFILES = {
        "happy":     dict(bpm=130, base_freq=523.25, n_harmonics=5, noise=0.02,  amp=0.55),
        "sad":       dict(bpm=60,  base_freq=261.63, n_harmonics=2, noise=0.005, amp=0.30),
        "calm":      dict(bpm=75,  base_freq=392.00, n_harmonics=3, noise=0.003, amp=0.35),
        "angry":     dict(bpm=160, base_freq=329.63, n_harmonics=8, noise=0.08,  amp=0.80),
        "energetic": dict(bpm=145, base_freq=440.00, n_harmonics=6, noise=0.04,  amp=0.70),
    }

    def __init__(
        self,
        sr: int = 22050,
        duration: float = 3.0,
        output_dir: str | Path = "data/raw",
    ) -> None:
        self.sr = sr
        self.duration = duration
        self.output_dir = Path(output_dir)

    def generate(self, n_per_class: int = 200) -> Dict[str, List[Path]]:
        """
        Generate n_per_class audio clips per emotion.

        Returns
        -------
        Dict mapping emotion name → list of generated file paths
        """
        paths: Dict[str, List[Path]] = {}

        for emotion, profile in self.PROFILES.items():
            emotion_dir = self.output_dir / emotion
            emotion_dir.mkdir(parents=True, exist_ok=True)

            file_paths: List[Path] = []
            for i in range(n_per_class):
                samples = self._generate_clip(emotion, profile, seed=i)
                path = emotion_dir / f"{emotion}_{i:04d}.wav"
                sf.write(str(path), samples, self.sr)
                file_paths.append(path)

            paths[emotion] = file_paths
            logger.info("Generated %d synthetic clips → %s/", n_per_class, emotion)

        return paths

    def _generate_clip(
        self, emotion: str, profile: dict, seed: int
    ) -> np.ndarray:
        rng = np.random.default_rng(seed + LABEL2IDX[emotion] * 1000)
        n = int(self.sr * self.duration)
        t = np.linspace(0, self.duration, n, endpoint=False)

        # Base tone with harmonics
        signal = np.zeros(n, dtype=np.float32)
        base_freq = profile["base_freq"] * (1 + rng.uniform(-0.05, 0.05))
        for h in range(1, profile["n_harmonics"] + 1):
            amp_h = profile["amp"] / h
            phase = rng.uniform(0, 2 * np.pi)
            signal += amp_h * np.sin(2 * np.pi * base_freq * h * t + phase).astype(np.float32)

        # Rhythmic amplitude envelope at BPM
        beat_freq = profile["bpm"] / 60.0
        envelope = 0.7 + 0.3 * np.sin(2 * np.pi * beat_freq * t)
        signal *= envelope.astype(np.float32)

        # Add noise
        signal += (profile["noise"] * rng.standard_normal(n)).astype(np.float32)

        # Slight pitch variation per sample for diversity
        pitch_var = rng.uniform(-1.0, 1.0)
        try:
            signal = librosa.effects.pitch_shift(signal, sr=self.sr, n_steps=pitch_var)
        except Exception:
            pass

        return np.clip(signal, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  EmotionDataset
# ─────────────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    """
    PyTorch Dataset that yields (mel_spectrogram, feature_vector, label) tuples.

    mel_spectrogram : float32 tensor, shape (1, n_mels, target_frames)
                      — CNN input (1 = mono channel)
    feature_vector  : float32 tensor, shape (feature_dim,)
                      — LSTM / auxiliary input
    label           : int64 scalar tensor (0–4)

    Parameters
    ----------
    samples      : list of (audio_path, label_idx) tuples
    sr           : sample rate
    n_mels       : mel bins
    n_fft        : STFT window
    hop_length   : STFT hop
    f_min/f_max  : mel frequency range
    target_frames: fixed time-axis length after interpolation
    augmentor    : DataAugmentor instance (None = no augmentation)
    spec_augment : SpecAugment instance  (None = no spec masking)
    feature_dim  : Phase-1 feature vector dimension
    cache_dir    : if set, pre-computed spectrograms are cached here
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        target_frames: int = 130,
        augmentor: Optional[DataAugmentor] = None,
        spec_augment: Optional[SpecAugment] = None,
        feature_dim: int = 289,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.samples = samples
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.target_frames = target_frames
        self.augmentor = augmentor
        self.spec_augment = spec_augment
        self.feature_dim = feature_dim
        self.cache_dir = cache_dir

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        # ── load waveform ────────────────────────────────────
        try:
            audio, _ = librosa.load(str(path), sr=self.sr, mono=True, duration=self.sr * 4)
            audio = audio.astype(np.float32)
        except Exception as e:
            logger.warning("Failed to load %s: %s — using silence", path, e)
            audio = np.zeros(self.sr * 3, dtype=np.float32)

        # Pad / trim to exactly window_samples
        window_samples = self.sr * 3
        if len(audio) < window_samples:
            audio = np.pad(audio, (0, window_samples - len(audio)))
        else:
            audio = audio[:window_samples]

        # ── waveform augmentation (train only) ───────────────
        if self.augmentor is not None:
            audio = self.augmentor(audio)

        # ── mel-spectrogram ──────────────────────────────────
        mel = compute_melspec(
            audio, sr=self.sr,
            n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length,
            f_min=self.f_min, f_max=self.f_max,
            target_frames=self.target_frames,
        )  # (n_mels, target_frames)

        # ── spec augmentation (train only) ───────────────────
        if self.spec_augment is not None:
            mel = self.spec_augment(mel)

        # Add channel dim → (1, n_mels, target_frames)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)

        # ── feature vector (acoustic features) ───────────────
        feat_vec = self._compute_feature_vector(audio)
        feat_tensor = torch.from_numpy(feat_vec)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return mel_tensor, feat_tensor, label_tensor

    def _compute_feature_vector(self, audio: np.ndarray) -> np.ndarray:
        """Compute a compact acoustic feature vector inline."""
        try:
            # MFCC (mean only — keep it fast during training)
            mfcc = librosa.feature.mfcc(
                y=audio, sr=self.sr, n_mfcc=40,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            mfcc_feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

            # Chroma
            chroma = librosa.feature.chroma_stft(
                y=audio, sr=self.sr, n_chroma=12,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            chroma_feat = np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)])

            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(
                y=audio, sr=self.sr, n_bands=6,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            contrast_feat = np.concatenate([contrast.mean(axis=1), contrast.std(axis=1)])

            # Spectral descriptors
            centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr, hop_length=self.hop_length)
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr, hop_length=self.hop_length)
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr, hop_length=self.hop_length)
            spec_feat = np.array([
                centroid.mean(), centroid.std(),
                bandwidth.mean(), bandwidth.std(),
                rolloff.mean(), rolloff.std(),
            ])

            # RMS + ZCR
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)
            energy_feat = np.array([
                rms.mean(), rms.std(),
                zcr.mean(), zcr.std(),
                0.0,   # tempo placeholder (expensive; skipped for speed)
            ])

            vec = np.concatenate([
                mfcc_feat, chroma_feat, contrast_feat, spec_feat, energy_feat
            ]).astype(np.float32)

        except Exception:
            vec = np.zeros(self.feature_dim, dtype=np.float32)

        # Pad or truncate to exactly feature_dim
        if len(vec) < self.feature_dim:
            vec = np.pad(vec, (0, self.feature_dim - len(vec)))
        else:
            vec = vec[:self.feature_dim]

        # Z-score normalise
        std = vec.std() + 1e-8
        vec = (vec - vec.mean()) / std

        return vec.astype(np.float32)

    @property
    def class_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {i: 0 for i in range(len(EMOTIONS))}
        for _, label in self.samples:
            counts[label] += 1
        return counts

    @property
    def labels(self) -> List[int]:
        return [label for _, label in self.samples]


# ─────────────────────────────────────────────────────────────
#  Dataset builder
# ─────────────────────────────────────────────────────────────

def scan_dataset(raw_dir: str | Path) -> List[Tuple[Path, int]]:
    """
    Scan a folder tree and return (path, label_idx) pairs.

    Expected structure:
      raw_dir/
        <emotion_name>/
          *.wav  *.mp3  …
    """
    raw_dir = Path(raw_dir)
    samples: List[Tuple[Path, int]] = []
    extensions = {".wav", ".mp3", ".flac", ".ogg"}

    for emotion in EMOTIONS:
        emotion_dir = raw_dir / emotion
        if not emotion_dir.exists():
            logger.warning("Missing class folder: %s", emotion_dir)
            continue
        label = LABEL2IDX[emotion]
        found = [
            p for p in emotion_dir.rglob("*")
            if p.suffix.lower() in extensions
        ]
        for p in found:
            samples.append((p, label))
        logger.info("  %-12s: %d files", emotion, len(found))

    return samples


def split_dataset(
    samples: List[Tuple[Path, int]],
    train: float = 0.80,
    val: float = 0.10,
    seed: int = 42,
) -> Tuple[
    List[Tuple[Path, int]],
    List[Tuple[Path, int]],
    List[Tuple[Path, int]],
]:
    """Stratified split into train / val / test."""
    from collections import defaultdict

    rng = random.Random(seed)

    # Group by label
    by_label: Dict[int, List] = defaultdict(list)
    for item in samples:
        by_label[item[1]].append(item)

    train_s, val_s, test_s = [], [], []
    for label_items in by_label.values():
        rng.shuffle(label_items)
        n = len(label_items)
        n_train = int(n * train)
        n_val = int(n * val)
        train_s.extend(label_items[:n_train])
        val_s.extend(label_items[n_train:n_train + n_val])
        test_s.extend(label_items[n_train + n_val:])

    rng.shuffle(train_s)
    return train_s, val_s, test_s


def build_dataloaders(
    cfg: dict,
    raw_dir: Optional[str | Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from config.

    If cfg.dataset.use_synthetic is True and no real data exists,
    synthetic audio is generated first.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    ds_cfg = cfg["dataset"]
    audio_cfg = cfg["audio"]
    feat_cfg = cfg["features"]
    aug_cfg = cfg["augmentation"]
    train_cfg = cfg["training"]

    raw_dir = Path(raw_dir or ds_cfg["raw_dir"])

    # ── generate synthetic data if needed ───────────────────
    if ds_cfg.get("use_synthetic", True):
        existing = list(raw_dir.rglob("*.wav")) + list(raw_dir.rglob("*.mp3"))
        if len(existing) < 10:
            logger.info("Generating synthetic dataset (%d per class)…",
                        ds_cfg["synthetic_samples"])
            gen = SyntheticGenerator(
                sr=audio_cfg["sample_rate"],
                duration=audio_cfg["window_sec"],
                output_dir=raw_dir,
            )
            gen.generate(n_per_class=ds_cfg["synthetic_samples"])

    # ── scan & split ─────────────────────────────────────────
    logger.info("Scanning dataset at %s …", raw_dir)
    all_samples = scan_dataset(raw_dir)
    if not all_samples:
        raise RuntimeError(f"No audio files found in {raw_dir}")

    train_s, val_s, test_s = split_dataset(
        all_samples,
        train=ds_cfg["train_split"],
        val=ds_cfg["val_split"],
        seed=ds_cfg["random_seed"],
    )
    logger.info(
        "Split: train=%d  val=%d  test=%d", len(train_s), len(val_s), len(test_s)
    )

    # ── augmentation objects ─────────────────────────────────
    augmentor = DataAugmentor(
        time_stretch_range=tuple(aug_cfg["time_stretch_range"]),
        pitch_shift_range=tuple(aug_cfg["pitch_shift_range"]),
        noise_factor=aug_cfg["noise_factor"],
        sr=audio_cfg["sample_rate"],
    ) if aug_cfg["enabled"] else None

    spec_aug = SpecAugment(
        freq_mask_param=aug_cfg["spec_augment"]["freq_mask_param"],
        time_mask_param=aug_cfg["spec_augment"]["time_mask_param"],
        num_freq_masks=aug_cfg["spec_augment"]["num_freq_masks"],
        num_time_masks=aug_cfg["spec_augment"]["num_time_masks"],
    ) if aug_cfg["enabled"] else None

    common = dict(
        sr=audio_cfg["sample_rate"],
        n_mels=feat_cfg["n_mels"],
        n_fft=feat_cfg["n_fft"],
        hop_length=feat_cfg["hop_length"],
        f_min=feat_cfg["f_min"],
        f_max=feat_cfg["f_max"],
        feature_dim=feat_cfg["feature_dim"],
    )

    train_ds = EmotionDataset(
        train_s, augmentor=augmentor, spec_augment=spec_aug, **common
    )
    val_ds = EmotionDataset(val_s, **common)
    test_ds = EmotionDataset(test_s, **common)

    # ── weighted sampler for class imbalance ─────────────────
    counts = train_ds.class_counts
    total = sum(counts.values())
    class_weights = {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}
    sample_weights = [class_weights[label] for _, label in train_s]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(train_s),
        replacement=True,
    )

    bs = train_cfg["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    return train_loader, val_loader, test_loader
