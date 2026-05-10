"""
audio/preprocessor.py
─────────────────────
Signal-level preprocessing applied before feature extraction:

  1. DC-offset removal      – subtract mean to eliminate low-frequency drift
  2. Noise gate             – zero out frames below a silence threshold
  3. RMS normalisation      – bring signal to a target loudness level
  4. Peak clip guard        – hard-limit any samples outside [-1, 1]
  5. Windowing / segmenting – chop a long AudioBuffer into overlapping frames
                              ready for STFT / feature extraction

All operations are stateless pure functions operating on numpy arrays,
plus the Preprocessor class that chains them using config values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import librosa
import numpy as np

from audio.loader import AudioBuffer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Result containers
# ─────────────────────────────────────────────────────────────

@dataclass
class PreprocessedAudio:
    """
    Output of Preprocessor.process().

    Attributes
    ----------
    samples     : cleaned, normalised float32 signal (N,)
    sample_rate : Hz
    source      : inherited from the input AudioBuffer
    rms_db      : measured RMS level before normalisation (dBFS)
    peak        : peak absolute sample value before normalisation
    is_silent   : True if the entire buffer was below the silence gate
    """
    samples: np.ndarray
    sample_rate: int
    source: str
    rms_db: float
    peak: float
    is_silent: bool

    def __repr__(self) -> str:
        tag = "SILENT" if self.is_silent else "OK"
        return (
            f"PreprocessedAudio({tag}, src={self.source!r}, "
            f"rms={self.rms_db:.1f} dBFS, peak={self.peak:.4f})"
        )


@dataclass
class AudioSegment:
    """
    A single analysis window cut from a longer PreprocessedAudio.

    Attributes
    ----------
    samples     : float32 array, shape (window_samples,)
    sample_rate : Hz
    start_sec   : time of the window's first sample (seconds)
    end_sec     : time of the window's last sample
    segment_idx : zero-based index in the segment sequence
    source      : inherited label
    """
    samples: np.ndarray
    sample_rate: int
    start_sec: float
    end_sec: float
    segment_idx: int
    source: str

    def __repr__(self) -> str:
        return (
            f"AudioSegment(idx={self.segment_idx}, "
            f"{self.start_sec:.2f}s–{self.end_sec:.2f}s, "
            f"src={self.source!r})"
        )


# ─────────────────────────────────────────────────────────────
#  Pure functional helpers
# ─────────────────────────────────────────────────────────────

def remove_dc_offset(samples: np.ndarray) -> np.ndarray:
    """Subtract the mean — removes any DC bias introduced by ADC hardware."""
    return samples - samples.mean()


def compute_rms_db(samples: np.ndarray, eps: float = 1e-10) -> float:
    """RMS level in dBFS (0 dBFS = full scale)."""
    rms = np.sqrt(np.mean(samples ** 2) + eps)
    return float(20.0 * np.log10(rms))


def is_silent(samples: np.ndarray, threshold_db: float = -60.0) -> bool:
    """Return True if RMS is below threshold_db."""
    return compute_rms_db(samples) < threshold_db


def apply_noise_gate(
    samples: np.ndarray,
    threshold_db: float = -60.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Frame-wise noise gate: frames below threshold_db are zeroed out.

    This preserves the original array length and avoids artefacts
    from hard sample-by-sample gating.
    """
    out = samples.copy()
    n = len(samples)
    for start in range(0, n - frame_length + 1, hop_length):
        frame = samples[start : start + frame_length]
        if compute_rms_db(frame) < threshold_db:
            out[start : start + frame_length] = 0.0
    return out


def normalise_rms(
    samples: np.ndarray,
    target_db: float = -20.0,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Scale samples so their RMS equals target_db (dBFS).

    Safe for silent/near-silent audio (eps prevents divide-by-zero).
    """
    current_rms = np.sqrt(np.mean(samples ** 2) + eps)
    target_rms = 10 ** (target_db / 20.0)
    gain = target_rms / current_rms
    return np.clip(samples * gain, -1.0, 1.0)


def peak_clip_guard(samples: np.ndarray) -> np.ndarray:
    """Hard-limit any values outside [-1, 1] after normalisation."""
    return np.clip(samples, -1.0, 1.0)


def segment_audio(
    samples: np.ndarray,
    sample_rate: int,
    window_sec: float = 3.0,
    overlap: float = 0.5,
    source: str = "",
    pad_last: bool = True,
) -> List[AudioSegment]:
    """
    Split a 1-D signal into overlapping fixed-length windows.

    Parameters
    ----------
    samples     : float32 array (N,)
    sample_rate : Hz
    window_sec  : window duration in seconds
    overlap     : fraction of window that overlaps with the next [0, 1)
    source      : label string for each returned segment
    pad_last    : if True, zero-pad the final segment to full length

    Returns
    -------
    List of AudioSegment objects, one per window
    """
    window_len = int(sample_rate * window_sec)
    hop_len = int(window_len * (1.0 - overlap))

    if hop_len <= 0:
        raise ValueError(f"overlap={overlap} is too large — hop would be <= 0")
    if window_len > len(samples):
        raise ValueError(
            f"window_sec={window_sec}s ({window_len} samples) exceeds "
            f"audio length {len(samples)/sample_rate:.2f}s"
        )

    segments: List[AudioSegment] = []
    idx = 0
    start = 0

    while start + window_len <= len(samples):
        chunk = samples[start : start + window_len].astype(np.float32)
        segments.append(
            AudioSegment(
                samples=chunk,
                sample_rate=sample_rate,
                start_sec=start / sample_rate,
                end_sec=(start + window_len) / sample_rate,
                segment_idx=idx,
                source=source,
            )
        )
        start += hop_len
        idx += 1

    # Handle the tail (partial last window)
    if pad_last and start < len(samples):
        tail = samples[start:]
        padded = np.zeros(window_len, dtype=np.float32)
        padded[: len(tail)] = tail
        segments.append(
            AudioSegment(
                samples=padded,
                sample_rate=sample_rate,
                start_sec=start / sample_rate,
                end_sec=(start + window_len) / sample_rate,
                segment_idx=idx,
                source=source,
            )
        )

    return segments


# ─────────────────────────────────────────────────────────────
#  Preprocessor class
# ─────────────────────────────────────────────────────────────

class Preprocessor:
    """
    Chains all preprocessing steps into a single .process() call.

    Parameters
    ----------
    norm_target_db       : RMS normalisation target level (dBFS)
    silence_threshold_db : frames below this level are zeroed (noise gate)
    frame_length         : STFT frame size for noise gate computation
    hop_length           : hop between frames for noise gate
    window_sec           : segmentation window length (seconds)
    overlap              : segmentation overlap fraction
    verbose              : whether to emit INFO logs

    Example
    -------
    prep = Preprocessor()
    result = prep.process(audio_buffer)          # → PreprocessedAudio
    segments = prep.segment(result)              # → List[AudioSegment]
    """

    def __init__(
        self,
        norm_target_db: float = -20.0,
        silence_threshold_db: float = -60.0,
        frame_length: int = 2048,
        hop_length: int = 512,
        window_sec: float = 3.0,
        overlap: float = 0.5,
        verbose: bool = True,
    ) -> None:
        self.norm_target_db = norm_target_db
        self.silence_threshold_db = silence_threshold_db
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.window_sec = window_sec
        self.overlap = overlap
        self.verbose = verbose

    # ── public ──────────────────────────────────────────────

    def process(self, buf: AudioBuffer) -> PreprocessedAudio:
        """
        Apply the full preprocessing chain to an AudioBuffer.

        Steps (in order):
          1. DC-offset removal
          2. Measure raw RMS & peak
          3. Noise gate
          4. RMS normalisation
          5. Peak clip guard

        Returns PreprocessedAudio with is_silent=True if the entire
        buffer was silent after gating.
        """
        s = buf.samples.astype(np.float32)

        # 1 — DC offset
        s = remove_dc_offset(s)

        # 2 — measure before normalisation
        raw_rms_db = compute_rms_db(s)
        raw_peak = float(np.abs(s).max())

        # 3 — noise gate
        s = apply_noise_gate(
            s,
            threshold_db=self.silence_threshold_db,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )

        # 4 — check silence
        silent = is_silent(s, threshold_db=self.silence_threshold_db)

        # 5 — RMS normalise (skip if silent to avoid amplifying noise)
        if not silent:
            s = normalise_rms(s, target_db=self.norm_target_db)

        # 6 — clip guard
        s = peak_clip_guard(s)

        result = PreprocessedAudio(
            samples=s,
            sample_rate=buf.sample_rate,
            source=buf.source,
            rms_db=raw_rms_db,
            peak=raw_peak,
            is_silent=silent,
        )

        if self.verbose:
            logger.info("Preprocessed: %s", result)

        return result

    def segment(self, processed: PreprocessedAudio) -> List[AudioSegment]:
        """
        Segment preprocessed audio into overlapping windows.

        Returns empty list if audio is silent.
        """
        if processed.is_silent:
            logger.warning("Skipping segmentation — audio is silent.")
            return []

        segments = segment_audio(
            samples=processed.samples,
            sample_rate=processed.sample_rate,
            window_sec=self.window_sec,
            overlap=self.overlap,
            source=processed.source,
        )

        if self.verbose:
            logger.info(
                "Segmented into %d windows (%.1fs, %.0f%% overlap)",
                len(segments), self.window_sec, self.overlap * 100,
            )

        return segments

    def process_and_segment(self, buf: AudioBuffer) -> List[AudioSegment]:
        """Convenience: process + segment in one call."""
        processed = self.process(buf)
        return self.segment(processed)
