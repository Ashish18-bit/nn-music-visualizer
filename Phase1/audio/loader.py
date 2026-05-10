"""
audio/loader.py
───────────────
Handles all audio ingestion:
  - Load from file  (MP3, WAV, FLAC, OGG, …)
  - Stream from microphone via PyAudio
  - Resample to a target sample rate
  - Convert multi-channel to mono
  - Basic duration / format validation
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Data container
# ─────────────────────────────────────────────────────────────

@dataclass
class AudioBuffer:
    """
    Immutable container for a loaded / captured audio chunk.

    Attributes
    ----------
    samples      : float32 numpy array, shape (N,), values in [-1, 1]
    sample_rate  : native sample rate of the data
    duration_sec : duration in seconds
    source       : human-readable label ('file:<path>' or 'mic')
    """
    samples: np.ndarray
    sample_rate: int
    duration_sec: float
    source: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.samples.ndim == 1, "samples must be 1-D (mono)"
        assert self.samples.dtype == np.float32, "samples must be float32"

    def __repr__(self) -> str:
        return (
            f"AudioBuffer(src={self.source!r}, "
            f"sr={self.sample_rate}, "
            f"dur={self.duration_sec:.2f}s, "
            f"shape={self.samples.shape})"
        )


# ─────────────────────────────────────────────────────────────
#  File loader
# ─────────────────────────────────────────────────────────────

class AudioFileLoader:
    """
    Load audio from a file on disk.

    Supports any format that librosa / soundfile can read:
    WAV, MP3, FLAC, OGG, M4A, AIFF, …

    Parameters
    ----------
    target_sr        : resample to this sample rate (Hz)
    min_duration_sec : reject files shorter than this
    verbose          : log INFO messages
    """

    SUPPORTED_EXTENSIONS = {
        ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif", ".opus"
    }

    def __init__(
        self,
        target_sr: int = 22050,
        min_duration_sec: float = 1.0,
        verbose: bool = True,
    ) -> None:
        self.target_sr = target_sr
        self.min_duration_sec = min_duration_sec
        self.verbose = verbose

    # ── public ──────────────────────────────────────────────

    def load(self, path: str | Path) -> AudioBuffer:
        """
        Load, validate, resample, and mono-mix a file.

        Returns
        -------
        AudioBuffer with samples at self.target_sr

        Raises
        ------
        FileNotFoundError   : path does not exist
        ValueError          : unsupported format or too short
        RuntimeError        : librosa decode failure
        """
        path = Path(path).resolve()
        self._validate_path(path)

        if self.verbose:
            logger.info("Loading %s", path.name)

        samples, native_sr = self._decode(path)
        samples = self._to_mono(samples)
        samples = self._resample(samples, native_sr)
        samples = self._to_float32(samples)
        duration = len(samples) / self.target_sr

        self._check_duration(duration, path)

        buf = AudioBuffer(
            samples=samples,
            sample_rate=self.target_sr,
            duration_sec=duration,
            source=f"file:{path}",
            metadata={
                "native_sr": native_sr,
                "path": str(path),
                "filename": path.name,
            },
        )

        if self.verbose:
            logger.info("Loaded  %s", buf)

        return buf

    def load_segment(
        self,
        path: str | Path,
        offset_sec: float = 0.0,
        duration_sec: Optional[float] = None,
    ) -> AudioBuffer:
        """Load only a segment (slice) of a file — memory-efficient for large files."""
        path = Path(path).resolve()
        self._validate_path(path)

        samples, native_sr = librosa.load(
            str(path),
            sr=self.target_sr,
            mono=True,
            offset=offset_sec,
            duration=duration_sec,
        )
        samples = self._to_float32(samples)
        duration = len(samples) / self.target_sr

        return AudioBuffer(
            samples=samples,
            sample_rate=self.target_sr,
            duration_sec=duration,
            source=f"file:{path}[{offset_sec:.1f}s]",
            metadata={"native_sr": native_sr, "path": str(path)},
        )

    # ── private ─────────────────────────────────────────────

    def _validate_path(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{ext}'. "
                f"Supported: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

    def _decode(self, path: Path) -> tuple[np.ndarray, int]:
        try:
            # librosa handles almost any format via soundfile + audioread fallback
            samples, sr = librosa.load(str(path), sr=None, mono=False)
            return samples, int(sr)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode {path.name}: {exc}") from exc

    @staticmethod
    def _to_mono(samples: np.ndarray) -> np.ndarray:
        if samples.ndim == 1:
            return samples
        # shape (channels, samples) → average channels
        return samples.mean(axis=0)

    def _resample(self, samples: np.ndarray, native_sr: int) -> np.ndarray:
        if native_sr == self.target_sr:
            return samples
        if self.verbose:
            logger.info("Resampling %d Hz → %d Hz", native_sr, self.target_sr)
        return librosa.resample(samples, orig_sr=native_sr, target_sr=self.target_sr)

    @staticmethod
    def _to_float32(samples: np.ndarray) -> np.ndarray:
        samples = samples.astype(np.float32)
        # hard-clip to [-1, 1] in case of minor float drift
        return np.clip(samples, -1.0, 1.0)

    def _check_duration(self, duration: float, path: Path) -> None:
        if duration < self.min_duration_sec:
            raise ValueError(
                f"{path.name} is {duration:.2f}s — "
                f"minimum required: {self.min_duration_sec}s"
            )


# ─────────────────────────────────────────────────────────────
#  Microphone streamer
# ─────────────────────────────────────────────────────────────

class MicrophoneStreamer:
    """
    Capture audio from the default microphone in real time.

    Yields fixed-length AudioBuffer chunks (window_sec seconds)
    with overlap_ratio overlap.  Runs a background thread to
    fill an internal queue so the caller never blocks on I/O.

    Parameters
    ----------
    target_sr     : desired sample rate
    chunk_size    : PyAudio frames per buffer read
    window_sec    : output chunk duration in seconds
    overlap       : fraction of window to overlap [0, 1)
    device_index  : PyAudio device index, None = system default
    """

    def __init__(
        self,
        target_sr: int = 22050,
        chunk_size: int = 1024,
        window_sec: float = 3.0,
        overlap: float = 0.5,
        device_index: Optional[int] = None,
    ) -> None:
        self.target_sr = target_sr
        self.chunk_size = chunk_size
        self.window_sec = window_sec
        self.overlap = overlap
        self.device_index = device_index

        self._window_samples = int(target_sr * window_sec)
        self._hop_samples = int(self._window_samples * (1 - overlap))
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
        self._ring = np.zeros(self._window_samples, dtype=np.float32)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pa = None
        self._stream = None

    # ── context manager ──────────────────────────────────────

    def __enter__(self) -> "MicrophoneStreamer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── public ──────────────────────────────────────────────

    def start(self) -> None:
        """Open the mic stream and start the capture thread."""
        try:
            import pyaudio  # lazy import — optional dependency
        except ImportError as exc:
            raise ImportError(
                "pyaudio is required for microphone input. "
                "Install it with: pip install pyaudio"
            ) from exc

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.target_sr,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Microphone stream opened (sr=%d, window=%.1fs, overlap=%.0f%%)",
            self.target_sr, self.window_sec, self.overlap * 100,
        )

    def stop(self) -> None:
        """Stop capture and release all resources."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()
        logger.info("Microphone stream closed.")

    def stream(self) -> Generator[AudioBuffer, None, None]:
        """
        Yield AudioBuffer chunks indefinitely until stop() is called.

        Usage
        -----
        with MicrophoneStreamer() as mic:
            for buf in mic.stream():
                process(buf)
        """
        chunk_idx = 0
        while not self._stop_event.is_set():
            try:
                chunk = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            yield AudioBuffer(
                samples=chunk.copy(),
                sample_rate=self.target_sr,
                duration_sec=self.window_sec,
                source="mic",
                metadata={"chunk_idx": chunk_idx},
            )
            chunk_idx += 1

    # ── private ─────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Background thread: read raw frames, maintain a ring buffer,
        and push full windows into the queue."""
        write_pos = 0

        while not self._stop_event.is_set():
            try:
                raw = self._stream.read(self.chunk_size, exception_on_overflow=False)
            except Exception as exc:
                logger.warning("Mic read error: %s", exc)
                time.sleep(0.01)
                continue

            chunk = np.frombuffer(raw, dtype=np.float32)

            # fill ring buffer (wrapping)
            for sample in chunk:
                self._ring[write_pos % self._window_samples] = sample
                write_pos += 1

                if write_pos >= self._window_samples and write_pos % self._hop_samples == 0:
                    window = np.roll(
                        self._ring, -(write_pos % self._window_samples)
                    ).copy()
                    if not self._q.full():
                        self._q.put(window)


# ─────────────────────────────────────────────────────────────
#  Convenience factory
# ─────────────────────────────────────────────────────────────

def load_audio(
    path: str | Path,
    target_sr: int = 22050,
    min_duration_sec: float = 1.0,
    offset_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    verbose: bool = True,
) -> AudioBuffer:
    """
    One-liner helper to load a file.

    Examples
    --------
    buf = load_audio("song.mp3")
    buf = load_audio("song.wav", target_sr=44100)
    buf = load_audio("long.mp3", offset_sec=30.0, duration_sec=10.0)
    """
    loader = AudioFileLoader(
        target_sr=target_sr,
        min_duration_sec=min_duration_sec,
        verbose=verbose,
    )
    if offset_sec > 0 or duration_sec is not None:
        return loader.load_segment(path, offset_sec=offset_sec, duration_sec=duration_sec)
    return loader.load(path)
