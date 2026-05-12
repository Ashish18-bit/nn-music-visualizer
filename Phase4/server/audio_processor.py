"""
audio_processor.py
──────────────────
Receives raw PCM audio chunks from the browser via WebSocket,
buffers them into 3-second windows, extracts features using
Phase 1 pipeline, runs Phase 2 CNN-LSTM inference, maps to
VisualState via Phase 3 mapper.

This is the core inference pipeline that runs server-side.
"""

from __future__ import annotations

import logging
import queue
import struct
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Phase imports ──────────────────────────────────────────────
try:
    import sys
    import os
    for phase in ["phase1", "phase2", "phase3"]:
        path = os.path.join(os.path.dirname(__file__), "..", "..", phase)
        if path not in sys.path:
            sys.path.insert(0, path)

    from audio.preprocessor import Preprocessor
    from features.extractor import FeatureExtractor
    from mapping.mapper import EmotionToVisualMapper
    from mapping.visual_state import VisualState
    _PHASES_AVAILABLE = True
except ImportError as e:
    logger.warning("Phase imports failed: %s", e)
    _PHASES_AVAILABLE = False

# ── PyTorch model loader ───────────────────────────────────────
try:
    import torch
    from model.cnn_lstm import CNNLSTMEmotionClassifier
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


SAMPLE_RATE = 22050
WINDOW_SEC = 3.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
EMOTIONS = ["happy", "sad", "calm", "angry", "energetic"]


class AudioProcessor:
    """
    Server-side audio analysis pipeline.

    Flow:
      raw PCM bytes → ring buffer → 3s window →
      Phase1 features → Phase2 CNN-LSTM →
      Phase3 mapper → VisualState callback
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        on_state_update: Optional[Callable] = None,
        inference_interval_sec: float = 1.0,
    ) -> None:
        self.on_state_update = on_state_update
        self.inference_interval_sec = inference_interval_sec

        # Ring buffer — holds last 3 seconds of audio at 22050 Hz
        self._buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._samples_received = 0

        # Phase 1 — preprocessor + feature extractor
        if _PHASES_AVAILABLE:
            self._preprocessor = Preprocessor(
                window_sec=WINDOW_SEC,
                overlap=0.0,
                verbose=False,
            )
            self._extractor = FeatureExtractor(verbose=False)
        else:
            self._preprocessor = None
            self._extractor = None

        # Phase 2 — CNN-LSTM model
        self._model = None
        self._device = "cpu"
        if _TORCH_AVAILABLE and checkpoint_path:
            self._load_model(checkpoint_path)

        # Phase 3 — emotion mapper
        self._mapper = EmotionToVisualMapper(
            ema_alpha=0.15,
            confidence_threshold=0.50,
            history_len=8,
            min_dwell_sec=1.5,
            fps=60,
            beat_reactivity=0.8,
        ) if _PHASES_AVAILABLE else None

        # Inference thread
        self._running = False
        self._inference_thread: Optional[threading.Thread] = None

        # Current state
        self._current_probs = {e: 1/5 for e in EMOTIONS}
        self._last_emotion = "calm"
        self._confidence = 0.5
        self._frame = 0

    # ── public ──────────────────────────────────────────────

    def start(self) -> None:
        """Start the background inference thread."""
        self._running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="audio-inference",
        )
        self._inference_thread.start()
        logger.info("AudioProcessor started (model=%s)",
                    "CNN-LSTM" if self._model else "rule-based fallback")

    def stop(self) -> None:
        self._running = False

    def ingest_pcm(self, pcm_bytes: bytes, client_sr: int = 44100) -> None:
        """
        Accept raw PCM float32 bytes from the browser.
        Called from the WebSocket handler on every audio chunk.

        Browser sends Float32Array as binary — each sample is 4 bytes.
        """
        try:
            n_samples = len(pcm_bytes) // 4
            if n_samples == 0:
                return
            samples = np.frombuffer(pcm_bytes, dtype=np.float32).copy()

            # Resample from browser SR to model SR if needed
            if client_sr != SAMPLE_RATE:
                samples = self._resample(samples, client_sr, SAMPLE_RATE)

            # Clip to [-1, 1]
            samples = np.clip(samples, -1.0, 1.0)

            # Write to ring buffer (circular)
            with self._buffer_lock:
                n = len(samples)
                if n >= WINDOW_SAMPLES:
                    self._buffer[:] = samples[-WINDOW_SAMPLES:]
                else:
                    self._buffer = np.roll(self._buffer, -n)
                    self._buffer[-n:] = samples
                self._samples_received += n

        except Exception as e:
            logger.error("PCM ingest error: %s", e)

    def get_current_probs(self) -> dict:
        return dict(self._current_probs)

    # ── inference loop ───────────────────────────────────────

    def _inference_loop(self) -> None:
        """Background thread: run inference every interval seconds."""
        while self._running:
            t0 = time.perf_counter()

            # Need at least 1 window of audio before inferring
            if self._samples_received >= WINDOW_SAMPLES:
                try:
                    self._run_inference()
                except Exception as e:
                    logger.error("Inference error: %s", e, exc_info=True)

            elapsed = time.perf_counter() - t0
            time.sleep(max(0, self.inference_interval_sec - elapsed))

    def _run_inference(self) -> None:
        """Extract features → run model → map to VisualState."""
        with self._buffer_lock:
            window = self._buffer.copy()

        # ── Feature extraction (Phase 1) ──────────────────────
        if self._extractor and _PHASES_AVAILABLE:
            try:
                from audio.loader import AudioBuffer
                from audio.preprocessor import AudioSegment

                segment = AudioSegment(
                    samples=window,
                    sample_rate=SAMPLE_RATE,
                    start_sec=0.0,
                    end_sec=WINDOW_SEC,
                    segment_idx=0,
                    source="browser",
                )
                fv = self._extractor.extract(segment)
                feat_vec = fv.vector   # shape (289,)
            except Exception as e:
                logger.warning("Feature extraction failed: %s", e)
                feat_vec = None
        else:
            feat_vec = None

        # ── Model inference (Phase 2) ─────────────────────────
        if self._model is not None and _TORCH_AVAILABLE:
            probs = self._run_model(window, feat_vec)
        else:
            # Rule-based fallback — uses audio properties
            probs = self._rule_based_emotion(window)

        self._current_probs = probs
        self._last_emotion = max(probs, key=probs.get)
        self._confidence = max(probs.values())

        # ── Phase 3 mapping → VisualState ─────────────────────
        if self._mapper:
            # Compute amplitude for beat reactivity
            amplitude = float(np.sqrt(np.mean(window ** 2)))
            vs = self._mapper.map(probs, amplitude=amplitude)

            # Push to WebSocket server via callback
            if self.on_state_update:
                self.on_state_update(vs, probs)

            self._frame += 1
            logger.debug(
                "Frame %d: %s (%.0f%%) amp=%.2f",
                self._frame, self._last_emotion,
                self._confidence * 100, amplitude,
            )

    def _run_model(self, window: np.ndarray, feat_vec) -> dict:
        """Run CNN-LSTM model and return probability dict."""
        try:
            import torch
            import librosa

            # Mel spectrogram for CNN input
            mel = librosa.feature.melspectrogram(
                y=window, sr=SAMPLE_RATE,
                n_mels=128, n_fft=2048, hop_length=512,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

            # Resize time axis to 130 frames
            from scipy.ndimage import zoom
            if mel_db.shape[1] != 130:
                mel_db = zoom(mel_db, (1.0, 130/mel_db.shape[1]), order=1)

            mel_tensor = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)

            # Feature vector
            if feat_vec is not None:
                feat_tensor = torch.FloatTensor(feat_vec).unsqueeze(0)
            else:
                feat_tensor = torch.zeros(1, 289)

            self._model.eval()
            with torch.no_grad():
                log_probs = self._model(mel_tensor, feat_tensor)
                probs_tensor = log_probs.exp()[0]

            return {
                EMOTIONS[i]: float(probs_tensor[i])
                for i in range(len(EMOTIONS))
            }

        except Exception as e:
            logger.warning("Model inference failed: %s", e)
            return self._rule_based_emotion(window)

    def _rule_based_emotion(self, window: np.ndarray) -> dict:
        """
        Fallback emotion detection using audio features directly.
        No model needed — good enough for a working demo.

        Rules:
          Tempo fast + high energy                → energetic
          Tempo fast + high spectral centroid     → happy
          Low energy + low centroid               → sad
          Low tempo + medium energy               → calm
          High energy + high ZCR + distortion     → angry
        """
        try:
            import librosa

            # Energy
            rms = float(np.sqrt(np.mean(window ** 2)))

            # Tempo
            tempo, _ = librosa.beat.beat_track(y=window, sr=SAMPLE_RATE)
            tempo = float(np.atleast_1d(tempo)[0])

            # Spectral centroid (brightness)
            centroid = float(librosa.feature.spectral_centroid(
                y=window, sr=SAMPLE_RATE
            ).mean())

            # Zero crossing rate (noisiness/distortion)
            zcr = float(librosa.feature.zero_crossing_rate(window).mean())

            # Normalise to [0, 1] ranges
            energy_n    = min(1.0, rms / 0.3)
            tempo_n     = min(1.0, max(0, (tempo - 50) / 150))
            centroid_n  = min(1.0, centroid / 4000)
            zcr_n       = min(1.0, zcr / 0.15)

            # Score each emotion
            scores = {
                "energetic": 0.2*energy_n + 0.4*tempo_n + 0.2*centroid_n + 0.2*zcr_n,
                "happy":     0.2*energy_n + 0.3*tempo_n + 0.4*centroid_n + 0.1*(1-zcr_n),
                "angry":     0.4*energy_n + 0.1*tempo_n + 0.1*centroid_n + 0.4*zcr_n,
                "sad":       0.4*(1-energy_n) + 0.3*(1-tempo_n) + 0.3*(1-centroid_n),
                "calm":      0.3*(1-energy_n) + 0.3*(1-tempo_n) + 0.2*(1-centroid_n) + 0.2*(1-zcr_n),
            }

            # Softmax
            vals = np.array(list(scores.values()))
            vals = np.exp(vals * 3) / np.sum(np.exp(vals * 3))
            return {k: float(v) for k, v in zip(scores.keys(), vals)}

        except Exception as e:
            logger.warning("Rule-based fallback failed: %s", e)
            return {e: 1/5 for e in EMOTIONS}

    # ── helpers ──────────────────────────────────────────────

    def _load_model(self, checkpoint_path: str) -> None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self._model = CNNLSTMEmotionClassifier(
                num_classes=5, n_mels=128, feature_dim=289
            )
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()
            logger.info("Loaded CNN-LSTM from %s", checkpoint_path)
        except Exception as e:
            logger.warning("Could not load model checkpoint: %s", e)
            self._model = None

    @staticmethod
    def _resample(
        samples: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        try:
            import librosa
            return librosa.resample(
                samples, orig_sr=orig_sr, target_sr=target_sr
            )
        except Exception:
            # Simple decimation fallback
            ratio = target_sr / orig_sr
            n_out = int(len(samples) * ratio)
            indices = np.linspace(0, len(samples)-1, n_out).astype(int)
            return samples[indices]
