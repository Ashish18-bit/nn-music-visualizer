"""
mapping/mapper.py
─────────────────
EmotionToVisualMapper — top-level class that connects Phase 2
model output to Phase 3 visual parameter stream.

Takes either:
  - raw softmax probability array  from the model
  - a single probability dict      {emotion: prob}
  - a predicted class index + confidence

And returns a smoothed VisualState each frame.

Also handles:
  - Audio amplitude modulation (beat_pulse reactivity)
  - Config-driven parameter overrides
  - Frame-rate-aware smoothing
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Dict, List, Optional, Union

import numpy as np

from mapping.visual_state import VisualState
from mapping.emotion_presets import get_preset, EMOTION_NAMES
from mapping.interpolator import StateInterpolator

logger = logging.getLogger(__name__)


class EmotionToVisualMapper:
    """
    Main Phase 3 orchestrator.

    Usage
    ─────
    mapper = EmotionToVisualMapper.from_config(cfg)

    # From a probability dict:
    state = mapper.map({"happy": 0.7, "energetic": 0.2, "calm": 0.1})

    # From a raw softmax array:
    state = mapper.map_array(prob_array)

    # With audio amplitude for beat reactivity:
    state = mapper.map(probs, amplitude=0.8)

    Parameters
    ----------
    ema_alpha            : EMA smoothing (0=frozen, 1=instant)
    confidence_threshold : below this, blend top-2 emotions
    history_len          : stability voting window (frames)
    min_dwell_sec        : minimum seconds per emotion
    fps                  : renderer target FPS
    beat_reactivity      : how much audio amplitude drives beat_pulse
    """

    def __init__(
        self,
        ema_alpha: float = 0.15,
        confidence_threshold: float = 0.50,
        history_len: int = 8,
        min_dwell_sec: float = 0.5,
        fps: int = 60,
        beat_reactivity: float = 0.7,
    ) -> None:
        self.interpolator = StateInterpolator(
            ema_alpha=ema_alpha,
            confidence_threshold=confidence_threshold,
            history_len=history_len,
            fps=fps,
            min_dwell_sec=min_dwell_sec,
        )
        self.beat_reactivity = float(np.clip(beat_reactivity, 0.0, 1.0))
        self._frame_count = 0

    # ── public ──────────────────────────────────────────────

    def map(
        self,
        probs: Dict[str, float],
        amplitude: float = 0.0,
    ) -> VisualState:
        """
        Map a probability dict to a smoothed VisualState.

        Parameters
        ----------
        probs     : {emotion_name: probability}
        amplitude : audio RMS amplitude [0, 1] for beat reactivity

        Returns
        -------
        VisualState ready for the renderer
        """
        self._frame_count += 1
        state = self.interpolator.update(probs)
        state = self._apply_amplitude(state, amplitude)
        return state

    def map_array(
        self,
        prob_array: Union[List[float], np.ndarray],
        amplitude: float = 0.0,
        emotion_names: Optional[List[str]] = None,
    ) -> VisualState:
        """
        Map a raw softmax probability array to a VisualState.

        Parameters
        ----------
        prob_array    : array of shape (num_classes,)
        amplitude     : audio RMS [0, 1]
        emotion_names : class name order (defaults to EMOTION_NAMES)
        """
        names = emotion_names or EMOTION_NAMES
        prob_array = np.asarray(prob_array, dtype=np.float64)
        if prob_array.shape[0] != len(names):
            raise ValueError(
                f"prob_array length {prob_array.shape[0]} != "
                f"len(emotion_names) {len(names)}"
            )
        probs = {name: float(prob_array[i]) for i, name in enumerate(names)}
        return self.map(probs, amplitude=amplitude)

    def map_index(
        self,
        class_idx: int,
        confidence: float = 1.0,
        amplitude: float = 0.0,
    ) -> VisualState:
        """
        Map a predicted class index + confidence to a VisualState.
        Constructs a near-one-hot probability dict and calls map().
        """
        if not 0 <= class_idx < len(EMOTION_NAMES):
            raise ValueError(
                f"class_idx {class_idx} out of range "
                f"[0, {len(EMOTION_NAMES) - 1}]"
            )
        probs = {e: 0.0 for e in EMOTION_NAMES}
        probs[EMOTION_NAMES[class_idx]] = float(confidence)
        remainder = (1.0 - confidence) / (len(EMOTION_NAMES) - 1)
        for e in EMOTION_NAMES:
            if e != EMOTION_NAMES[class_idx]:
                probs[e] = remainder
        return self.map(probs, amplitude=amplitude)

    def reset(self, emotion: str = "calm") -> None:
        """Reset the interpolator state (call on new track / silence)."""
        self.interpolator.reset(emotion)
        self._frame_count = 0
        logger.debug("Mapper reset to %s", emotion)

    @property
    def current_state(self) -> VisualState:
        return self.interpolator.current_state

    @property
    def current_emotion(self) -> str:
        return self.interpolator.current_emotion

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ── private ─────────────────────────────────────────────

    def _apply_amplitude(
        self, state: VisualState, amplitude: float
    ) -> VisualState:
        """
        Modulate beat_pulse with the current audio amplitude.
        beat_pulse = preset_value * (1 - reactivity)
                   + amplitude    * reactivity
        """
        if amplitude <= 0.0 or self.beat_reactivity <= 0.0:
            return state
        amp = float(np.clip(amplitude, 0.0, 1.0))
        new_pulse = (
            state.beat_pulse * (1.0 - self.beat_reactivity)
            + amp * self.beat_reactivity
        )
        return dataclasses.replace(
            state, beat_pulse=float(np.clip(new_pulse, 0.0, 1.0))
        )

    # ── factory ─────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "EmotionToVisualMapper":
        """Build mapper from config.yaml dict."""
        interp = cfg.get("interpolation", {})
        vis = cfg.get("visual", {})
        return cls(
            ema_alpha=interp.get("ema_alpha", 0.15),
            confidence_threshold=interp.get("confidence_threshold", 0.50),
            history_len=interp.get("history_len", 8),
            min_dwell_sec=interp.get("min_dwell_sec", 0.5),
            fps=vis.get("fps", 60),
            beat_reactivity=0.7,
        )
