"""
mapping/interpolator.py
───────────────────────
Smooth interpolation between VisualState instances.

Three components work together:

  EMAInterpolator
  ───────────────
  Exponential Moving Average applied to the numeric vector
  representation of VisualState. Produces perceptually smooth
  transitions: fast changes are dampened, slow changes are tracked.

    state_t = alpha * target + (1 - alpha) * state_(t-1)

  ConfidenceBlender
  ─────────────────
  When model confidence is below a threshold, the top-2 predicted
  emotions are blended proportionally to their softmax scores.
  This prevents jarring snaps when the model is uncertain.

    blended = w1 * preset_1 + w2 * preset_2   (w1 + w2 = 1)

  StateInterpolator   (main class)
  ─────────────────
  Combines both into a single .update() call that accepts raw model
  output (prob dict) and returns a smoothed VisualState each frame.

  Also implements:
  - minimum dwell time  (prevents epileptic emotion flickers)
  - history buffer      (majority-vote for stability)
  - hue interpolation   via shortest arc (avoids 350->10 spinning)
"""

from __future__ import annotations

import dataclasses
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from mapping.visual_state import VisualState
from mapping.emotion_presets import get_preset, EMOTION_NAMES

logger = logging.getLogger(__name__)

VECTOR_LEN = 21   # must match VisualState.to_vector() length


# ─────────────────────────────────────────────────────────────
#  Pure helper functions
# ─────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by factor t in [0,1]."""
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


def _lerp_hue(a: float, b: float, t: float) -> float:
    """
    Interpolate between two hue angles via the shortest arc.
    Avoids spinning through 180 degrees when crossing the 0/360 boundary.
    Example: lerp_hue(350, 10, 0.5) = 0  (not 180)
    """
    diff = (b - a + 180.0) % 360.0 - 180.0
    return (a + diff * float(np.clip(t, 0.0, 1.0))) % 360.0


def _lerp_vector(v1: List[float], v2: List[float], t: float) -> List[float]:
    """
    Element-wise lerp two visual-state vectors.
    Indices 0 and 3 are normalised hues -> use shortest-arc lerp.
    """
    result = []
    for i, (a, b) in enumerate(zip(v1, v2)):
        if i in (0, 3):   # hue fields (normalised 0-1, so x360 then back)
            h = _lerp_hue(a * 360.0, b * 360.0, t) / 360.0
            result.append(h)
        else:
            result.append(_lerp(a, b, t))
    return result


def _blend_states(
    states: List[VisualState],
    weights: List[float],
) -> Tuple[List[float], str, str]:
    """
    Weighted blend of multiple VisualState numeric vectors.
    Returns (blended_vector, dominant_emotion, dominant_shape).
    """
    assert len(states) == len(weights), "states and weights must match"
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()

    vecs = np.array([s.to_vector() for s in states], dtype=np.float64)
    blended = np.zeros(VECTOR_LEN, dtype=np.float64)

    for i in range(VECTOR_LEN):
        if i in (0, 3):   # hue: blend via complex phasor for correctness
            angles = vecs[:, i] * 360.0
            radians = np.deg2rad(angles)
            sin_w = np.sum(w * np.sin(radians))
            cos_w = np.sum(w * np.cos(radians))
            mean_hue = np.rad2deg(np.arctan2(sin_w, cos_w)) % 360.0
            blended[i] = mean_hue / 360.0
        else:
            blended[i] = np.dot(w, vecs[:, i])

    dominant_idx = int(np.argmax(w))
    dominant_emotion = states[dominant_idx].emotion
    dominant_shape = states[dominant_idx].particles.shape

    return blended.tolist(), dominant_emotion, dominant_shape


# ─────────────────────────────────────────────────────────────
#  EMAInterpolator
# ─────────────────────────────────────────────────────────────

class EMAInterpolator:
    """
    Exponential Moving Average smoother for VisualState vectors.

    alpha in (0, 1]:
      High alpha (e.g. 0.8): fast, snappy response
      Low alpha  (e.g. 0.1): slow, dreamy transitions

    Typical value for 60 fps visualiser: 0.10-0.20
    """

    def __init__(self, alpha: float = 0.15) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._current: Optional[List[float]] = None

    def reset(self) -> None:
        """Clear the EMA state (call on hard emotion switches)."""
        self._current = None

    def update(self, target: List[float]) -> List[float]:
        """
        Apply one EMA step toward target vector.
        On the first call, initialises to target (no smoothing).
        """
        if self._current is None:
            self._current = list(target)
            return self._current
        smoothed = _lerp_vector(self._current, target, self.alpha)
        self._current = smoothed
        return smoothed

    @property
    def current(self) -> Optional[List[float]]:
        return self._current


# ─────────────────────────────────────────────────────────────
#  ConfidenceBlender
# ─────────────────────────────────────────────────────────────

class ConfidenceBlender:
    """
    Blends the top-K emotion presets weighted by their softmax probabilities.

    When the model is confident (p_top >= threshold), returns the
    top-1 preset unchanged. When uncertain, blends top-2 presets
    proportionally so visuals reflect the model's ambiguity.

    Parameters
    ----------
    confidence_threshold : minimum top-1 prob to avoid blending
    top_k               : how many top emotions to blend (1 or 2)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.50,
        top_k: int = 2,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.top_k = min(max(top_k, 1), 5)

    def blend(
        self,
        probs: Dict[str, float],
    ) -> Tuple[VisualState, float]:
        """
        Blend emotion presets weighted by probability scores.

        Parameters
        ----------
        probs : dict mapping emotion name -> probability (must sum ~= 1)

        Returns
        -------
        (blended_state, top_confidence)
        """
        sorted_emotions = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_emotion, top_conf = sorted_emotions[0]

        # High confidence -> just return the top preset
        if top_conf >= self.confidence_threshold or self.top_k == 1:
            state = get_preset(top_emotion)
            return dataclasses.replace(
                state, confidence=top_conf, emotion=top_emotion
            ), top_conf

        # Low confidence -> blend top-K
        k = min(self.top_k, len(sorted_emotions))
        top_k_emotions = sorted_emotions[:k]
        states = [get_preset(e) for e, _ in top_k_emotions]
        weights = [p for _, p in top_k_emotions]

        blended_vec, dominant_emotion, dominant_shape = _blend_states(states, weights)
        result = VisualState.from_vector(
            blended_vec,
            emotion=dominant_emotion,
            confidence=top_conf,
            shape=dominant_shape,
            is_transitioning=True,
        )
        return result, top_conf


# ─────────────────────────────────────────────────────────────
#  StateInterpolator
# ─────────────────────────────────────────────────────────────

class StateInterpolator:
    """
    Full pipeline: model probabilities -> smoothed VisualState.

    Call .update(probs) every frame / inference step.
    Returns a VisualState ready for the renderer.

    Parameters
    ----------
    ema_alpha            : EMA smoothing factor
    confidence_threshold : below this, blend top-2 emotions
    history_len          : frames of history for stability voting
    fps                  : target FPS (used to compute dwell frames)
    min_dwell_sec        : minimum dwell seconds before emotion switch
    """

    def __init__(
        self,
        ema_alpha: float = 0.15,
        confidence_threshold: float = 0.50,
        history_len: int = 8,
        fps: int = 60,
        min_dwell_sec: float = 0.5,
    ) -> None:
        self.ema = EMAInterpolator(alpha=ema_alpha)
        self.blender = ConfidenceBlender(
            confidence_threshold=confidence_threshold
        )
        self.history: deque = deque(maxlen=history_len)
        self.min_dwell_frames = max(1, int(min_dwell_sec * fps))
        self.fps = fps

        self._current_emotion: str = "calm"
        self._dwell_counter: int = 0
        self._current_state: VisualState = get_preset("calm")
        self._frame_count: int = 0

    # ── public ──────────────────────────────────────────────

    def update(self, probs: Dict[str, float]) -> VisualState:
        """
        Accept model output probabilities and return the next
        smoothed VisualState.

        Parameters
        ----------
        probs : dict {emotion: probability}, values should sum to 1.0.
                Missing emotions are treated as 0.0 probability.

        Returns
        -------
        Smoothed VisualState ready for the renderer.
        """
        self._frame_count += 1
        probs = self._normalise_probs(probs)

        target_state, confidence = self.blender.blend(probs)

        # Dwell logic
        candidate = target_state.emotion
        self.history.append(candidate)
        stable_emotion = self._stable_emotion()

        if stable_emotion != self._current_emotion:
            if self._dwell_counter >= self.min_dwell_frames:
                self._current_emotion = stable_emotion
                self._dwell_counter = 0
                logger.debug(
                    "Emotion switched -> %s (conf=%.2f)",
                    self._current_emotion, confidence,
                )
            else:
                self._dwell_counter += 1
        else:
            self._dwell_counter = 0

        # EMA smoothing
        target_vec = target_state.to_vector()
        smoothed_vec = self.ema.update(target_vec)

        # Reconstruct VisualState
        is_transitioning = (
            target_state.is_transitioning or self._dwell_counter > 0
        )
        smoothed_state = VisualState.from_vector(
            smoothed_vec,
            emotion=self._current_emotion,
            confidence=confidence,
            shape=target_state.particles.shape,
            is_transitioning=is_transitioning,
        )

        self._current_state = smoothed_state
        return smoothed_state

    def update_from_array(
        self,
        prob_array,
        emotion_names: Optional[List[str]] = None,
    ) -> VisualState:
        """
        Convenience: accept a raw probability array (e.g. model output)
        and convert to dict before calling update().
        """
        if emotion_names is None:
            emotion_names = EMOTION_NAMES
        prob_array = list(prob_array)
        if len(prob_array) != len(emotion_names):
            raise ValueError(
                f"prob_array length {len(prob_array)} != "
                f"len(emotion_names) {len(emotion_names)}"
            )
        probs = {
            name: float(prob_array[i])
            for i, name in enumerate(emotion_names)
        }
        return self.update(probs)

    def reset(self, emotion: str = "calm") -> None:
        """Hard-reset to a specific emotion (e.g. on track change)."""
        self.ema.reset()
        self.history.clear()
        self._current_emotion = emotion
        self._dwell_counter = 0
        self._current_state = get_preset(emotion)
        self._frame_count = 0
        logger.debug("StateInterpolator reset to %s", emotion)

    @property
    def current_state(self) -> VisualState:
        return self._current_state

    @property
    def current_emotion(self) -> str:
        return self._current_emotion

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ── private ─────────────────────────────────────────────

    def _normalise_probs(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Ensure all emotions are present and probs sum to 1."""
        full = {e: probs.get(e, 0.0) for e in EMOTION_NAMES}
        total = sum(full.values())
        if total <= 0:
            return {e: 1.0 / len(EMOTION_NAMES) for e in EMOTION_NAMES}
        return {e: v / total for e, v in full.items()}

    def _stable_emotion(self) -> str:
        """Majority vote over the history buffer."""
        if not self.history:
            return self._current_emotion
        from collections import Counter
        counts = Counter(self.history)
        return counts.most_common(1)[0][0]
