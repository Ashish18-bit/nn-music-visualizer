"""tests/test_interpolator.py — EMAInterpolator, ConfidenceBlender, StateInterpolator."""

import math
import pytest

from mapping.interpolator import (
    EMAInterpolator, ConfidenceBlender, StateInterpolator,
    _lerp, _lerp_hue, _lerp_vector, _blend_states, VECTOR_LEN,
)
from mapping.emotion_presets import get_preset, EMOTION_NAMES
from mapping.visual_state import VisualState

UNIFORM_PROBS = {e: 1.0 / 5 for e in EMOTION_NAMES}


# ─────────────────────────────────────────────────────────────
#  Pure helper functions
# ─────────────────────────────────────────────────────────────

class TestLerpHelpers:
    def test_lerp_midpoint(self):
        assert _lerp(0.0, 1.0, 0.5) == pytest.approx(0.5)

    def test_lerp_at_zero(self):
        assert _lerp(3.0, 7.0, 0.0) == pytest.approx(3.0)

    def test_lerp_at_one(self):
        assert _lerp(3.0, 7.0, 1.0) == pytest.approx(7.0)

    def test_lerp_clamps_below_zero(self):
        assert _lerp(0.0, 1.0, -0.5) == pytest.approx(0.0)

    def test_lerp_clamps_above_one(self):
        assert _lerp(0.0, 1.0, 1.5) == pytest.approx(1.0)

    def test_lerp_hue_shortest_arc(self):
        # 350 deg -> 10 deg: shortest path is +20 deg, not -340 deg
        result = _lerp_hue(350.0, 10.0, 0.5)
        assert result == pytest.approx(0.0, abs=1.0)

    def test_lerp_hue_normal_case(self):
        # 0->180 is ambiguous (both arcs are 180 deg); either midpoint is valid
        result = _lerp_hue(0.0, 180.0, 0.5)
        assert result == pytest.approx(90.0, abs=1.0) or result == pytest.approx(270.0, abs=1.0)

    def test_lerp_hue_output_in_range(self):
        for a in range(0, 360, 30):
            for b in range(0, 360, 30):
                result = _lerp_hue(float(a), float(b), 0.5)
                assert 0.0 <= result < 360.0

    def test_lerp_hue_at_zero(self):
        assert _lerp_hue(100.0, 200.0, 0.0) == pytest.approx(100.0, abs=1.0)

    def test_lerp_hue_at_one(self):
        assert _lerp_hue(100.0, 200.0, 1.0) == pytest.approx(200.0, abs=1.0)

    def test_lerp_vector_length_preserved(self):
        v1 = [0.0] * VECTOR_LEN
        v2 = [1.0] * VECTOR_LEN
        assert len(_lerp_vector(v1, v2, 0.5)) == VECTOR_LEN

    def test_lerp_vector_midpoint_non_hue(self):
        v1 = [0.0] * VECTOR_LEN
        v2 = [1.0] * VECTOR_LEN
        result = _lerp_vector(v1, v2, 0.5)
        for i, v in enumerate(result):
            if i not in (0, 3):
                assert v == pytest.approx(0.5, abs=0.01)

    def test_lerp_vector_at_zero(self):
        v1 = [0.3] * VECTOR_LEN
        v2 = [0.7] * VECTOR_LEN
        result = _lerp_vector(v1, v2, 0.0)
        for i, v in enumerate(result):
            if i not in (0, 3):
                assert v == pytest.approx(0.3, abs=0.01)


# ─────────────────────────────────────────────────────────────
#  EMAInterpolator
# ─────────────────────────────────────────────────────────────

class TestEMAInterpolator:
    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError):
            EMAInterpolator(alpha=0.0)

    def test_alpha_above_one_raises(self):
        with pytest.raises(ValueError):
            EMAInterpolator(alpha=1.5)

    def test_alpha_one_valid(self):
        ema = EMAInterpolator(alpha=1.0)
        result = ema.update([0.5] * VECTOR_LEN)
        assert len(result) == VECTOR_LEN

    def test_first_call_returns_target_exactly(self):
        ema = EMAInterpolator(alpha=0.5)
        target = [0.5] * VECTOR_LEN
        result = ema.update(target)
        assert result == pytest.approx(target)

    def test_smoothing_moves_toward_target(self):
        ema = EMAInterpolator(alpha=0.5)
        ema.update([0.0] * VECTOR_LEN)
        result = ema.update([1.0] * VECTOR_LEN)
        # non-hue index: 0.0 * (1-0.5) + 1.0 * 0.5 = 0.5
        assert result[5] == pytest.approx(0.5, abs=0.05)

    def test_repeated_updates_converge(self):
        ema = EMAInterpolator(alpha=0.3)
        ema.update([0.0] * VECTOR_LEN)
        for _ in range(60):
            result = ema.update([1.0] * VECTOR_LEN)
        assert result[5] == pytest.approx(1.0, abs=0.01)

    def test_reset_clears_state(self):
        ema = EMAInterpolator(alpha=0.5)
        ema.update([1.0] * VECTOR_LEN)
        ema.reset()
        assert ema.current is None

    def test_after_reset_first_call_is_target(self):
        ema = EMAInterpolator(alpha=0.5)
        ema.update([0.0] * VECTOR_LEN)
        ema.reset()
        target = [0.7] * VECTOR_LEN
        result = ema.update(target)
        assert result == pytest.approx(target)

    def test_output_length_preserved(self):
        ema = EMAInterpolator(alpha=0.2)
        result = ema.update([0.5] * VECTOR_LEN)
        assert len(result) == VECTOR_LEN

    def test_current_property(self):
        ema = EMAInterpolator(alpha=0.5)
        assert ema.current is None
        ema.update([0.5] * VECTOR_LEN)
        assert ema.current is not None
        assert len(ema.current) == VECTOR_LEN


# ─────────────────────────────────────────────────────────────
#  ConfidenceBlender
# ─────────────────────────────────────────────────────────────

class TestConfidenceBlender:
    def test_high_confidence_returns_top_emotion(self):
        blender = ConfidenceBlender(confidence_threshold=0.5)
        probs = {"happy": 0.8, "sad": 0.1, "calm": 0.05,
                 "angry": 0.03, "energetic": 0.02}
        state, conf = blender.blend(probs)
        assert state.emotion == "happy"
        assert conf == pytest.approx(0.8)

    def test_low_confidence_sets_transitioning(self):
        blender = ConfidenceBlender(confidence_threshold=0.9, top_k=2)
        probs = {"happy": 0.45, "energetic": 0.40, "sad": 0.05,
                 "calm": 0.05, "angry": 0.05}
        state, conf = blender.blend(probs)
        assert state.is_transitioning is True

    def test_returns_visual_state(self):
        blender = ConfidenceBlender()
        state, _ = blender.blend(UNIFORM_PROBS)
        assert isinstance(state, VisualState)

    def test_confidence_returned_correctly(self):
        blender = ConfidenceBlender(confidence_threshold=0.5)
        probs = {"happy": 0.7, "sad": 0.1, "calm": 0.1,
                 "angry": 0.05, "energetic": 0.05}
        _, conf = blender.blend(probs)
        assert conf == pytest.approx(0.7)

    def test_single_emotion_unity(self):
        blender = ConfidenceBlender(confidence_threshold=0.5)
        probs = {"calm": 1.0, "happy": 0.0, "sad": 0.0,
                 "angry": 0.0, "energetic": 0.0}
        state, conf = blender.blend(probs)
        assert state.emotion == "calm"
        assert conf == pytest.approx(1.0)

    def test_top_k_one_never_blends(self):
        blender = ConfidenceBlender(confidence_threshold=0.99, top_k=1)
        probs = {"happy": 0.4, "energetic": 0.35, "sad": 0.1,
                 "calm": 0.1, "angry": 0.05}
        state, _ = blender.blend(probs)
        # With top_k=1, should never be transitioning
        assert state.is_transitioning is False

    def test_blended_state_vector_finite(self):
        blender = ConfidenceBlender(confidence_threshold=0.99, top_k=2)
        probs = {"happy": 0.5, "energetic": 0.5, "sad": 0.0,
                 "calm": 0.0, "angry": 0.0}
        state, _ = blender.blend(probs)
        assert all(math.isfinite(v) for v in state.to_vector())


# ─────────────────────────────────────────────────────────────
#  StateInterpolator
# ─────────────────────────────────────────────────────────────

class TestStateInterpolator:
    def test_update_returns_visual_state(self):
        interp = StateInterpolator(fps=10, min_dwell_sec=0.0)
        state = interp.update({"calm": 1.0})
        assert isinstance(state, VisualState)

    def test_frame_count_increments(self):
        interp = StateInterpolator(fps=10, min_dwell_sec=0.0)
        for _ in range(5):
            interp.update(UNIFORM_PROBS)
        assert interp.frame_count == 5

    def test_dominant_emotion_settles(self):
        interp = StateInterpolator(fps=10, min_dwell_sec=0.0, history_len=3)
        probs = {"happy": 0.9, "sad": 0.025, "calm": 0.025,
                 "angry": 0.025, "energetic": 0.025}
        for _ in range(10):
            interp.update(probs)
        assert interp.current_emotion == "happy"

    def test_state_vector_finite(self):
        interp = StateInterpolator(fps=10, min_dwell_sec=0.0)
        for _ in range(20):
            state = interp.update(UNIFORM_PROBS)
        assert all(math.isfinite(v) for v in state.to_vector())

    def test_reset_changes_emotion(self):
        interp = StateInterpolator(fps=10, min_dwell_sec=0.0)
        for _ in range(10):
            interp.update({"angry": 0.9, "happy": 0.025,
                           "sad": 0.025, "calm": 0.025, "energetic": 0.025})
        interp.reset("calm")
        assert interp.current_emotion == "calm"

    def test_reset_clears_frame_count(self):
        interp = StateInterpolator(fps=10, min_dwell_sec=0.0)
        for _ in range(10):
            interp.update(UNIFORM_PROBS)
        interp.reset()
        assert interp.frame_count == 0

    def test_update_from_array_valid(self):
        interp = StateInterpolator(fps=10, min_dwell_sec=0.0)
        arr = [0.1, 0.1, 0.6, 0.1, 0.1]
        state = interp.update_from_array(arr)
        assert isinstance(state, VisualState)

    def test_update_from_array_wrong_length_raises(self):
        interp = StateInterpolator()
        with pytest.raises(ValueError):
            interp.update_from_array([0.5, 0.5])

    def test_dwell_prevents_instant_switch(self):
        interp = StateInterpolator(fps=10, min_dwell_sec=1.0, history_len=3)
        for _ in range(5):
            interp.update({"calm": 0.9, "happy": 0.025, "sad": 0.025,
                           "angry": 0.025, "energetic": 0.025})
        assert interp.current_emotion == "calm"
        # One frame of angry should not immediately switch
        interp.update({"angry": 0.9, "calm": 0.025, "happy": 0.025,
                       "sad": 0.025, "energetic": 0.025})
        assert interp.current_emotion == "calm"

    def test_missing_emotions_normalised(self):
        interp = StateInterpolator(fps=10, min_dwell_sec=0.0)
        state = interp.update({"happy": 1.0})
        assert isinstance(state, VisualState)

    def test_zero_probs_no_crash(self):
        interp = StateInterpolator(fps=10, min_dwell_sec=0.0)
        state = interp.update({"happy": 0.0, "sad": 0.0, "calm": 0.0,
                               "angry": 0.0, "energetic": 0.0})
        assert isinstance(state, VisualState)

    def test_current_state_property(self):
        interp = StateInterpolator(fps=10, min_dwell_sec=0.0)
        interp.update(UNIFORM_PROBS)
        assert isinstance(interp.current_state, VisualState)

    def test_uniform_probs_stable(self):
        interp = StateInterpolator(fps=10, min_dwell_sec=0.0)
        for _ in range(30):
            state = interp.update(UNIFORM_PROBS)
        assert isinstance(state, VisualState)

    def test_vector_values_in_range_after_many_frames(self):
        interp = StateInterpolator(fps=10, min_dwell_sec=0.0)
        for i in range(100):
            e = EMOTION_NAMES[i % len(EMOTION_NAMES)]
            probs = {em: 0.05 for em in EMOTION_NAMES}
            probs[e] = 0.8
            state = interp.update(probs)
        for i, v in enumerate(state.to_vector()):
            assert 0.0 <= v <= 1.0 + 1e-6, f"Index {i} = {v} out of range"
