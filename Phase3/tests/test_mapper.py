"""tests/test_mapper.py — EmotionToVisualMapper integration tests."""

import math
import pytest
import numpy as np

from mapping.mapper import EmotionToVisualMapper
from mapping.visual_state import VisualState
from mapping.emotion_presets import EMOTION_NAMES

UNIFORM = {e: 0.2 for e in EMOTION_NAMES}


@pytest.fixture
def mapper():
    return EmotionToVisualMapper(
        ema_alpha=0.5,
        confidence_threshold=0.6,
        history_len=4,
        min_dwell_sec=0.0,
        fps=30,
        beat_reactivity=0.7,
    )


# ─────────────────────────────────────────────────────────────
#  map()
# ─────────────────────────────────────────────────────────────

class TestMapDict:
    def test_returns_visual_state(self, mapper):
        assert isinstance(mapper.map({"calm": 1.0}), VisualState)

    def test_frame_count_increments(self, mapper):
        for _ in range(7):
            mapper.map(UNIFORM)
        assert mapper.frame_count == 7

    def test_state_vector_finite(self, mapper):
        for _ in range(20):
            state = mapper.map(UNIFORM)
        assert all(math.isfinite(v) for v in state.to_vector())

    def test_beat_pulse_increases_with_amplitude(self, mapper):
        for _ in range(5):
            mapper.map({"calm": 1.0}, amplitude=0.0)
        low = mapper.map({"calm": 1.0}, amplitude=0.0).beat_pulse
        high = mapper.map({"calm": 1.0}, amplitude=1.0).beat_pulse
        assert high >= low

    def test_zero_amplitude_no_crash(self, mapper):
        assert isinstance(mapper.map({"calm": 1.0}, amplitude=0.0), VisualState)

    def test_full_amplitude_clipped(self, mapper):
        state = mapper.map({"calm": 1.0}, amplitude=2.0)
        assert 0.0 <= state.beat_pulse <= 1.0

    def test_confident_happy_warm_hue(self, mapper):
        probs = {"happy": 0.95, "sad": 0.01, "calm": 0.01,
                 "angry": 0.01, "energetic": 0.02}
        for _ in range(20):
            state = mapper.map(probs)
        assert state.color.hue < 100 or state.color.hue > 300

    def test_confident_sad_cool_hue(self, mapper):
        probs = {"sad": 0.95, "happy": 0.01, "calm": 0.01,
                 "angry": 0.01, "energetic": 0.02}
        for _ in range(20):
            state = mapper.map(probs)
        assert 150 <= state.color.hue <= 280

    def test_all_vector_values_in_range(self, mapper):
        for _ in range(30):
            state = mapper.map(UNIFORM)
        for i, v in enumerate(state.to_vector()):
            assert 0.0 <= v <= 1.0 + 1e-6, f"Index {i} = {v}"


# ─────────────────────────────────────────────────────────────
#  map_array()
# ─────────────────────────────────────────────────────────────

class TestMapArray:
    def test_numpy_array(self, mapper):
        arr = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        assert isinstance(mapper.map_array(arr), VisualState)

    def test_list_input(self, mapper):
        arr = [0.1, 0.6, 0.1, 0.1, 0.1]
        assert isinstance(mapper.map_array(arr), VisualState)

    def test_wrong_length_raises(self, mapper):
        with pytest.raises(ValueError):
            mapper.map_array([0.5, 0.5])

    def test_custom_emotion_names(self, mapper):
        arr = [1.0, 0.0, 0.0, 0.0, 0.0]
        state = mapper.map_array(arr, emotion_names=EMOTION_NAMES)
        assert isinstance(state, VisualState)

    def test_softmax_output_valid(self, mapper):
        raw = np.exp([2.0, 0.5, 0.3, 0.1, 0.1])
        softmax = raw / raw.sum()
        assert isinstance(mapper.map_array(softmax), VisualState)

    def test_with_amplitude(self, mapper):
        arr = [0.2, 0.2, 0.2, 0.2, 0.2]
        state = mapper.map_array(arr, amplitude=0.5)
        assert isinstance(state, VisualState)


# ─────────────────────────────────────────────────────────────
#  map_index()
# ─────────────────────────────────────────────────────────────

class TestMapIndex:
    def test_all_valid_indices(self, mapper):
        for i in range(len(EMOTION_NAMES)):
            assert isinstance(mapper.map_index(i, confidence=0.8), VisualState)

    def test_invalid_index_raises(self, mapper):
        with pytest.raises(ValueError):
            mapper.map_index(99)

    def test_negative_index_raises(self, mapper):
        with pytest.raises(ValueError):
            mapper.map_index(-1)

    def test_dominant_emotion_settles(self, mapper):
        for _ in range(30):
            mapper.map_index(0, confidence=0.99)
        assert mapper.current_emotion == EMOTION_NAMES[0]

    def test_beat_pulse_with_amplitude(self, mapper):
        state = mapper.map_index(0, confidence=0.9, amplitude=0.8)
        assert 0.0 <= state.beat_pulse <= 1.0


# ─────────────────────────────────────────────────────────────
#  reset()
# ─────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_frame_count(self, mapper):
        for _ in range(10):
            mapper.map(UNIFORM)
        mapper.reset()
        assert mapper.frame_count == 0

    def test_reset_sets_emotion(self, mapper):
        for _ in range(10):
            mapper.map({"angry": 1.0})
        mapper.reset("calm")
        assert mapper.current_emotion == "calm"

    def test_map_after_reset_works(self, mapper):
        mapper.reset()
        assert isinstance(mapper.map({"happy": 1.0}), VisualState)

    def test_reset_default_emotion_is_calm(self, mapper):
        mapper.reset()
        assert mapper.current_emotion == "calm"


# ─────────────────────────────────────────────────────────────
#  from_config()
# ─────────────────────────────────────────────────────────────

class TestFromConfig:
    def test_full_config(self):
        cfg = {
            "interpolation": {
                "ema_alpha": 0.2,
                "confidence_threshold": 0.6,
                "history_len": 6,
                "min_dwell_sec": 0.3,
            },
            "visual": {"fps": 30},
        }
        mapper = EmotionToVisualMapper.from_config(cfg)
        assert isinstance(mapper.map({"calm": 1.0}), VisualState)

    def test_empty_config_uses_defaults(self):
        mapper = EmotionToVisualMapper.from_config({})
        assert isinstance(mapper.map({"calm": 1.0}), VisualState)

    def test_partial_config(self):
        cfg = {"interpolation": {"ema_alpha": 0.3}}
        mapper = EmotionToVisualMapper.from_config(cfg)
        assert isinstance(mapper.map(UNIFORM), VisualState)


# ─────────────────────────────────────────────────────────────
#  Properties
# ─────────────────────────────────────────────────────────────

class TestProperties:
    def test_current_state_is_visual_state(self, mapper):
        mapper.map(UNIFORM)
        assert isinstance(mapper.current_state, VisualState)

    def test_current_emotion_is_string(self, mapper):
        mapper.map(UNIFORM)
        assert isinstance(mapper.current_emotion, str)
        assert mapper.current_emotion in EMOTION_NAMES

    def test_frame_count_starts_zero(self):
        m = EmotionToVisualMapper()
        assert m.frame_count == 0

    def test_current_state_updates_each_frame(self, mapper):
        mapper.map({"calm": 1.0})
        s1 = list(mapper.current_state.to_vector())
        mapper.map({"angry": 1.0})
        s2 = list(mapper.current_state.to_vector())
        assert s1 != s2


# ─────────────────────────────────────────────────────────────
#  Edge cases / stress
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_probs_no_crash(self, mapper):
        state = mapper.map({e: 0.0 for e in EMOTION_NAMES})
        assert isinstance(state, VisualState)

    def test_extreme_prob_no_crash(self, mapper):
        state = mapper.map({"angry": 100.0})
        assert isinstance(state, VisualState)

    def test_200_frames_no_nan(self, mapper):
        for i in range(200):
            e = EMOTION_NAMES[i % len(EMOTION_NAMES)]
            probs = {em: 0.05 for em in EMOTION_NAMES}
            probs[e] = 0.8
            state = mapper.map(probs, amplitude=0.3)
        assert all(math.isfinite(v) for v in state.to_vector())

    def test_alternating_emotions_stable(self, mapper):
        for i in range(100):
            e = "happy" if i % 2 == 0 else "sad"
            probs = {em: 0.025 for em in EMOTION_NAMES}
            probs[e] = 0.9
            state = mapper.map(probs)
        assert isinstance(state, VisualState)

    def test_single_emotion_all_frames(self, mapper):
        for _ in range(50):
            state = mapper.map({"energetic": 1.0})
        assert isinstance(state, VisualState)
        assert all(math.isfinite(v) for v in state.to_vector())
