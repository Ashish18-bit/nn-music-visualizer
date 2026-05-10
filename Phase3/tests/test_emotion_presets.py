"""tests/test_emotion_presets.py — emotion preset validation tests."""

import math
import pytest

from mapping.emotion_presets import (
    EMOTION_PRESETS, EMOTION_NAMES, get_preset, all_presets, EmotionPreset
)
from mapping.visual_state import VisualState

EMOTIONS = ["happy", "sad", "calm", "angry", "energetic"]


# ─────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────

class TestPresetRegistry:
    def test_all_emotions_present(self):
        for e in EMOTIONS:
            assert e in EMOTION_PRESETS

    def test_no_extra_emotions(self):
        assert set(EMOTION_PRESETS.keys()) == set(EMOTIONS)

    def test_each_value_is_emotion_preset(self):
        for name, preset in EMOTION_PRESETS.items():
            assert isinstance(preset, EmotionPreset)

    def test_preset_name_matches_key(self):
        for name, preset in EMOTION_PRESETS.items():
            assert preset.name == name

    def test_preset_state_is_visual_state(self):
        for preset in EMOTION_PRESETS.values():
            assert isinstance(preset.state, VisualState)

    def test_emotion_names_list_complete(self):
        for e in EMOTIONS:
            assert e in EMOTION_NAMES


# ─────────────────────────────────────────────────────────────
#  get_preset()
# ─────────────────────────────────────────────────────────────

class TestGetPreset:
    def test_returns_visual_state(self):
        for e in EMOTIONS:
            state = get_preset(e)
            assert isinstance(state, VisualState)

    def test_unknown_emotion_raises_key_error(self):
        with pytest.raises(KeyError):
            get_preset("confused")

    def test_emotion_field_matches(self):
        for e in EMOTIONS:
            state = get_preset(e)
            assert state.emotion == e

    def test_confidence_is_one(self):
        for e in EMOTIONS:
            state = get_preset(e)
            assert state.confidence == pytest.approx(1.0)

    def test_is_not_transitioning(self):
        for e in EMOTIONS:
            state = get_preset(e)
            assert state.is_transitioning is False


# ─────────────────────────────────────────────────────────────
#  Preset distinctness
# ─────────────────────────────────────────────────────────────

class TestPresetDistinctness:
    def test_happy_faster_than_sad(self):
        assert get_preset("happy").particles.speed > get_preset("sad").particles.speed

    def test_angry_more_particles_than_sad(self):
        assert get_preset("angry").particles.count > get_preset("sad").particles.count

    def test_angry_less_blur_than_calm(self):
        assert get_preset("angry").blur_radius < get_preset("calm").blur_radius

    def test_energetic_high_beat_pulse(self):
        assert get_preset("energetic").beat_pulse > get_preset("sad").beat_pulse

    def test_happy_warm_hue(self):
        hue = get_preset("happy").color.hue
        assert hue < 80 or hue > 300

    def test_sad_cool_hue(self):
        hue = get_preset("sad").color.hue
        assert 170 <= hue <= 280

    def test_angry_spike_shape(self):
        assert get_preset("angry").particles.shape == "spike"

    def test_happy_star_shape(self):
        assert get_preset("happy").particles.shape == "star"

    def test_sad_drop_shape(self):
        assert get_preset("sad").particles.shape == "drop"

    def test_energetic_diamond_shape(self):
        assert get_preset("energetic").particles.shape == "diamond"

    def test_calm_circle_shape(self):
        assert get_preset("calm").particles.shape == "circle"

    def test_angry_most_geometry(self):
        angry = get_preset("angry")
        sad = get_preset("sad")
        assert angry.geometry.complexity > sad.geometry.complexity

    def test_all_shapes_valid(self):
        valid = {"circle", "star", "diamond", "spike", "drop"}
        for e in EMOTIONS:
            assert get_preset(e).particles.shape in valid

    def test_all_presets_vectors_finite(self):
        for e in EMOTIONS:
            vec = get_preset(e).to_vector()
            assert all(math.isfinite(v) for v in vec), \
                f"{e} has non-finite vector values"

    def test_all_presets_vectors_in_range(self):
        for e in EMOTIONS:
            vec = get_preset(e).to_vector()
            for i, v in enumerate(vec):
                assert 0.0 <= v <= 1.0 + 1e-6, \
                    f"{e} index {i} = {v} out of [0,1]"

    def test_emotions_are_mutually_distinct(self):
        """No two presets should have identical vectors."""
        vecs = {e: tuple(get_preset(e).to_vector()) for e in EMOTIONS}
        seen = set()
        for e, vec in vecs.items():
            assert vec not in seen, f"{e} has duplicate vector"
            seen.add(vec)


# ─────────────────────────────────────────────────────────────
#  all_presets()
# ─────────────────────────────────────────────────────────────

class TestAllPresets:
    def test_returns_dict(self):
        assert isinstance(all_presets(), dict)

    def test_all_emotions_in_result(self):
        presets = all_presets()
        for e in EMOTIONS:
            assert e in presets

    def test_values_are_visual_states(self):
        for v in all_presets().values():
            assert isinstance(v, VisualState)

    def test_count_matches_emotion_count(self):
        assert len(all_presets()) == len(EMOTIONS)
