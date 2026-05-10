"""tests/test_visual_state.py — VisualState dataclass tests."""

import json
import math
import pytest

from mapping.visual_state import (
    VisualState, ColorPalette, ParticleConfig, GeometryConfig
)


# ─────────────────────────────────────────────────────────────
#  ColorPalette
# ─────────────────────────────────────────────────────────────

class TestColorPalette:
    def test_default_construction(self):
        c = ColorPalette()
        assert 0.0 <= c.hue <= 360.0
        assert 0.0 <= c.saturation <= 1.0
        assert 0.0 <= c.brightness <= 1.0

    def test_hue_wraps_360(self):
        c = ColorPalette(hue=400.0)
        assert c.hue == pytest.approx(40.0)

    def test_hue_wraps_negative(self):
        c = ColorPalette(hue=-10.0)
        assert 0.0 <= c.hue < 360.0

    def test_saturation_clipped_above(self):
        c = ColorPalette(saturation=1.5)
        assert c.saturation == pytest.approx(1.0)

    def test_saturation_clipped_below(self):
        c = ColorPalette(saturation=-0.5)
        assert c.saturation == pytest.approx(0.0)

    def test_brightness_clipped_above(self):
        c = ColorPalette(brightness=2.0)
        assert c.brightness == pytest.approx(1.0)

    def test_secondary_hue_wraps(self):
        c = ColorPalette(secondary_hue=370.0)
        assert c.secondary_hue == pytest.approx(10.0)

    def test_hue_spread_clipped_above(self):
        c = ColorPalette(hue_spread=100.0)
        assert c.hue_spread == pytest.approx(60.0)

    def test_hue_spread_clipped_below(self):
        c = ColorPalette(hue_spread=-5.0)
        assert c.hue_spread == pytest.approx(0.0)

    def test_css_hsl_format(self):
        c = ColorPalette(hue=120.0, saturation=0.5, brightness=0.6)
        css = c.css_hsl
        assert css.startswith("hsl(")
        assert "120" in css
        assert "50%" in css
        assert "60%" in css


# ─────────────────────────────────────────────────────────────
#  ParticleConfig
# ─────────────────────────────────────────────────────────────

class TestParticleConfig:
    def test_default_shape_valid(self):
        p = ParticleConfig()
        assert p.shape in ("circle", "star", "diamond", "spike", "drop")

    def test_invalid_shape_falls_back_to_circle(self):
        p = ParticleConfig(shape="hexagon")
        assert p.shape == "circle"

    def test_count_clipped_above(self):
        p = ParticleConfig(count=1.5)
        assert p.count == pytest.approx(1.0)

    def test_count_clipped_below(self):
        p = ParticleConfig(count=-0.1)
        assert p.count == pytest.approx(0.0)

    def test_speed_clipped(self):
        assert ParticleConfig(speed=5.0).speed == pytest.approx(1.0)

    def test_opacity_clipped(self):
        assert ParticleConfig(opacity=-1.0).opacity == pytest.approx(0.0)

    def test_all_shapes_accepted(self):
        for shape in ("circle", "star", "diamond", "spike", "drop"):
            p = ParticleConfig(shape=shape)
            assert p.shape == shape

    def test_turbulence_clipped(self):
        assert ParticleConfig(turbulence=2.0).turbulence == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────
#  GeometryConfig
# ─────────────────────────────────────────────────────────────

class TestGeometryConfig:
    def test_ring_count_max(self):
        g = GeometryConfig(ring_count=20)
        assert g.ring_count == 8

    def test_ring_count_min(self):
        g = GeometryConfig(ring_count=0)
        assert g.ring_count == 1

    def test_sides_min(self):
        g = GeometryConfig(sides=1)
        assert g.sides == 3

    def test_sides_max(self):
        g = GeometryConfig(sides=100)
        assert g.sides == 12

    def test_complexity_clipped(self):
        assert GeometryConfig(complexity=2.0).complexity == pytest.approx(1.0)

    def test_rotation_speed_clipped(self):
        assert GeometryConfig(rotation_speed=-1.0).rotation_speed == pytest.approx(0.0)

    def test_bar_height_clipped(self):
        assert GeometryConfig(bar_height=1.5).bar_height == pytest.approx(1.0)

    def test_radial_bars_bool(self):
        g1 = GeometryConfig(radial_bars=True)
        g2 = GeometryConfig(radial_bars=False)
        assert g1.radial_bars is True
        assert g2.radial_bars is False


# ─────────────────────────────────────────────────────────────
#  VisualState
# ─────────────────────────────────────────────────────────────

class TestVisualState:
    def test_default_construction(self):
        vs = VisualState()
        assert isinstance(vs.color, ColorPalette)
        assert isinstance(vs.particles, ParticleConfig)
        assert isinstance(vs.geometry, GeometryConfig)
        assert vs.emotion == "calm"
        assert vs.is_transitioning is False

    def test_confidence_clipped_above(self):
        vs = VisualState(confidence=2.0)
        assert vs.confidence == pytest.approx(1.0)

    def test_confidence_clipped_below(self):
        vs = VisualState(confidence=-1.0)
        assert vs.confidence == pytest.approx(0.0)

    def test_blur_radius_clipped(self):
        assert VisualState(blur_radius=1.5).blur_radius == pytest.approx(1.0)

    def test_bloom_intensity_clipped(self):
        assert VisualState(bloom_intensity=-0.5).bloom_intensity == pytest.approx(0.0)

    # ── to_dict ─────────────────────────────────────────────

    def test_to_dict_returns_dict(self):
        vs = VisualState()
        assert isinstance(vs.to_dict(), dict)

    def test_to_dict_has_color_keys(self):
        d = VisualState().to_dict()
        assert "color_hue" in d
        assert "color_saturation" in d
        assert "color_brightness" in d
        assert "color_secondary_hue" in d
        assert "color_hue_spread" in d

    def test_to_dict_has_particle_keys(self):
        d = VisualState().to_dict()
        assert "particle_count" in d
        assert "particle_speed" in d
        assert "particle_shape" in d
        assert "particle_turbulence" in d

    def test_to_dict_has_geometry_keys(self):
        d = VisualState().to_dict()
        assert "geo_complexity" in d
        assert "geo_ring_count" in d
        assert "geo_sides" in d
        assert "geo_radial_bars" in d

    def test_to_dict_has_scalar_keys(self):
        d = VisualState().to_dict()
        assert "blur_radius" in d
        assert "bloom_intensity" in d
        assert "transition_speed" in d
        assert "background_dim" in d
        assert "beat_pulse" in d

    def test_to_dict_has_metadata_keys(self):
        d = VisualState().to_dict()
        assert "emotion" in d
        assert "confidence" in d
        assert "is_transitioning" in d

    # ── roundtrip ────────────────────────────────────────────

    def test_to_dict_from_dict_roundtrip(self):
        vs = VisualState(
            color=ColorPalette(hue=42.0, saturation=0.8),
            particles=ParticleConfig(count=0.7, shape="star"),
            emotion="happy",
            confidence=0.9,
        )
        vs2 = VisualState.from_dict(vs.to_dict())
        assert vs2.color.hue == pytest.approx(42.0, abs=0.1)
        assert vs2.particles.shape == "star"
        assert vs2.emotion == "happy"
        assert vs2.confidence == pytest.approx(0.9)

    def test_from_dict_missing_keys_use_defaults(self):
        vs = VisualState.from_dict({})
        assert isinstance(vs, VisualState)
        assert vs.emotion == "calm"

    # ── JSON ─────────────────────────────────────────────────

    def test_to_json_is_valid_json(self):
        vs = VisualState()
        j = vs.to_json()
        d = json.loads(j)
        assert "emotion" in d

    def test_to_json_contains_all_keys(self):
        vs = VisualState(emotion="angry")
        d = json.loads(vs.to_json())
        assert d["emotion"] == "angry"

    # ── vector ───────────────────────────────────────────────

    def test_to_vector_length(self):
        vs = VisualState()
        assert len(vs.to_vector()) == 21

    def test_to_vector_all_finite(self):
        vs = VisualState()
        assert all(math.isfinite(v) for v in vs.to_vector())

    def test_to_vector_values_in_range(self):
        vs = VisualState()
        for i, v in enumerate(vs.to_vector()):
            assert 0.0 <= v <= 1.0 + 1e-6, f"Index {i} = {v} out of [0,1]"

    def test_from_vector_roundtrip(self):
        vs = VisualState(
            color=ColorPalette(hue=200.0, saturation=0.7),
            particles=ParticleConfig(count=0.5, speed=0.6),
            geometry=GeometryConfig(complexity=0.4),
            blur_radius=0.3,
            emotion="sad",
        )
        vec = vs.to_vector()
        vs2 = VisualState.from_vector(vec, emotion="sad", shape="drop")
        assert vs2.color.hue == pytest.approx(200.0, abs=1.0)
        assert vs2.particles.count == pytest.approx(0.5, abs=0.01)
        assert vs2.blur_radius == pytest.approx(0.3, abs=0.01)
        assert vs2.emotion == "sad"

    def test_from_vector_shape_preserved(self):
        vs = VisualState()
        vec = vs.to_vector()
        vs2 = VisualState.from_vector(vec, shape="spike")
        assert vs2.particles.shape == "spike"

    # ── repr ─────────────────────────────────────────────────

    def test_repr_contains_emotion(self):
        vs = VisualState(emotion="angry")
        assert "angry" in repr(vs)

    def test_repr_is_string(self):
        vs = VisualState()
        assert isinstance(repr(vs), str)
