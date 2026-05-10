"""tests/test_visual_types.py — RendererState, Particle, colour helpers."""

import math
import pytest
from renderer.visual_types import (
    hsl_to_rgb, lerp_color, with_alpha,
    RendererState, Particle,
)

# ── Minimal VisualState stub ─────────────────────────────────

class _Color:
    def __init__(self, hue=265, sat=0.55, bri=0.6, sec_hue=280, spread=20):
        self.hue = hue; self.saturation = sat; self.brightness = bri
        self.secondary_hue = sec_hue; self.hue_spread = spread

class _Particles:
    count=0.3; speed=0.35; size=0.4; opacity=0.75
    lifetime=0.5; shape="circle"; turbulence=0.2

class _Geometry:
    complexity=0.4; ring_count=3; rotation_speed=0.25
    sides=6; radial_bars=True; bar_height=0.5

class _VS:
    color = _Color(); particles = _Particles(); geometry = _Geometry()
    blur_radius=0.6; bloom_intensity=0.3; transition_speed=0.5
    background_dim=0.85; beat_pulse=0.4; emotion="calm"; confidence=1.0
    is_transitioning=False

MINIMAL_CFG = {"visual_params": {
    "particle_count_range": [20, 300],
    "particle_speed_range": [0.5, 6.0],
    "particle_size_range": [3, 18],
    "geo_rotation_speed_range": [0.002, 0.04],
}}


# ─────────────────────────────────────────────────────────────
#  hsl_to_rgb
# ─────────────────────────────────────────────────────────────

class TestHslToRgb:
    def test_red(self):
        r, g, b = hsl_to_rgb(0, 1.0, 0.5)
        assert r == 255 and g == 0 and b == 0

    def test_green(self):
        r, g, b = hsl_to_rgb(120, 1.0, 0.5)
        assert r == 0 and g == 255 and b == 0

    def test_blue(self):
        r, g, b = hsl_to_rgb(240, 1.0, 0.5)
        assert r == 0 and g == 0 and b == 255

    def test_white(self):
        r, g, b = hsl_to_rgb(0, 0.0, 1.0)
        assert r == 255 and g == 255 and b == 255

    def test_black(self):
        r, g, b = hsl_to_rgb(0, 0.0, 0.0)
        assert r == 0 and g == 0 and b == 0

    def test_output_in_0_255(self):
        for h in range(0, 360, 30):
            r, g, b = hsl_to_rgb(h, 0.8, 0.5)
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255

    def test_hue_wraps(self):
        c1 = hsl_to_rgb(0, 1.0, 0.5)
        c2 = hsl_to_rgb(360, 1.0, 0.5)
        assert c1 == c2

    def test_returns_tuple_of_three(self):
        result = hsl_to_rgb(180, 0.5, 0.5)
        assert len(result) == 3


# ─────────────────────────────────────────────────────────────
#  lerp_color
# ─────────────────────────────────────────────────────────────

class TestLerpColor:
    def test_t_zero_returns_c1(self):
        assert lerp_color((0, 0, 0), (255, 255, 255), 0.0) == (0, 0, 0)

    def test_t_one_returns_c2(self):
        assert lerp_color((0, 0, 0), (255, 255, 255), 1.0) == (255, 255, 255)

    def test_midpoint(self):
        r, g, b = lerp_color((0, 0, 0), (200, 100, 50), 0.5)
        assert r == pytest.approx(100, abs=1)
        assert g == pytest.approx(50, abs=1)

    def test_clamped_below(self):
        result = lerp_color((100, 100, 100), (200, 200, 200), -1.0)
        assert result == (100, 100, 100)

    def test_clamped_above(self):
        result = lerp_color((100, 100, 100), (200, 200, 200), 2.0)
        assert result == (200, 200, 200)


# ─────────────────────────────────────────────────────────────
#  with_alpha
# ─────────────────────────────────────────────────────────────

class TestWithAlpha:
    def test_appends_alpha(self):
        result = with_alpha((100, 150, 200), 128)
        assert result == (100, 150, 200, 128)

    def test_alpha_clipped_above(self):
        assert with_alpha((0, 0, 0), 300)[3] == 255

    def test_alpha_clipped_below(self):
        assert with_alpha((0, 0, 0), -10)[3] == 0


# ─────────────────────────────────────────────────────────────
#  RendererState
# ─────────────────────────────────────────────────────────────

class TestRendererState:
    def test_default_construction(self):
        rs = RendererState()
        assert rs.emotion == "calm"
        assert isinstance(rs.primary_color, tuple)
        assert len(rs.primary_color) == 3

    def test_from_visual_state(self):
        vs = _VS()
        rs = RendererState.from_visual_state(vs, MINIMAL_CFG, 1280, 720)
        assert isinstance(rs, RendererState)
        assert rs.emotion == "calm"

    def test_particle_count_in_range(self):
        vs = _VS()
        rs = RendererState.from_visual_state(vs, MINIMAL_CFG)
        assert 20 <= rs.particle_count <= 300

    def test_particle_speed_positive(self):
        rs = RendererState.from_visual_state(_VS(), MINIMAL_CFG)
        assert rs.particle_speed > 0

    def test_particle_size_positive(self):
        rs = RendererState.from_visual_state(_VS(), MINIMAL_CFG)
        assert rs.particle_size > 0

    def test_primary_color_valid_rgb(self):
        rs = RendererState.from_visual_state(_VS(), MINIMAL_CFG)
        for channel in rs.primary_color:
            assert 0 <= channel <= 255

    def test_geo_ring_count_preserved(self):
        vs = _VS()
        vs.geometry = _Geometry()
        vs.geometry.ring_count = 5
        rs = RendererState.from_visual_state(vs, MINIMAL_CFG)
        assert rs.geo_ring_count == 5

    def test_blur_radius_preserved(self):
        vs = _VS()
        vs.blur_radius = 0.9
        rs = RendererState.from_visual_state(vs, MINIMAL_CFG)
        assert rs.blur_radius == pytest.approx(0.9)

    def test_emotion_preserved(self):
        vs = _VS()
        vs.emotion = "angry"
        rs = RendererState.from_visual_state(vs, MINIMAL_CFG)
        assert rs.emotion == "angry"

    def test_empty_cfg_no_crash(self):
        rs = RendererState.from_visual_state(_VS(), {})
        assert isinstance(rs, RendererState)


# ─────────────────────────────────────────────────────────────
#  Particle
# ─────────────────────────────────────────────────────────────

class TestParticle:
    def _rs(self, shape="circle"):
        rs = RendererState()
        rs.particle_shape = shape
        rs.particle_speed = 2.0
        rs.particle_size = 8.0
        rs.particle_opacity = 200
        rs.particle_lifetime = 60
        rs.particle_turbulence = 0.1
        return rs

    def test_spawn_returns_particle(self):
        rs = self._rs()
        p = Particle.spawn(1280, 720, rs)
        assert isinstance(p, Particle)

    def test_particle_alive_at_birth(self):
        p = Particle.spawn(1280, 720, self._rs())
        assert p.alive is True

    def test_particle_dies_after_max_life(self):
        p = Particle.spawn(1280, 720, self._rs())
        p.life = p.max_life
        assert p.alive is False

    def test_life_fraction_at_birth(self):
        p = Particle.spawn(1280, 720, self._rs())
        assert p.life_fraction == pytest.approx(0.0, abs=0.01)

    def test_life_fraction_at_death(self):
        p = Particle.spawn(1280, 720, self._rs())
        p.life = p.max_life
        assert p.life_fraction == pytest.approx(1.0, abs=0.01)

    def test_alpha_at_birth_equals_opacity(self):
        p = Particle.spawn(1280, 720, self._rs())
        assert p.alpha == p.opacity

    def test_alpha_at_death_is_zero(self):
        p = Particle.spawn(1280, 720, self._rs())
        p.life = p.max_life
        assert p.alpha == 0

    def test_update_moves_particle(self):
        p = Particle.spawn(1280, 720, self._rs())
        x0, y0 = p.x, p.y
        p.update(turbulence=0.0)
        assert p.x != x0 or p.y != y0

    def test_update_increments_life(self):
        p = Particle.spawn(1280, 720, self._rs())
        p.update()
        assert p.life == 1

    def test_all_shapes_spawn(self):
        for shape in ("circle", "star", "diamond", "spike", "drop"):
            rs = self._rs(shape)
            p = Particle.spawn(800, 600, rs)
            assert p.shape == shape

    def test_no_nan_in_position(self):
        rs = self._rs()
        p = Particle.spawn(1280, 720, rs)
        for _ in range(30):
            p.update(turbulence=0.5)
        assert math.isfinite(p.x) and math.isfinite(p.y)
