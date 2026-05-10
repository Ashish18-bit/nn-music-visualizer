"""tests/test_background.py — BackgroundRenderer tests."""

import pytest
from renderer.visual_types import RendererState
from renderer.background import BackgroundRenderer


def _rs(**kwargs) -> RendererState:
    rs = RendererState()
    rs.blur_radius     = kwargs.get("blur_radius", 0.6)
    rs.bloom_intensity = kwargs.get("bloom_intensity", 0.3)
    rs.background_dim  = kwargs.get("background_dim", 0.85)
    rs.beat_pulse      = kwargs.get("beat_pulse", 0.3)
    rs.primary_color   = kwargs.get("primary_color", (167, 139, 250))
    return rs


class TestBackgroundRenderer:
    def test_default_construction(self):
        bg = BackgroundRenderer()
        assert bg.canvas_w == 1280
        assert bg.canvas_h == 720

    def test_custom_dimensions(self):
        bg = BackgroundRenderer(800, 600)
        assert bg.canvas_w == 800
        assert bg.canvas_h == 600

    def test_custom_base_color(self):
        bg = BackgroundRenderer(base_color=(0, 0, 20))
        assert bg.base_color == (0, 0, 20)

    # ── _trail_alpha ─────────────────────────────────────────

    def test_trail_alpha_high_blur_low_alpha(self):
        bg = BackgroundRenderer()
        alpha = bg._trail_alpha(_rs(blur_radius=1.0))
        assert alpha <= 20   # nearly transparent overlay = long trails

    def test_trail_alpha_low_blur_high_alpha(self):
        bg = BackgroundRenderer()
        alpha = bg._trail_alpha(_rs(blur_radius=0.0))
        assert alpha >= 200   # opaque overlay = instant clear

    def test_trail_alpha_midpoint(self):
        bg = BackgroundRenderer()
        alpha = bg._trail_alpha(_rs(blur_radius=0.5))
        assert 8 <= alpha <= 255

    def test_trail_alpha_always_positive(self):
        bg = BackgroundRenderer()
        for blur in [0.0, 0.1, 0.5, 0.9, 1.0]:
            alpha = bg._trail_alpha(_rs(blur_radius=blur))
            assert alpha > 0

    def test_trail_alpha_never_exceeds_255(self):
        bg = BackgroundRenderer()
        for blur in [0.0, 0.25, 0.5, 0.75, 1.0]:
            alpha = bg._trail_alpha(_rs(blur_radius=blur))
            assert alpha <= 255

    def test_trail_alpha_monotone_decreasing_with_blur(self):
        bg = BackgroundRenderer()
        alphas = [bg._trail_alpha(_rs(blur_radius=b))
                  for b in [0.0, 0.25, 0.5, 0.75, 1.0]]
        assert alphas == sorted(alphas, reverse=True)

    # ── describe() ───────────────────────────────────────────

    def test_describe_returns_dict(self):
        bg = BackgroundRenderer()
        d = bg.describe(_rs())
        assert isinstance(d, dict)

    def test_describe_has_required_keys(self):
        bg = BackgroundRenderer()
        d = bg.describe(_rs())
        assert "trail_alpha" in d
        assert "bloom_intensity" in d
        assert "background_dim" in d
        assert "base_color" in d

    def test_describe_trail_alpha_matches_method(self):
        bg = BackgroundRenderer()
        rs = _rs(blur_radius=0.7)
        d = bg.describe(rs)
        assert d["trail_alpha"] == bg._trail_alpha(rs)

    def test_describe_bloom_intensity_preserved(self):
        bg = BackgroundRenderer()
        d = bg.describe(_rs(bloom_intensity=0.8))
        assert d["bloom_intensity"] == pytest.approx(0.8)

    def test_describe_base_color_correct(self):
        bg = BackgroundRenderer(base_color=(10, 20, 30))
        d = bg.describe(_rs())
        assert d["base_color"] == (10, 20, 30)

    # ── draw() headless ──────────────────────────────────────

    def test_draw_headless_no_crash(self):
        bg = BackgroundRenderer()
        bg.draw(None, _rs())   # no pygame surface → should not crash
