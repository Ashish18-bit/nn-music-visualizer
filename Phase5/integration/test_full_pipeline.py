"""
Full end-to-end pipeline integration test.
Runs without a real audio file — uses synthetic data.
"""
import sys
import os
import numpy as np
import pytest

# ── Phase 3 mapper ───────────────────────────────────────────
from mapping.mapper import EmotionToVisualMapper
from mapping.visual_state import VisualState

# ── Phase 4 renderer types ───────────────────────────────────
from renderer.visual_types import RendererState, hsl_to_rgb

EMOTIONS = ["happy", "sad", "calm", "angry", "energetic"]
MINIMAL_CFG = {
    "visual_params": {
        "particle_count_range": [20, 300],
        "particle_speed_range": [0.5, 6.0],
        "particle_size_range": [3, 18],
        "geo_rotation_speed_range": [0.002, 0.04],
    }
}


class TestPhase3To4Integration:
    """Phase 3 output feeds correctly into Phase 4 renderer."""

    def test_visual_state_to_renderer_state(self):
        mapper = EmotionToVisualMapper(min_dwell_sec=0.0)
        vs = mapper.map({"calm": 1.0})
        rs = RendererState.from_visual_state(vs, MINIMAL_CFG)
        assert isinstance(rs, RendererState)
        assert rs.emotion == "calm"

    def test_all_emotions_produce_valid_renderer_state(self):
        mapper = EmotionToVisualMapper(min_dwell_sec=0.0)
        for emotion in EMOTIONS:
            probs = {e: 0.025 for e in EMOTIONS}
            probs[emotion] = 0.9
            for _ in range(10):
                vs = mapper.map(probs)
            rs = RendererState.from_visual_state(vs, MINIMAL_CFG)
            assert 20 <= rs.particle_count <= 300
            assert rs.particle_speed > 0
            assert all(0 <= c <= 255 for c in rs.primary_color)

    def test_renderer_state_fields_finite(self):
        import math
        mapper = EmotionToVisualMapper(min_dwell_sec=0.0)
        for _ in range(30):
            probs = {e: 1/5 for e in EMOTIONS}
            vs = mapper.map(probs)
        rs = RendererState.from_visual_state(vs, MINIMAL_CFG)
        assert math.isfinite(rs.particle_speed)
        assert math.isfinite(rs.particle_size)
        assert math.isfinite(rs.geo_rotation_speed)


class TestPhase3Stability:
    """Phase 3 interpolator is stable over many frames."""

    def test_200_frames_no_crash(self):
        mapper = EmotionToVisualMapper(min_dwell_sec=0.0)
        for i in range(200):
            e = EMOTIONS[i % len(EMOTIONS)]
            probs = {em: 0.05 for em in EMOTIONS}
            probs[e] = 0.8
            vs = mapper.map(probs, amplitude=0.3)
        assert isinstance(vs, VisualState)

    def test_confidence_blending(self):
        mapper = EmotionToVisualMapper(
            confidence_threshold=0.9, min_dwell_sec=0.0
        )
        probs = {"happy": 0.45, "energetic": 0.45,
                 "sad": 0.03, "calm": 0.04, "angry": 0.03}
        vs = mapper.map(probs)
        assert vs.is_transitioning is True


class TestWebServerIntegration:
    """WebSocket server serialises VisualState correctly."""

    def test_server_serialises_state(self):
        from server.ws_server import VisualStateServer
        srv = VisualStateServer(port=8799)
        rs = RendererState()
        rs.emotion = "happy"
        rs.confidence = 0.9
        d = srv._serialise(rs, frame=1)
        assert d["emotion"] == "happy"
        assert d["confidence"] == pytest.approx(0.9, abs=0.01)
        assert "primary_color" in d
        assert "particle_count" in d

    def test_server_update_state(self):
        from server.ws_server import VisualStateServer
        srv = VisualStateServer(port=8800)
        mapper = EmotionToVisualMapper(min_dwell_sec=0.0)
        for _ in range(10):
            vs = mapper.map({"energetic": 0.9, "happy": 0.1})
            rs = RendererState.from_visual_state(vs, MINIMAL_CFG)
            srv.update_state(rs)
        assert srv.frame_count == 10
        assert srv._current_state.emotion in EMOTIONS


class TestHslToRgb:
    """Colour pipeline sanity checks."""

    def test_primary_colors(self):
        assert hsl_to_rgb(0, 1.0, 0.5) == (255, 0, 0)
        assert hsl_to_rgb(120, 1.0, 0.5) == (0, 255, 0)
        assert hsl_to_rgb(240, 1.0, 0.5) == (0, 0, 255)

    def test_all_emotion_colors_valid(self):
        from mapping.emotion_presets import get_preset
        for emotion in EMOTIONS:
            preset = get_preset(emotion)
            r, g, b = hsl_to_rgb(
                preset.color.hue,
                preset.color.saturation,
                preset.color.brightness,
            )
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255