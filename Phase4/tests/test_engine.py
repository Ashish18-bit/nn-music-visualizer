"""tests/test_engine.py — RenderEngine headless tests."""

import queue
import pytest
from renderer.visual_types import RendererState
from renderer.engine import RenderEngine

HEADLESS_CFG = {
    "renderer": {
        "width": 320, "height": 240, "fps": 60,
        "title": "Test", "fullscreen": False,
        "background_color": [8, 8, 15],
    },
    "particles": {"max_count": 50},
    "visual_params": {
        "particle_count_range": [5, 50],
        "particle_speed_range": [0.5, 4.0],
        "particle_size_range": [2, 12],
        "geo_rotation_speed_range": [0.002, 0.04],
    },
    "server": {},
}


@pytest.fixture
def engine():
    return RenderEngine(HEADLESS_CFG, headless=True, show_hud=False)


class TestRenderEngineConstruction:
    def test_default_state(self, engine):
        assert isinstance(engine.current_state, RendererState)

    def test_not_running_before_start(self, engine):
        assert engine.is_running is False

    def test_frame_count_zero(self, engine):
        assert engine.frame_count == 0

    def test_sub_renderers_created(self, engine):
        assert engine.background is not None
        assert engine.geometry is not None
        assert engine.particles is not None
        assert engine.canvas is not None


class TestPushState:
    def test_push_state_enqueues(self, engine):
        rs = RendererState()
        rs.emotion = "happy"
        engine.push_state(rs)
        assert not engine._state_queue.empty()

    def test_push_multiple_states(self, engine):
        for emotion in ["happy", "sad", "angry"]:
            rs = RendererState()
            rs.emotion = emotion
            engine.push_state(rs)
        assert engine._state_queue.qsize() == 3


class TestPollState:
    def test_poll_updates_current_state(self, engine):
        rs = RendererState()
        rs.emotion = "energetic"
        engine._state_queue.put(rs)
        engine._poll_state()
        assert engine.current_state.emotion == "energetic"

    def test_poll_keeps_latest_when_multiple(self, engine):
        for emotion in ["happy", "sad", "energetic"]:
            rs = RendererState()
            rs.emotion = emotion
            engine._state_queue.put(rs)
        engine._poll_state()
        assert engine.current_state.emotion == "energetic"

    def test_poll_empty_queue_no_change(self, engine):
        original_emotion = engine.current_state.emotion
        engine._poll_state()
        assert engine.current_state.emotion == original_emotion


class TestRunHeadless:
    def test_runs_n_frames_and_stops(self):
        eng = RenderEngine(HEADLESS_CFG, headless=True, show_hud=False)
        eng.run(max_frames=5)
        assert eng.frame_count == 5

    def test_stop_halts_loop(self):
        import threading
        eng = RenderEngine(HEADLESS_CFG, headless=True, show_hud=False)

        def stopper():
            import time
            time.sleep(0.1)
            eng.stop()

        t = threading.Thread(target=stopper, daemon=True)
        t.start()
        eng.run(max_frames=1000)
        t.join(timeout=2)
        assert eng.frame_count < 1000

    def test_render_frame_no_crash(self):
        eng = RenderEngine(HEADLESS_CFG, headless=True, show_hud=False)
        eng.canvas.init()
        eng._render_frame()
        eng.canvas.quit()

    def test_state_applied_during_run(self):
        q = queue.Queue()
        eng = RenderEngine(HEADLESS_CFG, state_queue=q, headless=True, show_hud=False)
        rs = RendererState()
        rs.emotion = "angry"
        q.put(rs)
        eng.run(max_frames=3)
        assert eng.current_state.emotion == "angry"


class TestFromConfigFile:
    def test_from_config_file(self, tmp_path):
        import yaml
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(HEADLESS_CFG))
        eng = RenderEngine.from_config_file(str(cfg_path), headless=True)
        assert isinstance(eng, RenderEngine)
