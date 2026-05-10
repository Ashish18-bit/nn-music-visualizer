"""tests/test_ws_server.py — VisualStateServer REST endpoint tests."""

import json
import pytest
from renderer.visual_types import RendererState

try:
    from fastapi.testclient import TestClient
    from server.ws_server import VisualStateServer
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _FASTAPI_AVAILABLE,
    reason="fastapi/httpx not installed"
)


@pytest.fixture
def server():
    return VisualStateServer(host="127.0.0.1", port=8799, broadcast_hz=10)


@pytest.fixture
def client(server):
    return TestClient(server.app)


# ─────────────────────────────────────────────────────────────
#  VisualStateServer construction
# ─────────────────────────────────────────────────────────────

class TestServerConstruction:
    def test_default_state_emotion(self, server):
        assert server._current_state.emotion == "calm"

    def test_initial_frame_count(self, server):
        assert server.frame_count == 0

    def test_initial_connection_count(self, server):
        assert server.connection_count == 0

    def test_app_not_none(self, server):
        assert server.app is not None


# ─────────────────────────────────────────────────────────────
#  update_state()
# ─────────────────────────────────────────────────────────────

class TestUpdateState:
    def test_updates_emotion(self, server):
        rs = RendererState()
        rs.emotion = "happy"
        server.update_state(rs)
        assert server._current_state.emotion == "happy"

    def test_increments_frame_count(self, server):
        server.update_state(RendererState())
        server.update_state(RendererState())
        assert server.frame_count == 2

    def test_thread_safe_multiple_updates(self, server):
        import threading
        def worker():
            for _ in range(50):
                rs = RendererState()
                server.update_state(rs)
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert server.frame_count == 200


# ─────────────────────────────────────────────────────────────
#  REST endpoints
# ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_status_ok(self, client):
        data = client.get("/").json()
        assert data["status"] == "ok"

    def test_has_emotion_field(self, client):
        data = client.get("/").json()
        assert "emotion" in data

    def test_has_frame_field(self, client):
        data = client.get("/").json()
        assert "frame" in data

    def test_emotion_matches_state(self, client, server):
        rs = RendererState()
        rs.emotion = "sad"
        server.update_state(rs)
        data = client.get("/").json()
        assert data["emotion"] == "sad"


class TestStateEndpoint:
    def test_returns_200(self, client):
        r = client.get("/state")
        assert r.status_code == 200

    def test_has_required_fields(self, client):
        data = client.get("/state").json()
        required = [
            "frame", "emotion", "confidence",
            "primary_color", "particle_count",
            "particle_speed", "geo_ring_count",
            "blur_radius", "bloom_intensity",
            "beat_pulse",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_primary_color_has_rgb(self, client):
        data = client.get("/state").json()
        col = data["primary_color"]
        assert "r" in col and "g" in col and "b" in col

    def test_confidence_in_range(self, client):
        data = client.get("/state").json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_particle_count_positive(self, client):
        data = client.get("/state").json()
        assert data["particle_count"] >= 0

    def test_emotion_updated_after_push(self, client, server):
        rs = RendererState()
        rs.emotion = "energetic"
        rs.confidence = 0.88
        server.update_state(rs)
        data = client.get("/state").json()
        assert data["emotion"] == "energetic"
        assert data["confidence"] == pytest.approx(0.88, abs=0.01)


# ─────────────────────────────────────────────────────────────
#  _serialise()
# ─────────────────────────────────────────────────────────────

class TestSerialise:
    def test_serialise_returns_dict(self, server):
        rs = RendererState()
        d = server._serialise(rs, frame=5)
        assert isinstance(d, dict)

    def test_frame_field(self, server):
        rs = RendererState()
        d = server._serialise(rs, frame=42)
        assert d["frame"] == 42

    def test_all_values_json_serialisable(self, server):
        rs = RendererState()
        d = server._serialise(rs, frame=0)
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_is_transitioning_bool(self, server):
        rs = RendererState()
        rs.is_transitioning = True
        d = server._serialise(rs, frame=0)
        assert d["is_transitioning"] is True

    def test_emotion_string(self, server):
        rs = RendererState()
        rs.emotion = "angry"
        d = server._serialise(rs, frame=0)
        assert d["emotion"] == "angry"

    def test_rounded_floats(self, server):
        rs = RendererState()
        rs.blur_radius = 0.123456789
        d = server._serialise(rs, frame=0)
        # Should be rounded to 3 decimal places
        assert d["blur_radius"] == pytest.approx(0.123, abs=0.001)


# ─────────────────────────────────────────────────────────────
#  from_config()
# ─────────────────────────────────────────────────────────────

class TestFromConfig:
    def test_builds_from_config(self):
        cfg = {
            "server": {
                "host": "127.0.0.1",
                "port": 9000,
                "broadcast_hz": 30,
                "max_connections": 4,
            }
        }
        srv = VisualStateServer.from_config(cfg)
        assert srv.port == 9000
        assert srv.broadcast_hz == 30
        assert srv.max_connections == 4

    def test_empty_config_uses_defaults(self):
        srv = VisualStateServer.from_config({})
        assert isinstance(srv, VisualStateServer)
