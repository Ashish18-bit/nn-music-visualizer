"""tests/test_geometry.py — GeometryRenderer tests."""

import math
import pytest
from renderer.visual_types import RendererState
from renderer.geometry import GeometryRenderer, _polygon_points


def _rs(**kwargs) -> RendererState:
    rs = RendererState()
    rs.geo_ring_count      = kwargs.get("geo_ring_count", 3)
    rs.geo_rotation_speed  = kwargs.get("geo_rotation_speed", 0.01)
    rs.geo_sides           = kwargs.get("geo_sides", 6)
    rs.geo_complexity      = kwargs.get("geo_complexity", 0.5)
    rs.geo_radial_bars     = kwargs.get("geo_radial_bars", True)
    rs.geo_bar_height      = kwargs.get("geo_bar_height", 0.5)
    rs.geo_base_radius     = kwargs.get("geo_base_radius", 80)
    rs.primary_color       = (167, 139, 250)
    rs.secondary_color     = (124, 58, 237)
    rs.beat_pulse          = kwargs.get("beat_pulse", 0.3)
    return rs


# ─────────────────────────────────────────────────────────────
#  _polygon_points helper
# ─────────────────────────────────────────────────────────────

class TestPolygonPoints:
    def test_triangle_count(self):
        pts = _polygon_points(0, 0, 50, 3, 0)
        assert len(pts) == 3

    def test_hexagon_count(self):
        pts = _polygon_points(0, 0, 50, 6, 0)
        assert len(pts) == 6

    def test_all_points_on_circle(self):
        cx, cy, r = 100, 200, 50
        pts = _polygon_points(cx, cy, r, 8, 0)
        for x, y in pts:
            dist = math.hypot(x - cx, y - cy)
            assert dist == pytest.approx(r, abs=0.1)

    def test_rotation_shifts_points(self):
        pts1 = _polygon_points(0, 0, 50, 6, 0)
        pts2 = _polygon_points(0, 0, 50, 6, math.pi / 6)
        assert pts1[0] != pytest.approx(pts2[0], abs=0.1)

    def test_centred_at_origin(self):
        pts = _polygon_points(0, 0, 50, 4, 0)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        assert cx == pytest.approx(0, abs=0.1)
        assert cy == pytest.approx(0, abs=0.1)


# ─────────────────────────────────────────────────────────────
#  GeometryRenderer construction
# ─────────────────────────────────────────────────────────────

class TestGeometryConstruction:
    def test_default_construction(self):
        g = GeometryRenderer()
        assert g.canvas_w == 1280
        assert g.canvas_h == 720

    def test_custom_dimensions(self):
        g = GeometryRenderer(canvas_w=800, canvas_h=600)
        assert g.canvas_w == 800
        assert g.canvas_h == 600

    def test_centre_computed(self):
        g = GeometryRenderer(canvas_w=800, canvas_h=600)
        assert g.cx == pytest.approx(400.0)
        assert g.cy == pytest.approx(300.0)

    def test_initial_ring_angles(self):
        g = GeometryRenderer()
        assert all(a == 0.0 for a in g._ring_angles)


# ─────────────────────────────────────────────────────────────
#  update()
# ─────────────────────────────────────────────────────────────

class TestGeometryUpdate:
    def test_update_increments_frame(self):
        g = GeometryRenderer()
        g.update(_rs())
        assert g._frame == 1

    def test_angles_change_after_update(self):
        g = GeometryRenderer()
        rs = _rs(geo_ring_count=3, geo_rotation_speed=0.05)
        g.update(rs)
        assert any(a != 0.0 for a in g._ring_angles[:3])

    def test_alternate_rings_rotate_opposite(self):
        g = GeometryRenderer()
        rs = _rs(geo_ring_count=2, geo_rotation_speed=0.1)
        g.update(rs)
        # ring 0 positive, ring 1 negative
        assert g._ring_angles[0] > 0
        assert g._ring_angles[1] < 0 or g._ring_angles[1] > 2 * math.pi - 0.2

    def test_many_updates_angles_stay_in_range(self):
        g = GeometryRenderer()
        rs = _rs(geo_ring_count=4, geo_rotation_speed=0.1)
        for _ in range(200):
            g.update(rs)
        for a in g._ring_angles[:4]:
            assert 0.0 <= a < 2 * math.pi

    def test_expands_angle_list_for_more_rings(self):
        g = GeometryRenderer()
        g.update(_rs(geo_ring_count=8))
        assert len(g._ring_angles) >= 8


# ─────────────────────────────────────────────────────────────
#  describe()
# ─────────────────────────────────────────────────────────────

class TestDescribe:
    def test_returns_dict(self):
        g = GeometryRenderer(800, 600)
        d = g.describe(_rs())
        assert isinstance(d, dict)

    def test_has_rings_key(self):
        g = GeometryRenderer(800, 600)
        d = g.describe(_rs(geo_ring_count=3))
        assert "rings" in d
        assert len(d["rings"]) == 3

    def test_ring_has_required_keys(self):
        g = GeometryRenderer(800, 600)
        d = g.describe(_rs(geo_ring_count=2))
        for ring in d["rings"]:
            assert "ring" in ring
            assert "radius" in ring
            assert "sides" in ring
            assert "angle" in ring

    def test_sides_matches_state(self):
        g = GeometryRenderer(800, 600)
        d = g.describe(_rs(geo_sides=5))
        for ring in d["rings"]:
            assert ring["sides"] == 5

    def test_radial_bars_flag(self):
        g = GeometryRenderer(800, 600)
        d1 = g.describe(_rs(geo_radial_bars=True))
        d2 = g.describe(_rs(geo_radial_bars=False))
        assert d1["radial_bars"] is True
        assert d2["radial_bars"] is False

    def test_ring_radius_increases(self):
        g = GeometryRenderer(800, 600)
        g.update(_rs(geo_ring_count=4))
        d = g.describe(_rs(geo_ring_count=4))
        radii = [r["radius"] for r in d["rings"]]
        assert radii == sorted(radii)

    def test_zero_rings_empty_list(self):
        g = GeometryRenderer(800, 600)

        class RS:
            geo_ring_count = 0
            geo_sides = 6
            geo_bar_height = 0.5
            geo_complexity = 0.4
            geo_radial_bars = False
            geo_base_radius = 80
            primary_color = (167, 139, 250)
            secondary_color = (124, 58, 237)
            geo_rotation_speed = 0.01
            beat_pulse = 0.3

        d = g.describe(RS())
        assert d["rings"] == []
