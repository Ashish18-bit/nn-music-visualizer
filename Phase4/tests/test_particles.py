"""tests/test_particles.py — ParticleSystem tests."""

import pytest
from renderer.visual_types import RendererState, Particle
from renderer.particles import (
    ParticleSystem,
    _star_points, _diamond_points, _spike_points, _drop_points,
)


def _rs(**kwargs) -> RendererState:
    rs = RendererState()
    rs.particle_count    = kwargs.get("particle_count", 20)
    rs.particle_speed    = kwargs.get("particle_speed", 2.0)
    rs.particle_size     = kwargs.get("particle_size", 8.0)
    rs.particle_opacity  = kwargs.get("particle_opacity", 200)
    rs.particle_lifetime = kwargs.get("particle_lifetime", 60)
    rs.particle_turbulence = kwargs.get("particle_turbulence", 0.0)
    rs.particle_shape    = kwargs.get("particle_shape", "circle")
    return rs


# ─────────────────────────────────────────────────────────────
#  Shape helpers
# ─────────────────────────────────────────────────────────────

class TestShapeHelpers:
    def test_star_points_count(self):
        pts = _star_points(0, 0, 10, 0)
        assert len(pts) == 10

    def test_diamond_points_count(self):
        pts = _diamond_points(0, 0, 10, 0)
        assert len(pts) == 4

    def test_spike_points_count(self):
        pts = _spike_points(0, 0, 10, 0)
        assert len(pts) == 3

    def test_drop_points_count(self):
        pts = _drop_points(0, 0, 10, 0)
        assert len(pts) == 8

    def test_star_points_all_tuples(self):
        for pt in _star_points(100, 200, 15, 0):
            assert len(pt) == 2

    def test_diamond_symmetric(self):
        pts = _diamond_points(0, 0, 10, 0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # Opposite vertices should sum to ~0
        assert abs(xs[0] + xs[2]) < 0.1
        assert abs(ys[1] + ys[3]) < 0.1


# ─────────────────────────────────────────────────────────────
#  ParticleSystem construction
# ─────────────────────────────────────────────────────────────

class TestParticleSystemConstruction:
    def test_default_construction(self):
        ps = ParticleSystem()
        assert ps.count == 0

    def test_custom_dimensions(self):
        ps = ParticleSystem(canvas_w=800, canvas_h=600, max_pool=100)
        assert ps.canvas_w == 800
        assert ps.canvas_h == 600

    def test_initial_count_zero(self):
        ps = ParticleSystem()
        assert ps.count == 0

    def test_particles_property_returns_list(self):
        ps = ParticleSystem()
        assert isinstance(ps.particles, list)


# ─────────────────────────────────────────────────────────────
#  update()
# ─────────────────────────────────────────────────────────────

class TestParticleSystemUpdate:
    def test_spawns_to_target_count(self):
        ps = ParticleSystem(canvas_w=800, canvas_h=600)
        rs = _rs(particle_count=30)
        ps.update(rs)
        assert ps.count == 30

    def test_respects_max_pool(self):
        ps = ParticleSystem(canvas_w=800, canvas_h=600, max_pool=10)
        rs = _rs(particle_count=50)
        ps.update(rs)
        assert ps.count <= 10

    def test_dead_particles_removed(self):
        ps = ParticleSystem(canvas_w=800, canvas_h=600)
        rs = _rs(particle_count=5)
        ps.update(rs)
        # Kill all particles
        for p in ps._particles:
            p.life = p.max_life + 1
        ps.update(rs)
        # Should have replaced dead ones
        assert ps.count == 5

    def test_count_stable_after_many_frames(self):
        ps = ParticleSystem(canvas_w=800, canvas_h=600)
        rs = _rs(particle_count=20)
        for _ in range(30):
            ps.update(rs)
        assert ps.count == 20

    def test_zero_count_clears_particles(self):
        ps = ParticleSystem(canvas_w=800, canvas_h=600)
        rs = _rs(particle_count=10)
        ps.update(rs)
        rs2 = _rs(particle_count=0)
        for _ in range(200):    # let old particles age out
            ps.update(rs2)
        assert ps.count == 0

    def test_turbulence_does_not_crash(self):
        ps = ParticleSystem(canvas_w=800, canvas_h=600)
        rs = _rs(particle_count=10, particle_turbulence=0.9)
        for _ in range(20):
            ps.update(rs)
        assert ps.count > 0


# ─────────────────────────────────────────────────────────────
#  draw_mock()
# ─────────────────────────────────────────────────────────────

class TestDrawMock:
    def test_returns_list(self):
        ps = ParticleSystem(800, 600)
        ps.update(_rs(particle_count=5))
        result = ps.draw_mock()
        assert isinstance(result, list)

    def test_length_matches_count(self):
        ps = ParticleSystem(800, 600)
        rs = _rs(particle_count=10)
        ps.update(rs)
        assert len(ps.draw_mock()) == ps.count

    def test_each_item_has_required_keys(self):
        ps = ParticleSystem(800, 600)
        ps.update(_rs(particle_count=5))
        for item in ps.draw_mock():
            assert "x" in item
            assert "y" in item
            assert "size" in item
            assert "shape" in item
            assert "alpha" in item
            assert "color" in item

    def test_alpha_in_range(self):
        ps = ParticleSystem(800, 600)
        ps.update(_rs(particle_count=10))
        for item in ps.draw_mock():
            assert 0 <= item["alpha"] <= 255

    def test_all_shapes_in_mock(self):
        for shape in ("circle", "star", "diamond", "spike", "drop"):
            ps = ParticleSystem(800, 600)
            ps.update(_rs(particle_count=5, particle_shape=shape))
            for item in ps.draw_mock():
                assert item["shape"] == shape


# ─────────────────────────────────────────────────────────────
#  clear()
# ─────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_removes_all(self):
        ps = ParticleSystem(800, 600)
        ps.update(_rs(particle_count=20))
        ps.clear()
        assert ps.count == 0

    def test_update_after_clear_respawns(self):
        ps = ParticleSystem(800, 600)
        ps.update(_rs(particle_count=10))
        ps.clear()
        ps.update(_rs(particle_count=10))
        assert ps.count == 10


# ─────────────────────────────────────────────────────────────
#  _in_bounds()
# ─────────────────────────────────────────────────────────────

class TestInBounds:
    def test_centre_is_in_bounds(self):
        ps = ParticleSystem(800, 600)
        p = Particle(x=400, y=300, vx=0, vy=0, size=5,
                     color=(255,255,255), opacity=200,
                     shape="circle", max_life=60)
        assert ps._in_bounds(p) is True

    def test_far_outside_is_not_in_bounds(self):
        ps = ParticleSystem(800, 600)
        p = Particle(x=2000, y=300, vx=0, vy=0, size=5,
                     color=(255,255,255), opacity=200,
                     shape="circle", max_life=60)
        assert ps._in_bounds(p) is False
