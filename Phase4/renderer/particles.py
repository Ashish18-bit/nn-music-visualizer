"""
renderer/particles.py
─────────────────────
ParticleSystem — manages a pool of Particle objects.

Responsibilities
────────────────
- Maintain a live pool of particles sized to RendererState.particle_count
- Spawn new particles to replace dead ones
- Update all live particles each frame (position, turbulence, age)
- Draw all live particles onto a Pygame surface

Particle shapes supported (all drawn via Pygame primitives):
  circle   — filled circle
  star     — 5-point star polygon
  diamond  — rotated square
  spike    — elongated triangle
  drop     — teardrop (circle + triangle)
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

from renderer.visual_types import Particle, RendererState, Color, with_alpha

# Pygame is imported lazily so unit tests can run without a display
try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  Shape drawing helpers  (pure geometry, return polygon points)
# ─────────────────────────────────────────────────────────────

def _star_points(cx: float, cy: float, r: float, angle: float) -> List:
    """5-point star polygon points."""
    pts = []
    for i in range(10):
        a = angle + i * math.pi / 5 - math.pi / 2
        radius = r if i % 2 == 0 else r * 0.4
        pts.append((cx + math.cos(a) * radius, cy + math.sin(a) * radius))
    return pts


def _diamond_points(cx: float, cy: float, r: float, angle: float) -> List:
    """4-point diamond."""
    pts = []
    for i in range(4):
        a = angle + i * math.pi / 2
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    return pts


def _spike_points(cx: float, cy: float, r: float, angle: float) -> List:
    """Elongated triangle spike."""
    return [
        (cx + math.cos(angle) * r * 1.8,
         cy + math.sin(angle) * r * 1.8),
        (cx + math.cos(angle + 2.5) * r * 0.4,
         cy + math.sin(angle + 2.5) * r * 0.4),
        (cx + math.cos(angle - 2.5) * r * 0.4,
         cy + math.sin(angle - 2.5) * r * 0.4),
    ]


def _drop_points(cx: float, cy: float, r: float, angle: float) -> List:
    """Teardrop approximated as a polygon."""
    pts = []
    for i in range(8):
        a = angle + i * math.pi / 4
        scale = 1.0 if i < 4 else 0.5
        pts.append((cx + math.cos(a) * r * scale,
                    cy + math.sin(a) * r * scale))
    return pts


# ─────────────────────────────────────────────────────────────
#  ParticleSystem
# ─────────────────────────────────────────────────────────────

class ParticleSystem:
    """
    Manages the full lifecycle of particles each frame.

    Parameters
    ----------
    canvas_w, canvas_h : canvas dimensions in pixels
    max_pool           : hard cap on number of particles (memory limit)
    """

    def __init__(
        self,
        canvas_w: int = 1280,
        canvas_h: int = 720,
        max_pool: int = 500,
    ) -> None:
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.max_pool = max_pool
        self._particles: List[Particle] = []
        self._frame = 0

    # ── public ──────────────────────────────────────────────

    def update(self, rs: RendererState) -> None:
        """
        One frame tick:
          1. Age and move all existing particles
          2. Remove dead ones
          3. Spawn replacements to hit target count
        """
        self._frame += 1

        # Update existing
        for p in self._particles:
            p.update(turbulence=rs.particle_turbulence)

        # Remove dead + out-of-bounds
        self._particles = [
            p for p in self._particles
            if p.alive and self._in_bounds(p)
        ]

        # Spawn to reach target count
        target = min(rs.particle_count, self.max_pool)
        needed = target - len(self._particles)
        for _ in range(max(0, needed)):
            self._particles.append(
                Particle.spawn(self.canvas_w, self.canvas_h, rs)
            )

    def draw(self, surface) -> None:
        """Draw all live particles onto a Pygame surface."""
        if not _PYGAME_AVAILABLE:
            return
        for p in self._particles:
            self._draw_particle(surface, p)

    def draw_mock(self) -> List[dict]:
        """
        Headless draw — returns list of dicts describing what would be drawn.
        Used by unit tests and the WebSocket server (JSON output).
        """
        result = []
        for p in self._particles:
            result.append({
                "x": round(p.x, 1),
                "y": round(p.y, 1),
                "size": round(p.size, 1),
                "shape": p.shape,
                "alpha": p.alpha,
                "color": p.color,
                "angle": round(p.angle, 3),
            })
        return result

    def clear(self) -> None:
        """Remove all particles (call on reset / track change)."""
        self._particles.clear()

    @property
    def count(self) -> int:
        return len(self._particles)

    @property
    def particles(self) -> List[Particle]:
        return list(self._particles)

    # ── private: drawing ─────────────────────────────────────

    def _draw_particle(self, surface, p: Particle) -> None:
        """Dispatch to the correct shape drawing function."""
        if not _PYGAME_AVAILABLE:
            return
        alpha = p.alpha
        if alpha <= 0:
            return

        r = max(1, int(p.size))
        cx, cy = int(p.x), int(p.y)
        color = p.color

        # Create a small surface for alpha blending
        try:
            shape_surf = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
            shape_surf.fill((0, 0, 0, 0))
            local_cx, local_cy = r * 2, r * 2

            if p.shape == "circle":
                pygame.draw.circle(shape_surf, (*color, alpha),
                                   (local_cx, local_cy), r)

            elif p.shape == "star":
                pts = _star_points(local_cx, local_cy, r, p.angle)
                if len(pts) >= 3:
                    pygame.draw.polygon(shape_surf, (*color, alpha), pts)

            elif p.shape == "diamond":
                pts = _diamond_points(local_cx, local_cy, r, p.angle)
                pygame.draw.polygon(shape_surf, (*color, alpha), pts)

            elif p.shape == "spike":
                pts = _spike_points(local_cx, local_cy, r, p.angle)
                pygame.draw.polygon(shape_surf, (*color, alpha), pts)

            elif p.shape == "drop":
                pts = _drop_points(local_cx, local_cy, r, p.angle)
                if len(pts) >= 3:
                    pygame.draw.polygon(shape_surf, (*color, alpha), pts)
            else:
                pygame.draw.circle(shape_surf, (*color, alpha),
                                   (local_cx, local_cy), r)

            surface.blit(shape_surf, (cx - r * 2, cy - r * 2))

        except Exception:
            pass   # never let a single particle crash the renderer

    # ── private: bounds ──────────────────────────────────────

    def _in_bounds(self, p: Particle, margin: int = 60) -> bool:
        return (
            -margin <= p.x <= self.canvas_w + margin
            and -margin <= p.y <= self.canvas_h + margin
        )
