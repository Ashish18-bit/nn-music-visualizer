"""
renderer/geometry.py
────────────────────
GeometryRenderer — draws the procedural geometric layer:

  1. Polygon rings   — concentric n-sided polygons that rotate slowly.
                       Ring count, sides, and speed come from RendererState.
  2. Radial bars     — frequency-spectrum bars radiating from the centre.
                       Heights are driven by audio amplitude + beat_pulse.

Both are drawn with alpha transparency so they layer naturally
under the particle system.

All geometry is centred at (canvas_w/2, canvas_h/2).
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

from renderer.visual_types import RendererState, Color, hsl_to_rgb

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


def _polygon_points(
    cx: float, cy: float,
    radius: float,
    sides: int,
    rotation: float,
) -> List[Tuple[float, float]]:
    """Compute vertices of a regular polygon."""
    pts = []
    for i in range(sides):
        angle = rotation + (2 * math.pi * i / sides)
        pts.append((
            cx + math.cos(angle) * radius,
            cy + math.sin(angle) * radius,
        ))
    return pts


class GeometryRenderer:
    """
    Renders concentric polygon rings and radial frequency bars.

    Parameters
    ----------
    canvas_w, canvas_h : surface dimensions
    """

    def __init__(self, canvas_w: int = 1280, canvas_h: int = 720) -> None:
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.cx = canvas_w / 2
        self.cy = canvas_h / 2
        # Per-ring rotation angles, updated each frame
        self._ring_angles: List[float] = [0.0] * 8
        self._frame = 0

    # ── public ──────────────────────────────────────────────

    def update(self, rs: RendererState) -> None:
        """Advance rotation angles for all rings."""
        self._frame += 1
        n = max(1, rs.geo_ring_count)
        while len(self._ring_angles) < n:
            self._ring_angles.append(0.0)
        for i in range(n):
            # Alternate rotation direction per ring
            direction = 1 if i % 2 == 0 else -1
            speed = rs.geo_rotation_speed * (1.0 + i * 0.15)
            self._ring_angles[i] = (
                self._ring_angles[i] + direction * speed
            ) % (2 * math.pi)

    def draw(
        self,
        surface,
        rs: RendererState,
        audio_data: Optional[List[float]] = None,
    ) -> None:
        """
        Draw rings and radial bars onto a Pygame surface.

        Parameters
        ----------
        surface    : pygame.Surface to draw on
        rs         : current RendererState
        audio_data : optional list of normalised amplitudes [0,1]
                     per frequency bin (for radial bars)
        """
        if not _PYGAME_AVAILABLE:
            return
        self._draw_rings(surface, rs)
        if rs.geo_radial_bars:
            self._draw_radial_bars(surface, rs, audio_data)

    def describe(self, rs: RendererState) -> dict:
        """
        Headless description of what would be drawn.
        Used by tests and the WebSocket JSON output.
        """
        n = rs.geo_ring_count
        rings = []
        base = rs.geo_base_radius
        spacing = int(min(self.canvas_w, self.canvas_h) * 0.07)
        for i in range(n):
            radius = base + i * spacing
            rings.append({
                "ring": i,
                "radius": radius,
                "sides": rs.geo_sides,
                "angle": round(self._ring_angles[i] if i < len(self._ring_angles) else 0.0, 3),
            })
        return {
            "rings": rings,
            "radial_bars": rs.geo_radial_bars,
            "bar_height": rs.geo_bar_height,
            "complexity": rs.geo_complexity,
        }

    # ── private ─────────────────────────────────────────────

    def _draw_rings(self, surface, rs: RendererState) -> None:
        n = rs.geo_ring_count
        base = rs.geo_base_radius
        spacing = int(min(self.canvas_w, self.canvas_h) * 0.07)

        for i in range(n):
            radius = base + i * spacing
            # Opacity fades for outer rings
            opacity = max(15, int((1.0 - i / max(n, 1)) * rs.geo_complexity * 100))
            # Hue shifts slightly per ring
            ring_color = rs.primary_color
            if i % 2 == 1:
                ring_color = rs.secondary_color

            angle = self._ring_angles[i] if i < len(self._ring_angles) else 0.0
            pts = _polygon_points(
                self.cx, self.cy, radius, rs.geo_sides, angle
            )
            if len(pts) >= 2:
                try:
                    ring_surf = pygame.Surface(
                        (self.canvas_w, self.canvas_h), pygame.SRCALPHA
                    )
                    ring_surf.fill((0, 0, 0, 0))
                    pygame.draw.polygon(
                        ring_surf,
                        (*ring_color, opacity),
                        [(int(x), int(y)) for x, y in pts],
                        width=max(1, int(rs.geo_complexity * 3)),
                    )
                    surface.blit(ring_surf, (0, 0))
                except Exception:
                    pass

    def _draw_radial_bars(
        self,
        surface,
        rs: RendererState,
        audio_data: Optional[List[float]],
    ) -> None:
        n_bars = 48
        inner_r = rs.geo_base_radius
        max_h = int(min(self.canvas_w, self.canvas_h) * 0.18 * rs.geo_bar_height)
        max_h = max(4, max_h)

        beat_boost = 1.0 + rs.beat_pulse * 0.5

        for i in range(n_bars):
            angle = (2 * math.pi * i / n_bars) - math.pi / 2

            # Bar height: audio data if available, else sinusoidal demo
            if audio_data and i < len(audio_data):
                amp = float(audio_data[i])
            else:
                t = self._frame * 0.05
                amp = 0.3 + 0.5 * abs(
                    math.sin(t + i * 0.3 + rs.beat_pulse * 2)
                )
            amp = max(0.0, min(1.0, amp * beat_boost))
            bar_len = int(inner_r + amp * max_h)

            x1 = self.cx + math.cos(angle) * inner_r
            y1 = self.cy + math.sin(angle) * inner_r
            x2 = self.cx + math.cos(angle) * bar_len
            y2 = self.cy + math.sin(angle) * bar_len

            opacity = int(60 + amp * 180)
            try:
                bar_surf = pygame.Surface(
                    (self.canvas_w, self.canvas_h), pygame.SRCALPHA
                )
                bar_surf.fill((0, 0, 0, 0))
                pygame.draw.line(
                    bar_surf,
                    (*rs.primary_color, opacity),
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    max(1, int(rs.geo_complexity * 2.5)),
                )
                surface.blit(bar_surf, (0, 0))
            except Exception:
                pass
