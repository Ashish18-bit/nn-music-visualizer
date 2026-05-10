"""
renderer/background.py
──────────────────────
BackgroundRenderer — handles the background layer:

  1. Trail effect    — semi-transparent dark overlay each frame creates
                       motion trails. Opacity driven by VisualState.blur_radius.
                       Low blur → trails fade fast (sharp/angry).
                       High blur → trails linger (calm/sad).

  2. Bloom glow      — a soft radial glow at the canvas centre,
                       intensity from VisualState.bloom_intensity.

  3. Clear mode      — for very low blur_radius, performs a hard clear
                       each frame (no trails at all).

These are drawn first each frame so everything else layers on top.
"""

from __future__ import annotations

import math
from typing import Tuple

from renderer.visual_types import RendererState, Color

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


class BackgroundRenderer:
    """
    Draws the background layer each frame.

    Parameters
    ----------
    canvas_w, canvas_h : surface dimensions
    base_color         : RGB base background colour
    """

    def __init__(
        self,
        canvas_w: int = 1280,
        canvas_h: int = 720,
        base_color: Color = (8, 8, 15),
    ) -> None:
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.base_color = base_color
        self._bloom_surface = None   # lazily created

    # ── public ──────────────────────────────────────────────

    def draw(self, surface, rs: RendererState) -> None:
        """
        Draw background trail and bloom onto surface.
        Must be called first in the render loop each frame.
        """
        if not _PYGAME_AVAILABLE:
            return
        self._draw_trail(surface, rs)
        if rs.bloom_intensity > 0.05:
            self._draw_bloom(surface, rs)

    def describe(self, rs: RendererState) -> dict:
        """Headless description for tests / WebSocket JSON."""
        trail_alpha = self._trail_alpha(rs)
        return {
            "trail_alpha": trail_alpha,
            "bloom_intensity": rs.bloom_intensity,
            "background_dim": rs.background_dim,
            "base_color": self.base_color,
        }

    # ── private ─────────────────────────────────────────────

    def _trail_alpha(self, rs: RendererState) -> int:
        """
        Compute the alpha of the dark overlay applied each frame.

        blur_radius=0.0 → alpha=255 (instant clear, no trails)
        blur_radius=1.0 → alpha=8   (very long trails)
        """
        # Invert: high blur = low alpha overlay = more trail
        raw = (1.0 - rs.blur_radius) * 240 + 10
        return int(max(8, min(255, raw)))

    def _draw_trail(self, surface, rs: RendererState) -> None:
        """Apply a semi-transparent dark rect to create motion trails."""
        alpha = self._trail_alpha(rs)
        try:
            overlay = pygame.Surface(
                (self.canvas_w, self.canvas_h), pygame.SRCALPHA
            )
            r, g, b = self.base_color
            overlay.fill((r, g, b, alpha))
            surface.blit(overlay, (0, 0))
        except Exception:
            pass

    def _draw_bloom(self, surface, rs: RendererState) -> None:
        """Draw a soft radial glow at the canvas centre."""
        cx, cy = self.canvas_w // 2, self.canvas_h // 2
        max_radius = int(min(self.canvas_w, self.canvas_h) * 0.35)
        n_circles = 6
        pr, pg, pb = rs.primary_color

        try:
            bloom_surf = pygame.Surface(
                (self.canvas_w, self.canvas_h), pygame.SRCALPHA
            )
            bloom_surf.fill((0, 0, 0, 0))

            for i in range(n_circles):
                t = i / n_circles
                radius = int(max_radius * (1.0 - t * 0.6))
                alpha = int(
                    rs.bloom_intensity * (1.0 - t) * 35
                    * (1.0 + rs.beat_pulse * 0.5)
                )
                if alpha > 0 and radius > 0:
                    pygame.draw.circle(
                        bloom_surf,
                        (pr, pg, pb, alpha),
                        (cx, cy),
                        radius,
                    )

            surface.blit(bloom_surf, (0, 0))
        except Exception:
            pass
