"""
renderer/canvas.py
──────────────────
Canvas — wraps the Pygame display surface and provides:

  - Window creation (windowed + fullscreen toggle)
  - Frame rate limiting (clock.tick)
  - HUD overlay: emotion label, confidence bar, FPS counter
  - Screenshot capture
  - Headless mode (no display, for testing)

The Canvas is the top-level owner of the Pygame window.
All sub-renderers receive the surface from Canvas.surface.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

from renderer.visual_types import RendererState, Color, hsl_to_rgb

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


class Canvas:
    """
    Manages the Pygame window and frame lifecycle.

    Parameters
    ----------
    width, height   : initial window size
    fps             : target frame rate
    title           : window title bar text
    fullscreen      : start in fullscreen mode
    headless        : if True, create an off-screen surface (no window)
    bg_color        : RGB background fill colour
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        title: str = "NN Music Visualizer",
        fullscreen: bool = False,
        headless: bool = False,
        bg_color: Color = (8, 8, 15),
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.title = title
        self.bg_color = bg_color
        self.headless = headless
        self._fullscreen = fullscreen
        self._surface: Optional[object] = None
        self._clock = None
        self._font = None
        self._small_font = None
        self._frame_count = 0
        self._start_time = time.perf_counter()
        self._fps_measured = 0.0
        self._last_fps_time = time.perf_counter()
        self._fps_frame_acc = 0

    # ── lifecycle ────────────────────────────────────────────

    def init(self) -> None:
        """Initialise Pygame and create the window / surface."""
        if not _PYGAME_AVAILABLE:
            return
        if self.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        pygame.init()
        if self.headless:
            self._surface = pygame.Surface((self.width, self.height))
        elif self._fullscreen:
            self._surface = pygame.display.set_mode(
                (self.width, self.height),
                pygame.FULLSCREEN | pygame.DOUBLEBUF,
            )
        else:
            self._surface = pygame.display.set_mode(
                (self.width, self.height),
                pygame.DOUBLEBUF | pygame.RESIZABLE,
            )
        pygame.display.set_caption(self.title)
        self._clock = pygame.time.Clock()

        # Try to load fonts (graceful fallback)
        try:
            self._font = pygame.font.SysFont("monospace", 20, bold=True)
            self._small_font = pygame.font.SysFont("monospace", 14)
        except Exception:
            self._font = pygame.font.Font(None, 24)
            self._small_font = pygame.font.Font(None, 18)

    def quit(self) -> None:
        """Shut down Pygame."""
        if _PYGAME_AVAILABLE:
            pygame.quit()

    # ── frame lifecycle ──────────────────────────────────────

    def begin_frame(self) -> None:
        """Call at the start of each frame."""
        self._frame_count += 1
        self._fps_frame_acc += 1
        now = time.perf_counter()
        if now - self._last_fps_time >= 1.0:
            self._fps_measured = self._fps_frame_acc / (now - self._last_fps_time)
            self._fps_frame_acc = 0
            self._last_fps_time = now

    def end_frame(self) -> None:
        """Flip buffers and tick the clock."""
        if not _PYGAME_AVAILABLE or self.headless:
            return
        pygame.display.flip()
        if self._clock:
            self._clock.tick(self.fps)

    # ── drawing helpers ──────────────────────────────────────

    def draw_hud(self, rs: RendererState, show_fps: bool = True) -> None:
        """
        Draw the HUD overlay: emotion label, confidence bar, FPS.
        Semi-transparent so it doesn't distract from the visuals.
        """
        if not _PYGAME_AVAILABLE or self._surface is None:
            return
        if self._font is None:
            return

        pad = 12
        text_color = (220, 220, 240)
        dim_color = (100, 100, 120)

        try:
            # Emotion name
            emotion_surf = self._font.render(
                rs.emotion.upper(), True, rs.primary_color
            )
            self._surface.blit(emotion_surf, (pad, pad))

            # Confidence bar
            bar_w = 120
            bar_h = 6
            bar_x, bar_y = pad, pad + 28
            pygame.draw.rect(
                self._surface, (40, 40, 60),
                (bar_x, bar_y, bar_w, bar_h), border_radius=3,
            )
            fill_w = int(bar_w * rs.confidence)
            if fill_w > 0:
                pygame.draw.rect(
                    self._surface, rs.primary_color,
                    (bar_x, bar_y, fill_w, bar_h), border_radius=3,
                )

            conf_label = self._small_font.render(
                f"conf {rs.confidence:.0%}", True, dim_color
            )
            self._surface.blit(conf_label, (bar_x + bar_w + 8, bar_y - 2))

            # FPS
            if show_fps:
                fps_surf = self._small_font.render(
                    f"{self._fps_measured:.0f} fps", True, dim_color
                )
                self._surface.blit(
                    fps_surf,
                    (self.width - fps_surf.get_width() - pad, pad),
                )

            # Transitioning indicator
            if rs.is_transitioning:
                trans_surf = self._small_font.render(
                    "⟳ blending", True, (160, 160, 200)
                )
                self._surface.blit(trans_surf, (pad, pad + 46))

        except Exception:
            pass

    def screenshot(self, path: str | Path = "screenshot.png") -> Path:
        """Save current frame to a PNG file."""
        path = Path(path)
        if _PYGAME_AVAILABLE and self._surface is not None:
            pygame.image.save(self._surface, str(path))
        return path

    def toggle_fullscreen(self) -> None:
        """Toggle between windowed and fullscreen."""
        if not _PYGAME_AVAILABLE:
            return
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            self._surface = pygame.display.set_mode(
                (self.width, self.height),
                pygame.FULLSCREEN | pygame.DOUBLEBUF,
            )
        else:
            self._surface = pygame.display.set_mode(
                (self.width, self.height),
                pygame.DOUBLEBUF | pygame.RESIZABLE,
            )

    def handle_resize(self, w: int, h: int) -> None:
        """Handle window resize event."""
        if not _PYGAME_AVAILABLE:
            return
        self.width = w
        self.height = h
        if not self._fullscreen and not self.headless:
            self._surface = pygame.display.set_mode(
                (w, h), pygame.DOUBLEBUF | pygame.RESIZABLE
            )

    # ── properties ───────────────────────────────────────────

    @property
    def surface(self):
        return self._surface

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps_measured(self) -> float:
        return self._fps_measured

    @property
    def elapsed_sec(self) -> float:
        return time.perf_counter() - self._start_time
