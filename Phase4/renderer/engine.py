"""
renderer/engine.py
──────────────────
RenderEngine — the top-level render loop.

Wires together:
  Canvas → BackgroundRenderer → GeometryRenderer → ParticleSystem → HUD

The engine runs at the configured FPS and reads the latest
RendererState from a thread-safe queue filled by the inference
thread (Phase 2 model output → Phase 3 mapper → Phase 4 renderer).

In standalone / demo mode it cycles through preset emotions
automatically to demonstrate the visuals without a live model.

Architecture
────────────
                 ┌──────────────────────────┐
    Phase 2+3    │  Inference thread        │
    (model +     │  produces VisualState    │
     mapper)     │  → puts in state_queue   │
                 └──────────┬───────────────┘
                            │  thread-safe Queue
                 ┌──────────▼───────────────┐
                 │  RenderEngine.run()      │
                 │  (main thread, 60 fps)   │
                 │                          │
                 │  canvas.begin_frame()    │
                 │  background.draw()       │
                 │  geometry.draw()         │
                 │  particles.draw()        │
                 │  canvas.draw_hud()       │
                 │  canvas.end_frame()      │
                 └──────────────────────────┘
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import yaml

from renderer.visual_types import RendererState
from renderer.canvas import Canvas
from renderer.background import BackgroundRenderer
from renderer.geometry import GeometryRenderer
from renderer.particles import ParticleSystem

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

logger = logging.getLogger(__name__)


class RenderEngine:
    """
    Main render loop. Must run on the main thread (Pygame requirement).

    Parameters
    ----------
    cfg          : full config dict (loaded from config.yaml)
    state_queue  : thread-safe Queue[RendererState] fed by inference thread
                   If None, the engine runs in demo mode.
    headless     : no display (for testing)
    show_hud     : whether to draw the HUD overlay
    """

    def __init__(
        self,
        cfg: dict,
        state_queue: Optional[queue.Queue] = None,
        headless: bool = False,
        show_hud: bool = True,
    ) -> None:
        r_cfg = cfg.get("renderer", {})
        self.cfg = cfg
        self.headless = headless
        self.show_hud = show_hud
        self._state_queue = state_queue or queue.Queue()
        self._running = False
        self._frame_count = 0

        # Current renderer state (updated from queue each frame)
        self._current_rs = RendererState()

        # Sub-renderers
        w = r_cfg.get("width", 1280)
        h = r_cfg.get("height", 720)
        fps = r_cfg.get("fps", 60)
        bg = tuple(r_cfg.get("background_color", [8, 8, 15]))

        self.canvas = Canvas(
            width=w, height=h, fps=fps,
            title=r_cfg.get("title", "NN Music Visualizer"),
            fullscreen=r_cfg.get("fullscreen", False),
            headless=headless,
            bg_color=bg,
        )
        self.background = BackgroundRenderer(w, h, base_color=bg)
        self.geometry = GeometryRenderer(w, h)
        self.particles = ParticleSystem(
            canvas_w=w, canvas_h=h,
            max_pool=cfg.get("particles", {}).get("max_count", 400),
        )

    # ── public ──────────────────────────────────────────────

    def run(self, max_frames: Optional[int] = None) -> None:
        """
        Start the render loop. Blocks until the window is closed
        or max_frames is reached (useful for testing).

        Parameters
        ----------
        max_frames : stop after this many frames (None = run forever)
        """
        self.canvas.init()
        self._running = True
        logger.info("RenderEngine started (%dx%d @ %d fps)",
                    self.canvas.width, self.canvas.height, self.canvas.fps)

        try:
            while self._running:
                if max_frames and self._frame_count >= max_frames:
                    break

                # ── event handling ───────────────────────────
                if not self.headless and _PYGAME_AVAILABLE:
                    for event in pygame.event.get():
                        self._handle_event(event)
                    if not self._running:
                        break

                # ── pull latest state from queue ─────────────
                self._poll_state()

                # ── render ───────────────────────────────────
                self.canvas.begin_frame()
                self._render_frame()
                self.canvas.end_frame()

                self._frame_count += 1

        except Exception as exc:
            logger.error("RenderEngine crashed: %s", exc, exc_info=True)
        finally:
            self._running = False
            self.canvas.quit()
            logger.info("RenderEngine stopped after %d frames",
                        self._frame_count)

    def stop(self) -> None:
        """Signal the render loop to stop."""
        self._running = False

    def push_state(self, rs: RendererState) -> None:
        """Push a new RendererState into the queue (called from any thread)."""
        try:
            self._state_queue.put_nowait(rs)
        except queue.Full:
            pass   # drop frame if queue is full — renderer catches up

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def current_state(self) -> RendererState:
        return self._current_rs

    @property
    def is_running(self) -> bool:
        return self._running

    # ── private ─────────────────────────────────────────────

    def _poll_state(self) -> None:
        """Drain the state queue, keeping only the latest state."""
        latest = None
        try:
            while True:
                latest = self._state_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            self._current_rs = latest

    def _render_frame(self) -> None:
        """Execute one full render frame."""
        rs = self._current_rs
        surface = self.canvas.surface
        if surface is None:
            return

        # 1. Background + trail effect
        self.background.draw(surface, rs)

        # 2. Geometry (rings + radial bars)
        self.geometry.update(rs)
        self.geometry.draw(surface, rs)

        # 3. Particles
        self.particles.update(rs)
        self.particles.draw(surface)

        # 4. HUD
        if self.show_hud:
            self.canvas.draw_hud(rs)

    def _handle_event(self, event) -> None:
        """Handle Pygame events."""
        if not _PYGAME_AVAILABLE:
            return
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._running = False
            elif event.key == pygame.K_f:
                self.canvas.toggle_fullscreen()
            elif event.key == pygame.K_s:
                path = self.canvas.screenshot()
                logger.info("Screenshot saved: %s", path)
        elif event.type == pygame.VIDEORESIZE:
            self.canvas.handle_resize(event.w, event.h)

    # ── factory ─────────────────────────────────────────────

    @classmethod
    def from_config_file(
        cls,
        path: str = "config.yaml",
        state_queue: Optional[queue.Queue] = None,
        headless: bool = False,
    ) -> "RenderEngine":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(cfg, state_queue=state_queue, headless=headless)
