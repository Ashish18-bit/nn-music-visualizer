"""
server/ws_server.py
───────────────────
FastAPI WebSocket server that broadcasts VisualState JSON to
connected browser clients at the configured Hz rate.

The server runs in a background thread alongside the render engine.
Browser clients connect to ws://host:port/ws and receive a stream
of JSON frames they can use to drive a Three.js or Canvas renderer.

Endpoints
─────────
GET  /          → health check (JSON)
GET  /state     → current VisualState as JSON (REST snapshot)
WS   /ws        → real-time VisualState stream

JSON frame format (sent to each WS client each broadcast tick):
{
  "frame": 42,
  "emotion": "happy",
  "confidence": 0.87,
  "color_hue": 42.0,
  "color_saturation": 0.92,
  ...  (all RendererState fields flattened)
}
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Optional, Set

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    FastAPI = None
    WebSocket = None
    WebSocketDisconnect = Exception

from renderer.visual_types import RendererState


# ─────────────────────────────────────────────────────────────
#  ConnectionManager
# ─────────────────────────────────────────────────────────────

class ConnectionManager:
    """Tracks active WebSocket connections and broadcasts to all."""

    def __init__(self) -> None:
        self._connections: Set = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        logger.info("Client connected. Total: %d", len(self._connections))

    async def disconnect(self, ws) -> None:
        async with self._lock:
            self._connections.discard(ws)
        logger.info("Client disconnected. Total: %d", len(self._connections))

    async def broadcast(self, message: str) -> None:
        """Send message to all connected clients. Drops dead connections."""
        if not self._connections:
            return
        dead = set()
        async with self._lock:
            targets = set(self._connections)
        for ws in targets:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self._connections -= dead

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# ─────────────────────────────────────────────────────────────
#  VisualStateServer
# ─────────────────────────────────────────────────────────────

class VisualStateServer:
    """
    FastAPI server that streams VisualState to browser clients.

    Parameters
    ----------
    host         : bind address
    port         : bind port
    broadcast_hz : frames per second to push to WS clients
    max_connections : hard limit on simultaneous clients
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        broadcast_hz: int = 60,
        max_connections: int = 8,
    ) -> None:
        self.host = host
        self.port = port
        self.broadcast_hz = broadcast_hz
        self.max_connections = max_connections

        self._current_state: RendererState = RendererState()
        self._frame_count: int = 0
        self._lock = threading.Lock()
        self._server_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._manager = ConnectionManager()

        if _FASTAPI_AVAILABLE:
            self._app = self._build_app()
        else:
            self._app = None

    # ── public ──────────────────────────────────────────────

    def update_state(self, rs: RendererState) -> None:
        """Thread-safe: update the state that will be broadcast."""
        with self._lock:
            self._current_state = rs
            self._frame_count += 1

    def start_background(self) -> None:
        """Start the server in a daemon background thread."""
        if not _FASTAPI_AVAILABLE:
            logger.warning("FastAPI not available — server not started")
            return
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="ws-server",
        )
        self._server_thread.start()
        logger.info("WebSocket server starting on ws://%s:%d/ws",
                    self.host, self.port)

    def stop(self) -> None:
        """Signal the server to stop."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    @property
    def connection_count(self) -> int:
        return self._manager.connection_count

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def app(self):
        """Return the FastAPI app instance (for testing with TestClient)."""
        return self._app

    # ── private: FastAPI app ─────────────────────────────────

    def _build_app(self):
        app = FastAPI(title="NN Music Visualizer", version="1.0")
        manager = self._manager

        @app.get("/")
        async def health():
            with self._lock:
                rs = self._current_state
                frame = self._frame_count
            return JSONResponse({
                "status": "ok",
                "emotion": rs.emotion,
                "confidence": rs.confidence,
                "frame": frame,
                "connections": manager.connection_count,
            })

        @app.get("/state")
        async def get_state():
            with self._lock:
                rs = self._current_state
                frame = self._frame_count
            return JSONResponse(self._serialise(rs, frame))

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            if manager.connection_count >= self.max_connections:
                await ws.close(code=1008)
                return
            await manager.connect(ws)
            try:
                # Start broadcaster if not already running
                asyncio.ensure_future(self._broadcaster())
                # Keep connection alive by reading (client may send pings)
                while True:
                    try:
                        await asyncio.wait_for(ws.receive_text(), timeout=30)
                    except asyncio.TimeoutError:
                        pass
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                await manager.disconnect(ws)

        return app

    async def _broadcaster(self) -> None:
        """Coroutine: push state to all connected WS clients at broadcast_hz."""
        interval = 1.0 / max(1, self.broadcast_hz)
        while True:
            t0 = asyncio.get_event_loop().time()
            with self._lock:
                rs = self._current_state
                frame = self._frame_count
            msg = json.dumps(self._serialise(rs, frame))
            await self._manager.broadcast(msg)
            elapsed = asyncio.get_event_loop().time() - t0
            await asyncio.sleep(max(0, interval - elapsed))

    def _serialise(self, rs: RendererState, frame: int) -> dict:
        """Convert RendererState to a flat JSON-serialisable dict."""
        r, g, b = rs.primary_color
        sr, sg, sb = rs.secondary_color
        return {
            "frame": frame,
            "emotion": rs.emotion,
            "confidence": round(rs.confidence, 3),
            "is_transitioning": rs.is_transitioning,
            "primary_color": {"r": r, "g": g, "b": b},
            "secondary_color": {"r": sr, "g": sg, "b": sb},
            "particle_count": rs.particle_count,
            "particle_speed": round(rs.particle_speed, 3),
            "particle_size": round(rs.particle_size, 2),
            "particle_opacity": rs.particle_opacity,
            "particle_shape": rs.particle_shape,
            "particle_turbulence": round(rs.particle_turbulence, 3),
            "geo_ring_count": rs.geo_ring_count,
            "geo_rotation_speed": round(rs.geo_rotation_speed, 4),
            "geo_sides": rs.geo_sides,
            "geo_complexity": round(rs.geo_complexity, 3),
            "geo_radial_bars": rs.geo_radial_bars,
            "geo_bar_height": round(rs.geo_bar_height, 3),
            "blur_radius": round(rs.blur_radius, 3),
            "bloom_intensity": round(rs.bloom_intensity, 3),
            "background_dim": round(rs.background_dim, 3),
            "beat_pulse": round(rs.beat_pulse, 3),
        }

    def _run_server(self) -> None:
        """Blocking: run the uvicorn server (called in background thread)."""
        if not _FASTAPI_AVAILABLE:
            return
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            loop="asyncio",
            log_level="warning",
        )
        server = uvicorn.Server(config)
        self._loop.run_until_complete(server.serve())

    @classmethod
    def from_config(cls, cfg: dict) -> "VisualStateServer":
        s = cfg.get("server", {})
        return cls(
            host=s.get("host", "0.0.0.0"),
            port=s.get("port", 8765),
            broadcast_hz=s.get("broadcast_hz", 60),
            max_connections=s.get("max_connections", 8),
        )
