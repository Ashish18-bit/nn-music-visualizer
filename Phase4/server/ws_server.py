"""
ws_server.py  (rebuilt)
───────────────────────
Two WebSocket endpoints:

  GET  /          → health check
  GET  /state     → current VisualState snapshot
  WS   /ws        → browser receives VisualState JSON stream
  WS   /audio     → browser sends raw PCM audio chunks here

Flow:
  Browser mic/file → /audio WS → AudioProcessor →
  CNN-LSTM → Phase3 mapper → /ws WS → browser canvas
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
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from renderer.visual_types import RendererState
from server.audio_processor import AudioProcessor


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: Set = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)

    async def disconnect(self, ws) -> None:
        async with self._lock:
            self._connections.discard(ws)

    async def broadcast(self, message: str) -> None:
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
    def count(self) -> int:
        return len(self._connections)


class VisualStateServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        broadcast_hz: int = 30,
        max_connections: int = 8,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.broadcast_hz = broadcast_hz
        self.max_connections = max_connections

        self._current_state = RendererState()
        self._current_probs = {}
        self._frame_count = 0
        self._lock = threading.Lock()
        self._manager = ConnectionManager()
        self._server_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Audio processor — runs CNN-LSTM inference
        self._processor = AudioProcessor(
            checkpoint_path=checkpoint_path,
            on_state_update=self._on_new_visual_state,
            inference_interval_sec=1.0,
        )

        if _FASTAPI_AVAILABLE:
            self._app = self._build_app()
        else:
            self._app = None

    # ── called by AudioProcessor when new emotion detected ───

    def _on_new_visual_state(self, vs, probs: dict) -> None:
        """
        Callback from inference thread → convert VisualState to
        RendererState and store for broadcasting.
        """
        cfg = {
            "visual_params": {
                "particle_count_range": [20, 300],
                "particle_speed_range": [0.5, 6.0],
                "particle_size_range": [3, 18],
                "geo_rotation_speed_range": [0.002, 0.04],
            }
        }
        try:
            rs = RendererState.from_visual_state(vs, cfg)
            with self._lock:
                self._current_state = rs
                self._current_probs = probs
                self._frame_count += 1
        except Exception as e:
            logger.error("VisualState conversion failed: %s", e)

    def update_state(self, rs: RendererState) -> None:
        with self._lock:
            self._current_state = rs
            self._frame_count += 1

    # ── FastAPI app ──────────────────────────────────────────

    def _build_app(self):
        app = FastAPI(title="NN Music Visualizer", version="2.0")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        manager = self._manager

        @app.get("/")
        async def health():
            with self._lock:
                rs = self._current_state
                frame = self._frame_count
                probs = self._current_probs
            return JSONResponse({
                "status": "ok",
                "emotion": rs.emotion,
                "confidence": round(rs.confidence, 3),
                "frame": frame,
                "connections": manager.count,
                "probs": {k: round(v, 3) for k, v in probs.items()},
            })

        @app.get("/state")
        async def get_state():
            with self._lock:
                rs = self._current_state
                frame = self._frame_count
            return JSONResponse(self._serialise(rs, frame))

        # ── Visual state WebSocket (browser receives) ─────────
        @app.websocket("/ws")
        async def ws_visual(ws: WebSocket):
            if manager.count >= self.max_connections:
                await ws.close(code=1008)
                return
            await manager.connect(ws)
            try:
                asyncio.ensure_future(self._broadcaster())
                while True:
                    try:
                        await asyncio.wait_for(
                            ws.receive_text(), timeout=30
                        )
                    except asyncio.TimeoutError:
                        pass
            except (WebSocketDisconnect, Exception):
                pass
            finally:
                await manager.disconnect(ws)

        # ── Audio WebSocket (browser sends PCM) ───────────────
        @app.websocket("/audio")
        async def ws_audio(ws: WebSocket):
            await ws.accept()
            client_sr = 44100
            logger.info("Audio client connected")
            try:
                while True:
                    try:
                        # Receive either text (metadata) or binary (PCM)
                        data = await asyncio.wait_for(
                            ws.receive(), timeout=10
                        )

                        if "text" in data:
                            # Metadata: {"sampleRate": 44100}
                            meta = json.loads(data["text"])
                            client_sr = meta.get("sampleRate", 44100)
                            await ws.send_text(json.dumps({"status": "ready"}))

                        elif "bytes" in data:
                            # Raw PCM Float32 audio chunk
                            self._processor.ingest_pcm(
                                data["bytes"], client_sr
                            )

                    except asyncio.TimeoutError:
                        # Send keepalive
                        await ws.send_text(json.dumps({"status": "alive"}))

            except (WebSocketDisconnect, Exception) as e:
                logger.info("Audio client disconnected: %s", e)

        return app

    async def _broadcaster(self) -> None:
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

    def start_background(self) -> None:
        if not _FASTAPI_AVAILABLE:
            return
        self._processor.start()
        self._server_thread = threading.Thread(
            target=self._run_server, daemon=True, name="ws-server"
        )
        self._server_thread.start()
        logger.info("Server on ws://%s:%d", self.host, self.port)

    def stop(self) -> None:
        self._processor.stop()
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    @property
    def connection_count(self) -> int:
        return self._manager.count

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def app(self):
        return self._app

    def _run_server(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        config = uvicorn.Config(
            self._app, host=self.host, port=self.port,
            loop="asyncio", log_level="warning",
        )
        server = uvicorn.Server(config)
        self._loop.run_until_complete(server.serve())

    @classmethod
    def from_config(cls, cfg: dict) -> "VisualStateServer":
        s = cfg.get("server", {})
        return cls(
            host=s.get("host", "0.0.0.0"),
            port=s.get("port", 8765),
            broadcast_hz=s.get("broadcast_hz", 30),
            max_connections=s.get("max_connections", 8),
            checkpoint_path=s.get("checkpoint_path", None),
        )