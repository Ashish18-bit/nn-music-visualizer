"""
main.py
───────
Unified entry point — wires Phase 1 through Phase 4 together.

Modes
─────
  desktop   : Pygame window (default)
  web       : FastAPI WebSocket server + browser client
  demo      : cycles through emotion presets, no model needed
  headless  : no display, used for CI / testing

Usage
─────
  python main.py                        # desktop demo
  python main.py --mode web             # start WebSocket server
  python main.py --mode demo --fps 30   # demo at 30 fps
  python main.py --mode headless --frames 120
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────
#  Demo emotion cycle
# ─────────────────────────────────────────────────────────────

DEMO_SEQUENCE = [
    ("calm",      4.0),
    ("happy",     3.5),
    ("energetic", 3.0),
    ("happy",     2.5),
    ("calm",      4.0),
    ("sad",       4.0),
    ("calm",      3.0),
    ("energetic", 3.0),
    ("angry",     2.5),
    ("energetic", 2.5),
    ("calm",      4.0),
]


def _demo_inference_thread(
    state_queue: queue.Queue,
    cfg: dict,
    stop_event: threading.Event,
) -> None:
    """
    Simulates Phase 2+3 output by cycling through emotion presets.
    Pushes RendererState into the queue at ~60 Hz.
    """
    try:
        from renderer.visual_types import RendererState
        # Phase 3 mapper
        sys.path.insert(0, "..")   # allow importing phase3 if co-located
        try:
            from mapping.mapper import EmotionToVisualMapper
            mapper = EmotionToVisualMapper.from_config(cfg)
            _use_mapper = True
        except ImportError:
            _use_mapper = False
            mapper = None

    except ImportError as e:
        logger.error("Import error in inference thread: %s", e)
        return

    seq_idx = 0
    emotion, dwell = DEMO_SEQUENCE[seq_idx]
    dwell_end = time.perf_counter() + dwell
    frame_interval = 1.0 / 60

    while not stop_event.is_set():
        t0 = time.perf_counter()

        # Advance demo sequence
        if t0 >= dwell_end:
            seq_idx = (seq_idx + 1) % len(DEMO_SEQUENCE)
            emotion, dwell = DEMO_SEQUENCE[seq_idx]
            dwell_end = t0 + dwell
            logger.info("Demo emotion → %s", emotion)

        # Build probability dict
        probs = {e: 0.025 for e in
                 ["happy", "sad", "calm", "angry", "energetic"]}
        probs[emotion] = 0.9

        # Map to VisualState then RendererState
        if _use_mapper and mapper:
            vs = mapper.map(probs)
            rs = RendererState.from_visual_state(vs, cfg)
        else:
            rs = RendererState()
            rs.emotion = emotion

        try:
            state_queue.put_nowait(rs)
        except queue.Full:
            pass

        elapsed = time.perf_counter() - t0
        time.sleep(max(0, frame_interval - elapsed))


# ─────────────────────────────────────────────────────────────
#  Main modes
# ─────────────────────────────────────────────────────────────

def run_desktop(cfg: dict, args: argparse.Namespace) -> None:
    """Desktop Pygame mode."""
    from renderer.engine import RenderEngine
    state_q: queue.Queue = queue.Queue(maxsize=4)
    stop_evt = threading.Event()

    # Start demo inference thread
    inf_thread = threading.Thread(
        target=_demo_inference_thread,
        args=(state_q, cfg, stop_evt),
        daemon=True,
        name="inference",
    )
    inf_thread.start()

    engine = RenderEngine(cfg, state_queue=state_q, headless=False)
    try:
        engine.run()
    finally:
        stop_evt.set()
        logger.info("Desktop renderer stopped.")


def run_web(cfg: dict, args: argparse.Namespace) -> None:
    """Web mode: start WebSocket server + demo inference."""
    from renderer.visual_types import RendererState
    from server.ws_server import VisualStateServer

    srv = VisualStateServer.from_config(cfg)
    srv.start_background()

    stop_evt = threading.Event()
    state_q: queue.Queue = queue.Queue(maxsize=4)

    inf_thread = threading.Thread(
        target=_demo_inference_thread,
        args=(state_q, cfg, stop_evt),
        daemon=True,
        name="inference",
    )
    inf_thread.start()

    logger.info("WebSocket server running at ws://%s:%d/ws",
                cfg["server"]["host"], cfg["server"]["port"])
    logger.info("Open server/index.html in your browser.")
    logger.info("Press Ctrl+C to stop.")

    try:
        while True:
            try:
                rs = state_q.get_nowait()
                srv.update_state(rs)
            except queue.Empty:
                pass
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        logger.info("Stopping web server.")
    finally:
        stop_evt.set()


def run_headless(cfg: dict, args: argparse.Namespace) -> None:
    """Headless mode for testing / CI."""
    from renderer.engine import RenderEngine
    state_q: queue.Queue = queue.Queue(maxsize=4)
    stop_evt = threading.Event()

    inf_thread = threading.Thread(
        target=_demo_inference_thread,
        args=(state_q, cfg, stop_evt),
        daemon=True,
    )
    inf_thread.start()

    engine = RenderEngine(cfg, state_queue=state_q, headless=True)
    frames = args.frames or 120
    engine.run(max_frames=frames)
    stop_evt.set()
    logger.info("Headless run complete: %d frames", engine.frame_count)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main",
        description="Neural Network Music Visualizer — Phase 4",
    )
    p.add_argument("--mode", choices=["desktop", "web", "headless"],
                   default="desktop")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--frames", type=int, default=None,
                   help="Max frames (headless mode)")
    p.add_argument("--fullscreen", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.fps:
        cfg["renderer"]["fps"] = args.fps
    if args.fullscreen:
        cfg["renderer"]["fullscreen"] = True

    dispatch = {
        "desktop":  run_desktop,
        "web":      run_web,
        "headless": run_headless,
    }
    dispatch[args.mode](cfg, args)


if __name__ == "__main__":
    main()
