"""
main.py  (rebuilt for real emotion detection)
─────────────────────────────────────────────
Starts the VisualStateServer which contains:
  - /ws   WebSocket  → streams VisualState to browser
  - /audio WebSocket → receives PCM audio from browser
  - AudioProcessor   → runs CNN-LSTM every 1 second
"""

from __future__ import annotations

import argparse
import logging
import time

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def run_web(cfg: dict) -> None:
    from server.ws_server import VisualStateServer

    srv = VisualStateServer.from_config(cfg)
    srv.start_background()

    logger.info("=" * 55)
    logger.info("  NN Music Visualizer — real-time emotion detection")
    logger.info("  WebSocket : ws://0.0.0.0:%d/ws", cfg["server"]["port"])
    logger.info("  Audio in  : ws://0.0.0.0:%d/audio", cfg["server"]["port"])
    logger.info("  Health    : http://0.0.0.0:%d/", cfg["server"]["port"])
    logger.info("=" * 55)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping server.")
        srv.stop()


def run_headless(cfg: dict, max_sec: int = 5) -> None:
    """Used by tests — starts server, waits, stops."""
    from server.ws_server import VisualStateServer
    srv = VisualStateServer.from_config(cfg)
    srv.start_background()
    time.sleep(max_sec)
    srv.stop()
    logger.info("Headless run complete.")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NN Music Visualizer")
    p.add_argument("--mode", choices=["web", "headless"], default="web")
    p.add_argument("--config", default="config.yaml")
    return p


def main() -> None:
    args = _parser().parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.mode == "headless":
        run_headless(cfg)
    else:
        run_web(cfg)


if __name__ == "__main__":
    main()