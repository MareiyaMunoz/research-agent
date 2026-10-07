"""Logging setup: console shows INFO, logs/agent.log keeps everything (DEBUG)."""

import logging
from pathlib import Path

LOG_DIR = Path("logs")
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATEFMT = "%H:%M:%S"

_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring handlers on first use."""
    global _configured
    if not _configured:
        _configure()
        _configured = True
    return logging.getLogger(name)


def _configure() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(FORMAT, datefmt=DATEFMT)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    LOG_DIR.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Third-party HTTP chatter drowns out the agent's own trail
    for noisy in ("httpx", "httpcore", "urllib3", "google", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
