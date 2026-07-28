"""Logging setup.

DynQuant never configures the root logger -- it is a library. Applications that
want output either configure logging themselves or call :func:`enable_logging`.
"""

from __future__ import annotations

import logging
import os

from .constants import ENV_LOG_LEVEL

__all__ = ["enable_logging", "get_logger"]

_ROOT_NAME = "dynquant"


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a submodule, e.g. ``get_logger(__name__)``."""
    if name.startswith(_ROOT_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def enable_logging(level: int | str | None = None) -> None:
    """Attach a stderr handler to the ``dynquant`` logger.

    Called by the CLI. Library users are left alone unless they opt in.
    """
    if level is None:
        level = os.environ.get(ENV_LOG_LEVEL, "INFO")
    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level)
    if not any(getattr(h, "_dynquant_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[dynquant] %(levelname)s %(name)s: %(message)s"))
        handler._dynquant_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False


# A library must not emit "No handlers could be found" warnings.
logging.getLogger(_ROOT_NAME).addHandler(logging.NullHandler())
