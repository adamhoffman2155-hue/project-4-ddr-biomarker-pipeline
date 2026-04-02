"""
Utility helpers: logging, seeding, directory creation, and timing.
"""

import logging
import os
import time
from contextlib import contextmanager
from typing import Generator

import numpy as np


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and return a configured logger.

    Args:
        name: Logger name (typically ``__name__``).
        level: Logging level (default ``INFO``).

    Returns:
        A :class:`logging.Logger` with a stream handler attached.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    np.random.seed(seed)


def ensure_dir(path: str) -> str:
    """Create *path* (and parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        The same *path* for convenience.
    """
    os.makedirs(path, exist_ok=True)
    return path


@contextmanager
def Timer(description: str = "Operation") -> Generator[None, None, None]:
    """Context manager that logs elapsed wall-clock time.

    Usage::

        with Timer("Model training"):
            model.fit(X, y)

    Args:
        description: Label printed alongside the elapsed time.
    """
    logger = logging.getLogger("Timer")
    logger.info("%s started ...", description)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s finished in %.2f s", description, elapsed)
