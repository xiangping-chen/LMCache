# SPDX-License-Identifier: Apache-2.0
"""Instrumentation utilities for LMCache storage backends.

These utilities are controlled via the ``LMCACHE_INSTRUMENTATION``
environment variable. When set to ``1`` (or any truthy value),
detailed performance counters and I/O path tracing are enabled.

This is useful for debugging, baseline testing, and performance
analysis, but incurs a small runtime overhead. In production, the
default (disabled) should be used.
"""

# Standard
import os
import time
from typing import Callable, Optional

_logger = None

# Cache the env-var check at module load time so we avoid calling
# os.environ.get() on every I/O operation in the hot path.
_ENABLED: bool = os.environ.get("LMCACHE_INSTRUMENTATION", "0").lower() in (
    "1",
    "true",
    "yes",
)


def is_enabled() -> bool:
    """Return True if instrumentation is enabled via environment variable.

    Controlled by ``LMCACHE_INSTRUMENTATION`` env var (read once at
    module load time).  Set to ``1`` to enable.
    """
    return _ENABLED


def log_trace(tag: str, message: str) -> None:
    """Log an instrumentation trace message if instrumentation is enabled.

    Args:
        tag: Short identifier for the trace point (e.g. ``GDS_TRACE``).
        message: Human-readable trace message.
    """
    if is_enabled():
        global _logger
        if _logger is None:
            from lmcache.logging import init_logger

            _logger = init_logger(__name__)
        _logger.info("[%s] %s", tag, message)


class Timer:
    """Context manager that times a block and records the latency."""

    def __init__(self, callback: Optional[Callable[[float], None]] = None) -> None:
        self.callback = callback
        self.latency_ms: float = 0.0
        self._t0: Optional[float] = None

    def __enter__(self) -> "Timer":
        if is_enabled():
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        if is_enabled() and self._t0 is not None:
            self.latency_ms = (time.perf_counter() - self._t0) * 1000
            if self.callback:
                self.callback(self.latency_ms)
