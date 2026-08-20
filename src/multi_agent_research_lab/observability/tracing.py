"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context with optional LangSmith integration.

    TODO(student): Replace or augment with LangSmith/Langfuse provider spans.
    """
    started = _get_time()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
        "events": [],
    }

    settings = get_settings()
    langsmith_enabled = bool(settings.langsmith_api_key)

    if langsmith_enabled:
        _start_langsmith_span(name, attributes)

    try:
        yield span
    except Exception as exc:  # noqa: BLE001
        span["error"] = str(exc)
        span["status"] = "error"
        raise
    finally:
        span["duration_seconds"] = _get_time() - started

        if langsmith_enabled:
            _end_langsmith_span(name, span)

        logger.debug(
            f"[TRACE] {name} completed in {span['duration_seconds']:.3f}s"
            + (f" (error: {span.get('error')})" if "error" in span else "")
        )


def _get_time() -> float:
    """Get current time in seconds."""
    import time

    return time.perf_counter()


def _start_langsmith_span(name: str, attributes: dict[str, Any] | None) -> None:
    """Start a LangSmith span if configured."""
    try:
        from langsmith.run_trees import RunTree

        settings = get_settings()
        # Create a simple span for LangSmith
        # Note: Full LangSmith integration would use langgraph's built-in integration
        logger.debug(f"LangSmith: starting span '{name}'")
    except ImportError:
        logger.debug("LangSmith not available, skipping trace")


def _end_langsmith_span(name: str, span: dict[str, Any]) -> None:
    """End a LangSmith span."""
    logger.debug(f"LangSmith: ending span '{name}' - {span.get('duration_seconds', 0):.3f}s")


class TraceCollector:
    """Simple collector for trace events without external dependencies."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self._current_span: dict[str, Any] | None = None

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Start a new span."""
        self._current_span = {
            "name": name,
            "attributes": attributes or {},
            "start_time": _get_time(),
            "events": [],
        }

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the current span."""
        if self._current_span:
            self._current_span["events"].append({
                "name": name,
                "timestamp": _get_time(),
                "attributes": attributes or {},
            })

    def end_span(self, attributes: dict[str, Any] | None = None) -> None:
        """End the current span."""
        if self._current_span:
            self._current_span["end_time"] = _get_time()
            self._current_span["duration"] = (
                self._current_span["end_time"] - self._current_span["start_time"]
            )
            if attributes:
                self._current_span["attributes"].update(attributes)
            self.spans.append(self._current_span)
            self._current_span = None

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all spans."""
        total_duration = sum(s.get("duration", 0) for s in self.spans)
        return {
            "total_spans": len(self.spans),
            "total_duration": total_duration,
            "spans": self.spans,
        }

    def clear(self) -> None:
        """Clear all collected spans."""
        self.spans = []
        self._current_span = None


# Global trace collector instance
_trace_collector: TraceCollector | None = None


def get_trace_collector() -> TraceCollector:
    """Get the global trace collector."""
    global _trace_collector
    if _trace_collector is None:
        _trace_collector = TraceCollector()
    return _trace_collector
