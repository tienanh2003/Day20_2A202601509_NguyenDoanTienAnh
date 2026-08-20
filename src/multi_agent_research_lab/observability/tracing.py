"""Tracing hooks with LangSmith/Langfuse integration.

This module provides observability for the multi-agent workflow using LangSmith.
Configure via environment variables:
- LANGSMITH_API_KEY: Enable LangSmith tracing
- LANGSMITH_PROJECT: Project name (default: multi-agent-research-lab)
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any
from uuid import uuid4

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# Global trace session
_trace_session: dict[str, Any] = {
    "session_id": None,
    "spans": [],
    "enabled": False,
}


def init_langsmith() -> bool:
    """Initialize LangSmith client if API key is configured.

    Returns True if LangSmith is enabled.
    """
    settings = get_settings()
    if not settings.langsmith_api_key:
        logger.debug("LangSmith API key not configured - tracing disabled")
        return False

    try:
        # LangGraph automatically picks up LANGSMITH_* env vars
        import os
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_TRACING"] = "true"

        logger.info(f"LangSmith tracing enabled: project={settings.langsmith_project}")
        _trace_session["enabled"] = True
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to initialize LangSmith: {exc}")
        return False


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager for tracing spans.

    When LangSmith is configured, traces are sent to LangSmith.
    Otherwise, traces are collected locally.
    """
    started = _get_time()
    span_id = str(uuid4())[:8]

    span: dict[str, Any] = {
        "span_id": span_id,
        "name": name,
        "attributes": attributes or {},
        "start_time": started,
        "duration_seconds": None,
        "events": [],
        "status": "ok",
    }

    # Start LangSmith span if available
    _start_langsmith_span(name, attributes)

    try:
        yield span
    except Exception as exc:  # noqa: BLE001
        span["status"] = "error"
        span["error"] = str(exc)
        _end_langsmith_span(name, span)
        raise
    finally:
        span["duration_seconds"] = _get_time() - started
        _end_langsmith_span(name, span)

        # Store locally if not using LangSmith
        if not _trace_session["enabled"]:
            _trace_session["spans"].append(span)

        logger.debug(
            f"[TRACE] {name} ({span_id}) - {span['duration_seconds']:.3f}s"
            + (f" [ERROR: {span.get('error')}]" if span.get("error") else "")
        )


def _get_time() -> float:
    """Get current time in seconds."""
    import time
    return time.perf_counter()


def _start_langsmith_span(name: str, attributes: dict[str, Any] | None) -> None:
    """Start a LangSmith span."""
    try:
        from langsmith.run_trees import RunTree

        settings = get_settings()
        if not settings.langsmith_api_key:
            return

        # Create run tree for LangSmith
        parent_run_id = _trace_session.get("parent_run_id")

        run = RunTree(
            name=name,
            run_type="chain",
            inputs=attributes or {},
            project_name=settings.langsmith_project,
            parent_run_id=parent_run_id,
        )
        _trace_session[f"run_{name}"] = run

    except ImportError:
        logger.debug("langsmith not installed - using local tracing only")
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to start LangSmith span: {exc}")


def _end_langsmith_span(name: str, span: dict[str, Any]) -> None:
    """End a LangSmith span."""
    try:
        run = _trace_session.pop(f"run_{name}", None)
        if run:
            run.end(
                outputs={
                    "duration_seconds": span.get("duration_seconds"),
                    "status": span.get("status"),
                    "error": span.get("error"),
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to end LangSmith span: {exc}")


class TraceCollector:
    """Simple collector for trace events without external dependencies."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self._current_span: dict[str, Any] | None = None
        self.session_id = str(uuid4())

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Start a new span."""
        self._current_span = {
            "span_id": str(uuid4())[:8],
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
            "session_id": self.session_id,
            "total_spans": len(self.spans),
            "total_duration": total_duration,
            "spans": self.spans,
        }

    def clear(self) -> None:
        """Clear all collected spans."""
        self.spans = []
        self._current_span = None

    def to_langsmith_format(self) -> list[dict[str, Any]]:
        """Convert to LangSmith-compatible format."""
        return [
            {
                "name": s["name"],
                "start_time": s["start_time"],
                "end_time": s.get("end_time", s["start_time"] + s.get("duration", 0)),
                "inputs": s["attributes"],
                "outputs": {"events": s["events"]},
                "error": s.get("error"),
            }
            for s in self.spans
        ]


@lru_cache(maxsize=1)
def get_trace_collector() -> TraceCollector:
    """Get the global trace collector singleton."""
    return TraceCollector()


def get_traces() -> dict[str, Any]:
    """Get all collected traces."""
    collector = get_trace_collector()
    return {
        "local_traces": collector.get_summary(),
        "langsmith_enabled": _trace_session["enabled"],
        "langsmith_project": get_settings().langsmith_project,
    }
