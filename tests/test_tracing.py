"""Tests for tracing module."""

from multi_agent_research_lab.observability.tracing import (
    TraceCollector,
    get_trace_collector,
    trace_span,
)


def test_trace_span_records_duration() -> None:
    """trace_span should record duration."""
    with trace_span("test-span", {"key": "value"}) as span:
        pass

    assert span["duration_seconds"] is not None
    assert span["duration_seconds"] >= 0
    assert span["name"] == "test-span"


def test_trace_span_captures_attributes() -> None:
    """trace_span should capture attributes."""
    with trace_span("test", {"foo": "bar"}) as span:
        pass

    assert span["attributes"]["foo"] == "bar"


def test_trace_span_handles_exception() -> None:
    """trace_span should record exceptions."""
    with trace_span("failing-span") as span:
        raise ValueError("Test error")

    assert span["status"] == "error"
    assert "Test error" in span["error"]


def test_trace_collector() -> None:
    """TraceCollector should collect spans."""
    collector = TraceCollector()

    collector.start_span("span1", {"key": "value"})
    collector.end_span()

    summary = collector.get_summary()
    assert summary["total_spans"] == 1
    assert len(summary["spans"]) == 1


def test_trace_collector_adds_events() -> None:
    """TraceCollector should support events."""
    collector = TraceCollector()

    collector.start_span("span1")
    collector.add_event("event1", {"detail": "info"})
    collector.end_span()

    span = collector.spans[0]
    assert len(span["events"]) == 1
    assert span["events"][0]["name"] == "event1"


def test_get_trace_collector_singleton() -> None:
    """get_trace_collector should return singleton."""
    collector1 = get_trace_collector()
    collector2 = get_trace_collector()

    assert collector1 is collector2
