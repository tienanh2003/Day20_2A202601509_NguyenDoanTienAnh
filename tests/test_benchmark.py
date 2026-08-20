"""Tests for benchmark module."""

import pytest

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    _calculate_citation_coverage,
    _calculate_total_cost,
    _estimate_quality_score,
    run_benchmark,
)


def test_run_benchmark_returns_metrics() -> None:
    """run_benchmark should return state and metrics."""
    def dummy_runner(query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        state.final_answer = "Test answer"
        return state

    state, metrics = run_benchmark("test-run", "test query", dummy_runner)

    assert isinstance(metrics, BenchmarkMetrics)
    assert metrics.run_name == "test-run"
    assert metrics.latency_seconds > 0
    assert metrics.failure_rate == 0.0


def test_run_benchmark_handles_failure() -> None:
    """run_benchmark should handle runner failures gracefully."""
    def failing_runner(query: str) -> ResearchState:
        raise RuntimeError("Test error")

    state, metrics = run_benchmark("failing-run", "test", failing_runner)

    assert metrics.failure_rate == 1.0
    assert "Test error" in metrics.notes


def test_calculate_total_cost() -> None:
    """_calculate_total_cost should sum agent costs."""
    from multi_agent_research_lab.core.schemas import AgentResult, AgentName

    state = ResearchState(request=ResearchQuery(query="test"))
    state.agent_results = [
        AgentResult(agent=AgentName.RESEARCHER, content="", metadata={"cost_usd": 0.01}),
        AgentResult(agent=AgentName.ANALYST, content="", metadata={"cost_usd": 0.02}),
    ]

    cost = _calculate_total_cost(state)
    assert cost == 0.03


def test_calculate_citation_coverage() -> None:
    """_calculate_citation_coverage should count citations."""
    from multi_agent_research_lab.core.schemas import SourceDocument

    state = ResearchState(request=ResearchQuery(query="test"))
    state.sources = [
        SourceDocument(title="A", snippet=""),
        SourceDocument(title="B", snippet=""),
    ]
    state.final_answer = "Answer with [1] and [2] citations"

    coverage = _calculate_citation_coverage(state)
    assert coverage == 1.0  # Both sources cited


def test_estimate_quality_score() -> None:
    """_estimate_quality_score should assess quality heuristics."""
    from multi_agent_research_lab.core.schemas import SourceDocument

    state = ResearchState(request=ResearchQuery(query="test"))
    state.sources = [SourceDocument(title="Test", snippet="")]
    state.research_notes = "Notes"
    state.analysis_notes = "Analysis"
    state.final_answer = "Answer with [1] citation"

    score = _estimate_quality_score(state)
    assert score >= 5.0  # Should have base + bonuses
