"""Tests for benchmark module."""

from multi_agent_research_lab.core.schemas import AgentName, AgentResult, BenchmarkMetrics, ResearchQuery, SourceDocument
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
    state = ResearchState(request=ResearchQuery(query="test"))
    state.agent_results = [
        AgentResult(agent=AgentName.RESEARCHER, content="", metadata={"cost_usd": 0.01}),
        AgentResult(agent=AgentName.ANALYST, content="", metadata={"cost_usd": 0.02}),
    ]

    cost = _calculate_total_cost(state)
    assert cost == 0.03


def test_calculate_total_cost_estimates_from_tokens() -> None:
    """_calculate_total_cost should estimate from tokens if no cost recorded."""
    state = ResearchState(request=ResearchQuery(query="test"))
    state.agent_results = [
        AgentResult(agent=AgentName.RESEARCHER, content="", metadata={"tokens": 1000}),
    ]

    cost = _calculate_total_cost(state)
    assert cost is not None
    assert cost > 0


def test_calculate_citation_coverage() -> None:
    """_calculate_citation_coverage should count citations."""
    state = ResearchState(request=ResearchQuery(query="test"))
    state.sources = [
        SourceDocument(title="A", snippet=""),
        SourceDocument(title="B", snippet=""),
    ]
    state.final_answer = "Answer with [1] and [2] citations"

    coverage = _calculate_citation_coverage(state)
    assert coverage == 1.0  # Both sources cited


def test_calculate_citation_coverage_no_sources() -> None:
    """_calculate_citation_coverage should return None if no sources."""
    state = ResearchState(request=ResearchQuery(query="test"))
    state.final_answer = "Answer with [1] citation"

    coverage = _calculate_citation_coverage(state)
    assert coverage is None


def test_estimate_quality_score() -> None:
    """_estimate_quality_score should assess quality heuristics."""
    state = ResearchState(request=ResearchQuery(query="test"))
    state.sources = [SourceDocument(title="Test", snippet="")]
    state.final_answer = "Answer with [1] citation and proper structure"

    score = _estimate_quality_score(state)
    assert score >= 5.0  # Should have base + bonuses
    assert score <= 10.0


def test_estimate_quality_score_no_answer() -> None:
    """_estimate_quality_score should return None if no final answer."""
    state = ResearchState(request=ResearchQuery(query="test"))

    score = _estimate_quality_score(state)
    assert score is None
