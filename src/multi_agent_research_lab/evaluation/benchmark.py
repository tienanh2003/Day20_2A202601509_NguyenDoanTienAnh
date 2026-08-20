"""Benchmark skeleton for single-agent vs multi-agent."""

import logging
import re
from collections.abc import Callable
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.schemas import AgentName, BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Callable[[str], ResearchState]
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, and quality metrics for a run.

    Args:
        run_name: Name identifier for this benchmark run
        query: The research query to run
        runner: Function that executes the query and returns ResearchState

    Returns:
        Tuple of (final state, benchmark metrics)
    """
    started = perf_counter()
    error_msg: str | None = None
    failure = False

    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001
        failure = True
        error_msg = str(exc)
        logger.error(f"Benchmark run '{run_name}' failed: {exc}")
        # Return empty state for failed runs
        state = ResearchState(request=ResearchQuery(query=query))

    latency = perf_counter() - started

    # Calculate metrics from state
    cost_usd = _calculate_total_cost(state)
    citation_coverage = _calculate_citation_coverage(state)
    quality_score = _estimate_quality_score(state)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost_usd,
        quality_score=quality_score,
        citation_coverage=citation_coverage,
        failure_rate=1.0 if failure else 0.0,
        notes=error_msg or "",
    )

    return state, metrics


def _calculate_total_cost(state: ResearchState) -> float | None:
    """Sum up costs from all agent results."""
    total = 0.0
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if cost is not None:
            total += cost

    # If no cost recorded but we have tokens, estimate
    if total == 0.0 and state.agent_results:
        for result in state.agent_results:
            tokens = result.metadata.get("tokens")
            if tokens:
                # Rough estimate: $0.01 per 1K tokens
                total += tokens * 0.00001

    return round(total, 6) if total > 0 else None


def _calculate_citation_coverage(state: ResearchState) -> float | None:
    """Calculate citation coverage as ratio of cited sources.

    Returns None if final_answer is empty.
    """
    if not state.final_answer:
        return None

    # Count citation references like [1], [2], etc.
    citations = set(re.findall(r"\[(\d+)\]", state.final_answer))
    total_sources = len(state.sources)

    if total_sources == 0:
        return None

    coverage = len(citations) / total_sources
    return min(coverage, 1.0)  # Cap at 1.0


def _estimate_quality_score(state: ResearchState) -> float | None:
    """Estimate quality score based on available metrics.

    This is a neutral heuristic that scores based on content completeness,
    not favoring either baseline or multi-agent approach.
    """
    if not state.final_answer:
        return None

    score = 5.0  # Base score

    # Content length bonus (0-2 points)
    answer_len = len(state.final_answer)
    if answer_len > 500:
        score += 1.0
    if answer_len > 1000:
        score += 0.5

    # Citation bonus (0-2 points)
    citations = set(re.findall(r"\[(\d+)\]", state.final_answer))
    if citations:
        score += min(len(citations) * 0.3, 2.0)

    # Source diversity bonus (0-1 point)
    if state.sources:
        score += 0.5

    # Structure bonus (0-1 point) - check for common structural elements
    if any(marker in state.final_answer.lower() for marker in ['##', '###', '**', '- ', '* ']):
        score += 0.5

    return round(min(score, 10.0), 1)


def run_comparative_benchmark(
    queries: list[str],
    baseline_runner: Runner,
    multi_agent_runner: Runner,
) -> dict[str, list[tuple[ResearchState, BenchmarkMetrics]]]:
    """Run benchmarks for both baseline and multi-agent approaches.

    Args:
        queries: List of queries to benchmark
        baseline_runner: Function for single-agent baseline
        multi_agent_runner: Function for multi-agent workflow

    Returns:
        Dict with 'baseline' and 'multi_agent' results
    """
    results: dict[str, list[tuple[ResearchState, BenchmarkMetrics]]] = {
        "baseline": [],
        "multi_agent": [],
    }

    for i, query in enumerate(queries):
        logger.info(f"Benchmarking query {i + 1}/{len(queries)}: {query[:50]}...")

        # Baseline run
        try:
            baseline_state, baseline_metrics = run_benchmark(
                f"baseline-{i + 1}", query, baseline_runner
            )
            results["baseline"].append((baseline_state, baseline_metrics))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Baseline run failed: {exc}")

        # Multi-agent run
        try:
            multi_state, multi_metrics = run_benchmark(
                f"multi-agent-{i + 1}", query, multi_agent_runner
            )
            results["multi_agent"].append((multi_state, multi_metrics))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Multi-agent run failed: {exc}")

    return results
