"""Benchmark report rendering."""

from datetime import datetime
from typing import Any

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown with analysis.

    TODO(student): Add richer analysis, examples, screenshots, and trace links.
    """
    if not metrics:
        return "# Benchmark Report\n\nNo benchmark results available.\n"

    lines = [
        "# Benchmark Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Metrics",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    # Calculate aggregates
    total_latency = 0.0
    total_cost = 0.0
    total_quality = 0.0
    total_citation = 0.0
    total_failure = 0.0
    count = len(metrics)

    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} |"
        )

        if item.latency_seconds:
            total_latency += item.latency_seconds
        if item.estimated_cost_usd is not None:
            total_cost += item.estimated_cost_usd
        if item.quality_score is not None:
            total_quality += item.quality_score
        if item.citation_coverage is not None:
            total_citation += item.citation_coverage
        if item.failure_rate is not None:
            total_failure += item.failure_rate

    # Add averages
    lines.extend(
        [
            "",
            f"**Averages** | {total_latency / count:.2f} | "
            f"${total_cost / count:.4f} | {total_quality / count:.1f} | "
            f"{total_citation / count:.0%} | {total_failure / count:.0%} |",
            "",
        ]
    )

    # Add failure analysis
    failed_runs = [m for m in metrics if m.failure_rate and m.failure_rate > 0]
    if failed_runs:
        lines.extend(["", "## Failure Analysis", ""])
        lines.append("The following runs failed:")
        for item in failed_runs:
            lines.append(f"- **{item.run_name}**: {item.notes}")

    return "\n".join(lines) + "\n"


def render_comparison_report(
    baseline_metrics: list[BenchmarkMetrics],
    multi_metrics: list[BenchmarkMetrics],
) -> str:
    """Render a comparative report between baseline and multi-agent approaches.

    Args:
        baseline_metrics: Metrics from baseline runs
        multi_metrics: Metrics from multi-agent runs

    Returns:
        Markdown-formatted comparison report
    """
    lines = [
        "# Comparative Benchmark Report: Baseline vs Multi-Agent",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Calculate averages
    baseline_avg = _average_metrics(baseline_metrics)
    multi_avg = _average_metrics(multi_metrics)

    # Summary table
    lines.extend(
        [
            "## Average Performance Comparison",
            "",
            "| Metric | Baseline | Multi-Agent | Winner |",
            "|---|---:|---:|---|",
            f"| Latency (s) | {baseline_avg['latency']:.2f} | {multi_avg['latency']:.2f} | "
            f"{_winner(baseline_avg['latency'], multi_avg['latency'], lower_is_better=True)} |",
            f"| Cost (USD) | {baseline_avg['cost']:.4f} | {multi_avg['cost']:.4f} | "
            f"{_winner(baseline_avg['cost'], multi_avg['cost'], lower_is_better=True)} |",
            f"| Quality Score | {baseline_avg['quality']:.1f} | {multi_avg['quality']:.1f} | "
            f"{_winner(baseline_avg['quality'], multi_avg['quality'], lower_is_better=False)} |",
            f"| Citation Coverage | {baseline_avg['citation']:.0%} | {multi_avg['citation']:.0%} | "
            f"{_winner(baseline_avg['citation'], multi_avg['citation'], lower_is_better=False)} |",
            f"| Failure Rate | {baseline_avg['failure']:.0%} | {multi_avg['failure']:.0%} | "
            f"{_winner(baseline_avg['failure'], multi_avg['failure'], lower_is_better=True)} |",
            "",
        ]
    )

    # Analysis section
    lines.extend(
        [
            "## Analysis",
            "",
            "### Key Findings:",
            "",
            _generate_findings(baseline_avg, multi_avg),
            "",
            "### Failure Mode Analysis:",
            "",
            _generate_failure_analysis(baseline_metrics, multi_metrics),
            "",
            "### Recommendations:",
            "",
            _generate_recommendations(baseline_avg, multi_avg),
            "",
        ]
    )

    return "\n".join(lines)


def _average_metrics(metrics: list[BenchmarkMetrics]) -> dict[str, float]:
    """Calculate average values from metrics list."""
    if not metrics:
        return {
            "latency": 0.0,
            "cost": 0.0,
            "quality": 0.0,
            "citation": 0.0,
            "failure": 0.0,
        }

    count = len(metrics)
    return {
        "latency": sum(m.latency_seconds for m in metrics) / count,
        "cost": sum(m.estimated_cost_usd or 0 for m in metrics) / count,
        "quality": sum(m.quality_score or 0 for m in metrics) / count,
        "citation": sum(m.citation_coverage or 0 for m in metrics) / count,
        "failure": sum(m.failure_rate or 0 for m in metrics) / count,
    }


def _winner(a: float, b: float, lower_is_better: bool) -> str:
    """Determine winner between two values."""
    if a == b:
        return "Tie"
    if lower_is_better:
        return "Baseline" if a < b else "Multi-Agent"
    return "Baseline" if a > b else "Multi-Agent"


def _generate_findings(baseline: dict[str, float], multi: dict[str, float]) -> str:
    """Generate key findings from comparison."""
    findings = []

    # Latency
    latency_diff = ((multi["latency"] - baseline["latency"]) / baseline["latency"] * 100) if baseline["latency"] > 0 else 0
    findings.append(f"- **Latency**: Multi-agent is {abs(latency_diff):.1f}% "
                   f"{'slower' if latency_diff > 0 else 'faster'} than baseline.")

    # Quality
    quality_diff = multi["quality"] - baseline["quality"]
    findings.append(f"- **Quality**: Multi-agent scores {abs(quality_diff):.1f} points "
                   f"{'higher' if quality_diff > 0 else 'lower'} than baseline.")

    # Citations
    citation_diff = multi["citation"] - baseline["citation"]
    findings.append(f"- **Citation Coverage**: Multi-agent has "
                   f"{citation_diff * 100:.0f} percentage points "
                   f"{'more' if citation_diff > 0 else 'less'} coverage.")

    # Cost
    cost_diff = ((multi["cost"] - baseline["cost"]) / baseline["cost"] * 100) if baseline["cost"] > 0 else 0
    findings.append(f"- **Cost**: Multi-agent costs {abs(cost_diff):.1f}% "
                   f"{'more' if cost_diff > 0 else 'less'} than baseline.")

    return "\n".join(findings)


def _generate_failure_analysis(
    baseline_metrics: list[BenchmarkMetrics],
    multi_metrics: list[BenchmarkMetrics],
) -> str:
    """Generate failure mode analysis."""
    baseline_failures = [m for m in baseline_metrics if m.failure_rate and m.failure_rate > 0]
    multi_failures = [m for m in multi_metrics if m.failure_rate and m.failure_rate > 0]

    analysis = []

    if not baseline_failures and not multi_failures:
        return "No failures detected in either approach."

    if baseline_failures:
        analysis.append(f"**Baseline failures** ({len(baseline_failures)}/{len(baseline_metrics)}):")
        for m in baseline_failures:
            analysis.append(f"  - {m.run_name}: {m.notes}")

    if multi_failures:
        analysis.append(f"\n**Multi-agent failures** ({len(multi_failures)}/{len(multi_metrics)}):")
        for m in multi_failures:
            analysis.append(f"  - {m.run_name}: {m.notes}")

    # Common failure patterns
    analysis.extend([
        "",
        "### Common Failure Patterns:",
        "",
        "1. **Single point of failure**: Baseline has no redundancy; if the single agent fails, the whole run fails.",
        "2. **Iteration limits**: Both approaches may hit max iterations without completing.",
        "3. **API errors**: Network issues, rate limits, or authentication problems can cause failures.",
        "4. **State corruption**: Multi-agent may have issues with state passing between agents.",
    ])

    return "\n".join(analysis)


def _generate_recommendations(baseline: dict[str, float], multi: dict[str, float]) -> str:
    """Generate recommendations based on results."""
    recommendations = []

    if multi["quality"] > baseline["quality"] + 1.0:
        recommendations.append(
            "- **Use multi-agent** when answer quality is critical and citations matter."
        )
    else:
        recommendations.append(
            "- **Use baseline** for simple queries where multi-agent overhead isn't justified."
        )

    if multi["latency"] < baseline["latency"] * 1.5:
        recommendations.append(
            "- Multi-agent provides good quality without significant latency penalty."
        )

    if multi["citation"] > baseline["citation"] + 0.2:
        recommendations.append(
            "- Multi-agent approach significantly improves citation coverage."
        )

    recommendations.extend([
        "",
        "### When to Use Each Approach:",
        "",
        "| Scenario | Recommended | Reason |",
        "|----------|-------------|--------|",
        "| Simple factual query | Baseline | Fast, low cost |",
        "| Complex research task | Multi-Agent | Better quality, citations |",
        "| Production system | Multi-Agent | Extensible, debuggable |",
        "| Rapid prototyping | Baseline | Quick iteration |",
    ])

    return "\n".join(recommendations)
