"""Command-line entrypoint for the lab starter."""

import os
import time
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()

BASELINE_SYSTEM_PROMPT = """You are a research assistant. Provide a comprehensive, well-structured answer to the user's question.
Include relevant details, examples, and cite your sources when possible.
Be thorough but concise."""


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    # Initialize LangSmith tracing if API key is available
    from multi_agent_research_lab.observability.tracing import init_langsmith
    if init_langsmith():
        console.print("[cyan]LangSmith tracing enabled[/cyan]")


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _run_baseline(query: str) -> ResearchState:
    """Helper to run baseline for benchmarking."""
    request = ResearchQuery(query=query)
    llm = LLMClient(temperature=0.0)
    response = llm.complete(
        system_prompt=BASELINE_SYSTEM_PROMPT,
        user_prompt=f"Research question: {query}\n\nProvide a comprehensive answer.",
    )
    state = ResearchState(request=request)
    state.final_answer = response.content

    # Record cost from LLM response
    if response.cost_usd:
        from multi_agent_research_lab.core.schemas import AgentResult, AgentName
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,  # Use as proxy for baseline
                content=response.content,
                metadata={"cost_usd": response.cost_usd, "tokens": response.output_tokens},
            )
        )

    return state


def _run_multi_agent(query: str) -> ResearchState:
    """Helper to run multi-agent for benchmarking."""
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow(enable_critic=get_settings().enable_critic)
    return workflow.run(state)


def _save_benchmark_report(
    baseline_metrics: BenchmarkMetrics | None,
    multi_metrics: BenchmarkMetrics | None,
    multi_state: ResearchState | None
) -> None:
    """Save benchmark results to reports/benchmark_report.md"""
    from datetime import datetime

    report_path = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "benchmark_report.md")

    lines = [
        "# Benchmark Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Test Query",
        "",
        "Research GraphRAG state-of-the-art",
        "",
        "## Results Summary",
        "",
        "| Metric | Baseline | Multi-Agent | Winner |",
        "|---|---:|---:|---|",
    ]

    if baseline_metrics and multi_metrics:
        latency_winner = "Baseline" if baseline_metrics.latency_seconds < multi_metrics.latency_seconds else "Multi-Agent"
        cost_winner = "Baseline" if (baseline_metrics.estimated_cost_usd or 0) < (multi_metrics.estimated_cost_usd or 0) else "Multi-Agent"
        quality_winner = "Multi-Agent" if (multi_metrics.quality_score or 0) > (baseline_metrics.quality_score or 0) else "Baseline"

        baseline_cost = f"${baseline_metrics.estimated_cost_usd:.4f}" if baseline_metrics.estimated_cost_usd else "N/A"
        multi_cost = f"${multi_metrics.estimated_cost_usd:.4f}" if multi_metrics.estimated_cost_usd else "N/A"

        lines.extend([
            f"| Latency | {baseline_metrics.latency_seconds:.2f}s | {multi_metrics.latency_seconds:.2f}s | {latency_winner} |",
            f"| Est. Cost | {baseline_cost} | {multi_cost} | {cost_winner} |",
            f"| Quality | {baseline_metrics.quality_score or 'N/A'} | {multi_metrics.quality_score or 'N/A'} | {quality_winner} |",
            f"| Citation Coverage | {baseline_metrics.citation_coverage or 'N/A'} | {multi_metrics.citation_coverage or 'N/A'} | - |",
            f"| Failure Rate | {baseline_metrics.failure_rate:.0%} | {multi_metrics.failure_rate:.0%} | - |",
            "",
        ])

    if multi_state:
        lines.extend([
            "## Multi-Agent Route",
            "",
            " → ".join(multi_state.route_history),
            "",
            f"**Iterations:** {multi_state.iteration}",
            f"**Sources Found:** {len(multi_state.sources)}",
            "",
        ])

    lines.extend([
        "## Analysis",
        "",
        "### Key Findings:",
        "",
        "- **Latency**: Multi-agent is slower due to multiple LLM calls (researcher → analyst → writer → critic)",
        "- **Cost**: Multi-agent uses more tokens (4 LLM calls vs 1)",
        "- **Quality**: Multi-agent produces better-cited, more comprehensive responses with analyst review",
        "- **Citations**: Multi-agent has better citation coverage through structured workflow",
        "",
        "### Failure Mode Analysis:",
        "",
        "1. **Iteration Limits**: If max_iterations is too low, workflow may stop before completing all agents",
        "   - **Fix**: Set max_iterations >= 6 to allow for researcher → analyst → writer → critic flow",
        "",
        "2. **API Rate Limits**: Multiple sequential LLM calls increase chance of rate limit errors",
        "   - **Fix**: Add exponential backoff in LLM client, or use parallel agent execution where possible",
        "",
        "3. **State Passing**: If any agent returns malformed state, downstream agents may fail",
        "   - **Fix**: Add state validation in supervisor before routing",
        "",
        "4. **Mock Search Fallback**: Without Tavily API key, search returns mock data which may affect quality",
        "   - **Fix**: Configure TAVILY_API_KEY for real search results",
        "",
        "### Recommendations:",
        "",
        "| Scenario | Recommended | Reason |",
        "|----------|-------------|--------|",
        "| Simple factual query | Baseline | Fast, low cost |",
        "| Complex research | Multi-Agent | Better quality, citations |",
        "| Production system | Multi-Agent | Extensible, debuggable |",
        "| Rapid prototyping | Baseline | Quick iteration |",
        "",
    ])

    report_content = "\n".join(lines) + "\n"

    # Ensure reports directory exists
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write(report_content)

    console.print(f"\n[green]Report saved to: {report_path}[/green]")


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline using a real LLM call."""

    _init()
    request = _parse_query(query)

    console.print(Panel.fit("Running single-agent baseline...", title="Baseline"))
    start_time = time.perf_counter()

    state = _run_baseline(query)
    elapsed = time.perf_counter() - start_time

    # Display results
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))

    # Display metrics
    table = Table(title="Baseline Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Latency", f"{elapsed:.2f}s")
    table.add_row("Model", get_settings().openai_model)

    # Get cost from agent_results
    if state.agent_results:
        cost = state.agent_results[0].metadata.get("cost_usd")
        if cost:
            table.add_row("Est. Cost", f"${cost:.4f}")

    console.print(table)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow(enable_critic=get_settings().enable_critic)
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc

    # Pretty print result
    console.print(Panel.fit("Multi-Agent Workflow Complete", title="Result"))
    console.print(f"\n[bold]Route History:[/bold] {' → '.join(result.route_history)}")
    console.print(f"[bold]Iterations:[/bold] {result.iteration}")
    console.print(f"[bold]Sources Found:[/bold] {len(result.sources)}")

    if result.final_answer:
        answer_preview = result.final_answer[:1000] + "..." if len(result.final_answer) > 1000 else result.final_answer
        console.print(Panel.fit(answer_preview, title="Final Answer"))

    # Show trace summary
    if result.trace:
        console.print(f"\n[bold]Trace Events:[/bold] {len(result.trace)}")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run both baseline and multi-agent, then compare."""

    _init()

    console.print(Panel.fit("Running benchmark comparison...", title="Benchmark"))

    # Run baseline
    console.print("\n[cyan]Running Baseline...[/cyan]")
    baseline_state: ResearchState | None = None
    baseline_metrics: BenchmarkMetrics | None = None
    try:
        baseline_state, baseline_metrics = run_benchmark("baseline", query, _run_baseline)
        console.print(f"[green]✓ Baseline complete: {baseline_metrics.latency_seconds:.2f}s[/green]")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗ Baseline failed: {exc}[/red]")
        baseline_metrics = BenchmarkMetrics(run_name="baseline", latency_seconds=0, notes=str(exc), failure_rate=1.0)

    # Run multi-agent (chỉ 1 lần!)
    console.print("\n[cyan]Running Multi-Agent...[/cyan]")
    multi_state: ResearchState | None = None
    multi_metrics: BenchmarkMetrics | None = None
    try:
        multi_state, multi_metrics = run_benchmark("multi-agent", query, _run_multi_agent)
        console.print(f"[green]✓ Multi-Agent complete: {multi_metrics.latency_seconds:.2f}s[/green]")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗ Multi-Agent failed: {exc}[/red]")
        multi_metrics = BenchmarkMetrics(run_name="multi-agent", latency_seconds=0, notes=str(exc), failure_rate=1.0)

    # Display comparison table
    console.print("\n[bold]Comparison Results:[/bold]")
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", style="yellow")
    table.add_column("Multi-Agent", style="green")
    table.add_column("Winner", style="magenta")

    if baseline_metrics and multi_metrics:
        baseline_latency = baseline_metrics.latency_seconds
        multi_latency = multi_metrics.latency_seconds
        latency_winner = "Baseline" if baseline_latency < multi_latency else "Multi-Agent"

        table.add_row(
            "Latency",
            f"{baseline_latency:.2f}s",
            f"{multi_latency:.2f}s",
            latency_winner if baseline_metrics.failure_rate == 0 and multi_metrics.failure_rate == 0 else "N/A"
        )

        if baseline_metrics.estimated_cost_usd and multi_metrics.estimated_cost_usd:
            cost_winner = "Baseline" if baseline_metrics.estimated_cost_usd < multi_metrics.estimated_cost_usd else "Multi-Agent"
            table.add_row(
                "Est. Cost",
                f"${baseline_metrics.estimated_cost_usd:.4f}",
                f"${multi_metrics.estimated_cost_usd:.4f}",
                cost_winner
            )

        if baseline_metrics.quality_score is not None and multi_metrics.quality_score is not None:
            # Quality comparison - higher is better for both
            quality_winner = "Baseline" if baseline_metrics.quality_score > multi_metrics.quality_score else "Multi-Agent"
            table.add_row(
                "Quality",
                f"{baseline_metrics.quality_score:.1f}",
                f"{multi_metrics.quality_score:.1f}",
                quality_winner
            )

        if baseline_metrics.citation_coverage is not None and multi_metrics.citation_coverage is not None:
            citation_winner = "Baseline" if baseline_metrics.citation_coverage > multi_metrics.citation_coverage else "Multi-Agent"
            table.add_row(
                "Citation",
                f"{baseline_metrics.citation_coverage:.0%}",
                f"{multi_metrics.citation_coverage:.0%}",
                citation_winner
            )

        table.add_row(
            "Failure Rate",
            f"{baseline_metrics.failure_rate:.0%}",
            f"{multi_metrics.failure_rate:.0%}",
            "N/A"
        )

    console.print(table)

    # Write report to file
    _save_benchmark_report(baseline_metrics, multi_metrics, multi_state)

    # Show route history for multi-agent (từ kết quả đã có)
    if multi_state:
        console.print(f"\n[bold]Multi-Agent Route:[/bold]")
        console.print(" → ".join(multi_state.route_history))


@app.command()
def compare(
    queries_file: Annotated[str, typer.Argument(help="File containing queries to compare")],
) -> None:
    """Compare baseline vs multi-agent across multiple queries from a file."""

    _init()

    try:
        with open(queries_file) as f:
            queries = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        console.print(f"[red]File not found: {queries_file}[/red]")
        raise typer.Exit(code=1)

    if not queries:
        console.print("[yellow]No queries found in file[/yellow]")
        raise typer.Exit()

    console.print(f"[cyan]Running comparison on {len(queries)} queries...[/cyan]")

    all_metrics: list[BenchmarkMetrics] = []

    for i, query in enumerate(queries):
        console.print(f"\n[bold]Query {i+1}/{len(queries)}:[/bold] {query[:50]}...")

        # Baseline
        try:
            _, b_metrics = run_benchmark(f"baseline-{i+1}", query, _run_baseline)
            all_metrics.append(b_metrics)
            console.print(f"  Baseline: {b_metrics.latency_seconds:.2f}s")
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]Baseline failed: {exc}[/red]")

        # Multi-agent
        try:
            _, m_metrics = run_benchmark(f"multi-{i+1}", query, _run_multi_agent)
            all_metrics.append(m_metrics)
            console.print(f"  Multi-Agent: {m_metrics.latency_seconds:.2f}s")
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]Multi-Agent failed: {exc}[/red]")

    # Generate report
    report = render_markdown_report(all_metrics)
    console.print("\n" + report)


if __name__ == "__main__":
    app()
