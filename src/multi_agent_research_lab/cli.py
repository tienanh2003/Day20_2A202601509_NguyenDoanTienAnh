"""Command-line entrypoint for the lab starter."""

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
    return state


def _run_multi_agent(query: str) -> ResearchState:
    """Helper to run multi-agent for benchmarking."""
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


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
    console.print(table)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
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
        console.print(Panel.fit(result.final_answer[:1000] + "..." if len(result.final_answer) > 1000 else result.final_answer, title="Final Answer"))

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
    try:
        baseline_state, baseline_metrics = run_benchmark("baseline", query, _run_baseline)
        console.print(f"[green]✓ Baseline complete: {baseline_metrics.latency_seconds:.2f}s[/green]")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗ Baseline failed: {exc}[/red]")
        baseline_metrics = BenchmarkMetrics(run_name="baseline", latency_seconds=0, notes=str(exc), failure_rate=1.0)

    # Run multi-agent
    console.print("\n[cyan]Running Multi-Agent...[/cyan]")
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

    if baseline_metrics.quality_score and multi_metrics.quality_score:
        quality_winner = "Baseline" if baseline_metrics.quality_score > multi_metrics.quality_score else "Multi-Agent"
        table.add_row(
            "Quality",
            f"{baseline_metrics.quality_score:.1f}",
            f"{multi_metrics.quality_score:.1f}",
            quality_winner
        )

    table.add_row(
        "Failure Rate",
        f"{baseline_metrics.failure_rate:.0%}",
        f"{multi_metrics.failure_rate:.0%}",
        "N/A"
    )

    console.print(table)

    # Show route history for multi-agent
    try:
        multi_state = _run_multi_agent(query)
        console.print(f"\n[bold]Multi-Agent Route:[/bold]")
        console.print(" → ".join(multi_state.route_history))
    except Exception:  # noqa: BLE001
        pass


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
