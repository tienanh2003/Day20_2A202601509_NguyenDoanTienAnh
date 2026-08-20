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


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline using a real LLM call."""

    _init()
    request = _parse_query(query)

    console.print(Panel.fit("Running single-agent baseline...", title="Baseline"))
    start_time = time.perf_counter()

    llm = LLMClient(temperature=0.0)
    response = llm.complete(
        system_prompt=BASELINE_SYSTEM_PROMPT,
        user_prompt=f"Research question: {request.query}\n\nProvide a comprehensive answer.",
    )

    elapsed = time.perf_counter() - start_time
    state = ResearchState(request=request)
    state.final_answer = response.content

    # Display results
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))

    # Display metrics
    table = Table(title="Baseline Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Latency", f"{elapsed:.2f}s")
    table.add_row("Model", response.model or get_settings().openai_model)
    table.add_row(
        "Input Tokens", str(response.input_tokens) if response.input_tokens else "N/A"
    )
    table.add_row(
        "Output Tokens",
        str(response.output_tokens) if response.output_tokens else "N/A",
    )
    table.add_row(
        "Estimated Cost", f"${response.cost_usd:.4f}" if response.cost_usd else "N/A"
    )
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
    console.print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
