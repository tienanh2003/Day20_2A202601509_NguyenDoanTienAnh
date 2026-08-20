#!/usr/bin/env python3
"""Run benchmark using offline research corpus."""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.services.llm_client import LLMClient

BASELINE_PROMPT = """You are a research assistant. Based ONLY on the provided corpus data, write a comprehensive research report.
Include citations using [source_id] format. Be thorough and accurate."""


def load_corpus_topic(corpus_dir: Path, topic_num: int) -> dict:
    """Load a single topic from the corpus."""
    manifest_path = corpus_dir / "manifest.csv"
    topics_dir = corpus_dir / "topics"

    # Read manifest to get filename
    with open(manifest_path) as f:
        lines = f.readlines()[1:]  # Skip header

    for line in lines:
        parts = line.strip().split(",")
        if int(parts[0]) == topic_num:
            filename = parts[2]
            filepath = topics_dir / filename
            if filepath.exists():
                with open(filepath) as f:
                    return json.load(f)

    raise ValueError(f"Topic {topic_num} not found")


def run_baseline_with_corpus(query: str, corpus_data: dict) -> ResearchState:
    """Run baseline using corpus as knowledge base."""
    state = ResearchState(request=ResearchQuery(query=query))

    # Extract key info from corpus
    topic_title = corpus_data.get("topic_title", query)
    articles = corpus_data.get("articles", [])
    sources = corpus_data.get("sources", [])

    # Build context from corpus
    context_parts = [f"Topic: {topic_title}\n"]
    context_parts.append(f"Research Task: {corpus_data.get('research_task', query)}\n")

    # Add articles
    if articles:
        context_parts.append("\n## Knowledge Articles:\n")
        for i, article in enumerate(articles[:5], 1):
            context_parts.append(f"\n### Article {i}: {article.get('title', 'Untitled')}\n")
            context_parts.append(f"{article.get('content', '')}\n")

    # Add sources
    if sources:
        context_parts.append("\n## Sources:\n")
        for src in sources[:10]:
            context_parts.append(f"- [{src.get('source_id', 'unknown')}] {src.get('title', 'Untitled')}: {src.get('content', '')[:200]}...\n")

    corpus_context = "".join(context_parts)

    # Create prompt with corpus
    prompt = f"""Based ONLY on the corpus data provided below, write a comprehensive research report answering:

{query}

CORPUS DATA:
{corpus_context[:15000]}

Write a detailed report with proper structure and cite sources using [source_id] format."""

    llm = LLMClient(temperature=0.0)
    response = llm.complete(
        system_prompt=BASELINE_PROMPT,
        user_prompt=prompt,
    )

    state.final_answer = response.content

    # Record cost
    if response.cost_usd:
        from multi_agent_research_lab.core.schemas import AgentResult, AgentName
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "cost_usd": response.cost_usd,
                    "tokens": response.output_tokens,
                },
            )
        )

    return state


def run_multi_agent_with_corpus(query: str, corpus_data: dict) -> ResearchState:
    """Run multi-agent using corpus as knowledge base."""
    from multi_agent_research_lab.agents.researcher import ResearcherAgent
    from multi_agent_research_lab.agents.analyst import AnalystAgent
    from multi_agent_research_lab.agents.writer import WriterAgent
    from multi_agent_research_lab.agents.critic import CriticAgent
    from multi_agent_research_lab.core.schemas import AgentName

    state = ResearchState(request=ResearchQuery(query=query))

    # Extract corpus info
    articles = corpus_data.get("articles", [])
    sources = corpus_data.get("sources", [])

    # Convert corpus sources to our format
    from multi_agent_research_lab.core.schemas import SourceDocument
    corpus_sources = []
    for src in sources[:10]:
        corpus_sources.append(SourceDocument(
            title=src.get("title", "Untitled"),
            url=src.get("url"),
            snippet=src.get("content", "")[:500],
            metadata={"source_id": src.get("source_id"), "is_synthetic": src.get("is_synthetic", False)},
        ))

    state.sources = corpus_sources

    # Build research notes from corpus
    research_parts = [f"Research Query: {query}\n\n"]
    research_parts.append(f"Topic: {corpus_data.get('topic_title', query)}\n\n")

    if articles:
        research_parts.append("## Key Information:\n")
        for article in articles[:5]:
            research_parts.append(f"\n### {article.get('title', 'Untitled')}\n")
            research_parts.append(f"{article.get('content', '')}\n")

    research_parts.append("\n## Sources with IDs:\n")
    for src in corpus_sources:
        sid = src.metadata.get("source_id", "unknown")
        research_parts.append(f"- [{sid}] {src.title}\n")

    state.research_notes = "".join(research_parts)[:10000]

    # Run analyst
    analyst = AnalystAgent()
    state = analyst.run(state)

    # Run writer
    writer = WriterAgent()
    state = writer.run(state)

    # Run critic
    critic = CriticAgent()
    state = critic.run(state)

    state.record_route("done")

    return state


def main():
    corpus_dir = Path(__file__).parent.parent / "ai_agent_offline_research_corpus_v2"

    if not corpus_dir.exists():
        print(f"Error: Corpus not found at {corpus_dir}")
        sys.exit(1)

    # Run on first 3 topics for quick benchmark
    topics = [1, 2, 3]

    results = {
        "baseline": [],
        "multi_agent": [],
    }

    print("=" * 60)
    print("Running Benchmark with Offline Research Corpus")
    print("=" * 60)

    for topic_num in topics:
        print(f"\n--- Topic {topic_num} ---")

        corpus_data = load_corpus_topic(corpus_dir, topic_num)
        query = corpus_data.get("research_task", {}).get("question", "Research this topic")
        topic_title = corpus_data.get("topic_title", f"Topic {topic_num}")

        print(f"Query: {topic_title}")

        # Baseline
        print("Running baseline...")
        start = time.perf_counter()
        baseline_state, baseline_metrics = run_benchmark(
            f"baseline-{topic_num}",
            query,
            lambda q: run_baseline_with_corpus(q, corpus_data)
        )
        baseline_time = time.perf_counter() - start
        baseline_metrics.latency_seconds = baseline_time
        results["baseline"].append(baseline_metrics)
        print(f"  Baseline: {baseline_time:.2f}s, Quality: {baseline_metrics.quality_score}")

        # Multi-agent
        print("Running multi-agent...")
        start = time.perf_counter()
        multi_state, multi_metrics = run_benchmark(
            f"multi-{topic_num}",
            query,
            lambda q: run_multi_agent_with_corpus(q, corpus_data)
        )
        multi_time = time.perf_counter() - start
        multi_metrics.latency_seconds = multi_time
        results["multi_agent"].append(multi_metrics)
        print(f"  Multi-Agent: {multi_time:.2f}s, Quality: {multi_metrics.quality_score}")

    # Generate report
    print("\n" + "=" * 60)
    print("Generating Report...")
    print("=" * 60)

    all_metrics = results["baseline"] + results["multi_agent"]

    report_lines = [
        "# Benchmark Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Test Dataset",
        "",
        "Using **AI Agent Offline Research Corpus v2** - 30 topics with embedded knowledge.",
        f"Tested on topics: {topics}",
        "",
        "## Results Summary",
        "",
        "| Metric | Baseline (avg) | Multi-Agent (avg) | Winner |",
        "|---|---:|---:|---|",
    ]

    # Calculate averages
    baseline_avg_latency = sum(m.latency_seconds for m in results["baseline"]) / len(results["baseline"])
    multi_avg_latency = sum(m.latency_seconds for m in results["multi_agent"]) / len(results["multi_agent"])

    baseline_avg_cost = sum(m.estimated_cost_usd or 0 for m in results["baseline"]) / len(results["baseline"])
    multi_avg_cost = sum(m.estimated_cost_usd or 0 for m in results["multi_agent"]) / len(results["multi_agent"])

    baseline_avg_quality = sum(m.quality_score or 0 for m in results["baseline"]) / len(results["baseline"])
    multi_avg_quality = sum(m.quality_score or 0 for m in results["multi_agent"]) / len(results["multi_agent"])

    latency_winner = "Baseline" if baseline_avg_latency < multi_avg_latency else "Multi-Agent"
    cost_winner = "Baseline" if baseline_avg_cost < multi_avg_cost else "Multi-Agent"
    quality_winner = "Multi-Agent" if multi_avg_quality > baseline_avg_quality else "Baseline"

    report_lines.extend([
        f"| Latency | {baseline_avg_latency:.2f}s | {multi_avg_latency:.2f}s | {latency_winner} |",
        f"| Est. Cost | ${baseline_avg_cost:.4f} | ${multi_avg_cost:.4f} | {cost_winner} |",
        f"| Quality | {baseline_avg_quality:.1f} | {multi_avg_quality:.1f} | {quality_winner} |",
        "",
        "## Detailed Results",
        "",
        "| Run | Latency | Cost | Quality |",
        "|---|---:|---:|---:|",
    ])

    for m in all_metrics:
        cost = f"${m.estimated_cost_usd:.4f}" if m.estimated_cost_usd else "N/A"
        quality = f"{m.quality_score:.1f}" if m.quality_score else "N/A"
        report_lines.append(f"| {m.run_name} | {m.latency_seconds:.2f}s | {cost} | {quality} |")

    report_lines.extend([
        "",
        "## Analysis",
        "",
        "### Key Findings:",
        "",
        f"- **Latency**: Multi-agent is {multi_avg_latency / baseline_avg_latency:.1f}x slower than baseline",
        f"- **Cost**: Multi-agent costs {multi_avg_cost / max(baseline_avg_cost, 0.0001):.1f}x more than baseline",
        f"- **Quality**: Multi-agent scores {multi_avg_quality - baseline_avg_quality:.1f} points higher",
        "",
        "### Failure Mode Analysis:",
        "",
        "1. **Iteration Limits**: If max_iterations is too low, workflow may stop before completing",
        "   - **Fix**: Set max_iterations >= 5 for researcher → analyst → writer → critic flow",
        "",
        "2. **Corpus Size**: Large corpus data may exceed context window",
        "   - **Fix**: Truncate corpus context or use smarter retrieval",
        "",
        "3. **Citation Discipline**: Writer may not always cite sources correctly",
        "   - **Fix**: Add citation validation in writer prompt",
        "",
        "4. **LLM Hallucination**: Analyst may misinterpret corpus facts",
        "   - **Fix**: Critic agent helps catch hallucinations",
        "",
        "### Recommendations:",
        "",
        "| Scenario | Recommended | Reason |",
        "|----------|-------------|--------|",
        "| Simple factual query | Baseline | Fast, low cost |",
        "| Complex research with citations | Multi-Agent | Better quality |",
        "| Production with observability | Multi-Agent | Full trace |",
        "| Large corpus queries | Baseline | Simpler, faster |",
        "",
    ])

    # Write report
    report_path = Path(__file__).parent.parent / "reports" / "benchmark_report.md"
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline Avg: {baseline_avg_latency:.2f}s, ${baseline_avg_cost:.4f}, Quality: {baseline_avg_quality:.1f}")
    print(f"Multi-Agent Avg: {multi_avg_latency:.2f}s, ${multi_avg_cost:.4f}, Quality: {multi_avg_quality:.1f}")
    print(f"Winners: Latency={latency_winner}, Cost={cost_winner}, Quality={quality_winner}")


if __name__ == "__main__":
    main()
