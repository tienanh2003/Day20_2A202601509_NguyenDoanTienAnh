"""Researcher agent skeleton."""

import logging
from typing import Any

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

RESEARCHER_SYSTEM_PROMPT = """You are a research agent. Your job is to:
1. Search for relevant information on the given topic
2. Compile concise research notes summarizing key findings
3. Cite your sources with titles and URLs

Be thorough but focus on high-quality, authoritative sources."""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self.llm = LLMClient(temperature=0.2)
        self.search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources

        logger.info(f"Researcher starting search for: {query[:80]}")

        with trace_span("researcher.search") as search_span:
            sources = self.search.search(query, max_results=max_sources)
            state.sources = sources
            search_span["result_count"] = len(sources)

        # Generate research notes using LLM
        with trace_span("researcher.notes_generation") as notes_span:
            source_text = self._format_sources(sources)
            notes = self.llm.complete(
                system_prompt=RESEARCHER_SYSTEM_PROMPT,
                user_prompt=self._build_notes_prompt(query, source_text),
            )
            state.research_notes = notes.content
            notes_span["token_count"] = notes.output_tokens or 0

        # Record agent result
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes or "",
                metadata={
                    "sources_count": len(sources),
                    "input_tokens": notes.input_tokens,
                    "output_tokens": notes.output_tokens,
                    "cost_usd": notes.cost_usd,
                },
            )
        )

        state.add_trace_event(
            "researcher",
            {
                "sources_found": len(sources),
                "notes_length": len(state.research_notes or ""),
            },
        )

        logger.info(
            f"Researcher completed: {len(sources)} sources, "
            f"{len(state.research_notes or '')} chars in notes"
        )
        return state

    def _format_sources(self, sources: list) -> str:
        """Format sources for LLM context."""
        lines = []
        for i, src in enumerate(sources, 1):
            lines.append(f"[{i}] {src.title}")
            if src.url:
                lines.append(f"    URL: {src.url}")
            lines.append(f"    Snippet: {src.snippet[:200]}")
            lines.append("")
        return "\n".join(lines)

    def _build_notes_prompt(self, query: str, source_text: str) -> str:
        """Build prompt for notes generation."""
        return f"""Topic: {query}

Sources found:
{source_text}

Task: Based on the above sources, create comprehensive research notes that:
1. Summarize the key findings and concepts
2. Identify main themes and trends
3. Note any conflicting information or gaps
4. Highlight the most important points

Format the notes clearly with headers and bullet points where appropriate.
"""
