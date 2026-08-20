"""Analyst agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """You are an analyst agent. Your job is to:
1. Analyze research notes and source materials
2. Compare and evaluate different viewpoints or findings
3. Assess the reliability and credibility of sources
4. Extract key claims and evaluate supporting evidence
5. Identify gaps, contradictions, or areas needing more research

Be critical but fair. Clearly distinguish between strong and weak evidence."""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self.llm = LLMClient(temperature=0.1)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        if not state.research_notes:
            logger.warning("Analyst received empty research_notes")
            state.analysis_notes = "No research notes available for analysis."
            return state

        logger.info("Analyst starting analysis")

        with trace_span("analyst.analyze") as span:
            source_text = self._format_sources(state.sources)
            response = self.llm.complete(
                system_prompt=ANALYST_SYSTEM_PROMPT,
                user_prompt=self._build_analysis_prompt(state, source_text),
            )
            state.analysis_notes = response.content
            span["token_count"] = response.output_tokens or 0

        # Record agent result
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )

        state.add_trace_event(
            "analyst",
            {
                "analysis_length": len(state.analysis_notes),
                "sources_analyzed": len(state.sources),
            },
        )

        logger.info(f"Analyst completed: {len(state.analysis_notes)} chars")
        return state

    def _format_sources(self, sources: list) -> str:
        """Format sources for analysis context."""
        if not sources:
            return "No sources available."

        lines = []
        for i, src in enumerate(sources, 1):
            lines.append(f"[{i}] {src.title}")
            if src.url:
                lines.append(f"    URL: {src.url}")
            lines.append(f"    Snippet: {src.snippet[:300]}")
            metadata = src.metadata or {}
            if "score" in metadata:
                lines.append(f"    Relevance score: {metadata['score']:.2f}")
            lines.append("")
        return "\n".join(lines)

    def _build_analysis_prompt(self, state: ResearchState, source_text: str) -> str:
        """Build prompt for analysis."""
        return f"""Original Query: {state.request.query}

Research Notes:
{state.research_notes}

Sources:
{source_text}

Task: Perform a thorough analysis that:
1. **Key Claims**: Extract and list the main claims or findings
2. **Evidence Assessment**: Evaluate the strength of evidence for each claim
3. **Source Credibility**: Rate the credibility of sources (high/medium/low) and explain why
4. **Viewpoint Comparison**: Compare different perspectives or findings
5. **Gaps & Limitations**: Identify what might be missing or needs further research
6. **Confidence Level**: Provide an overall confidence assessment (high/medium/low)

Format your analysis with clear headers for each section.
"""
