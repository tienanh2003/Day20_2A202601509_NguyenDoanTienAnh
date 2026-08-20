"""Optional critic agent skeleton for bonus work."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are a critical review agent. Your job is to:
1. Fact-check claims in the final answer against sources
2. Evaluate citation coverage and accuracy
3. Identify potential hallucinations or unsupported claims
4. Assess answer completeness and relevance to query
5. Provide actionable feedback for improvement

Be thorough but constructive. Flag issues clearly with severity levels."""


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self) -> None:
        self.llm = LLMClient(temperature=0.1)

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        if not state.final_answer:
            logger.warning("Critic received empty final_answer")
            state.add_trace_event("critic", {"error": "No final answer to review"})
            return state

        logger.info("Critic starting review")

        with trace_span("critic.review") as span:
            source_text = self._format_sources(state.sources)
            response = self.llm.complete(
                system_prompt=CRITIC_SYSTEM_PROMPT,
                user_prompt=self._build_critique_prompt(state, source_text),
            )
            span["token_count"] = response.output_tokens or 0

        # Record agent result
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )

        state.add_trace_event(
            "critic",
            {
                "review_length": len(response.content),
                "has_final_answer": bool(state.final_answer),
            },
        )

        logger.info(f"Critic completed: {len(response.content)} chars")
        return state

    def _format_sources(self, sources: list) -> str:
        """Format sources for critique context."""
        if not sources:
            return "No sources available."

        lines = []
        for i, src in enumerate(sources, 1):
            lines.append(f"[{i}] {src.title}")
            if src.url:
                lines.append(f"    URL: {src.url}")
            lines.append(f"    Content: {src.snippet[:300]}")
            lines.append("")
        return "\n".join(lines)

    def _build_critique_prompt(self, state: ResearchState, source_text: str) -> str:
        """Build prompt for critique."""
        return f"""Original Query: {state.request.query}

Final Answer to Review:
{state.final_answer}

Sources:
{source_text}

Task: Conduct a thorough critique of the final answer:

1. **Fact-Check**: Are the claims supported by the sources? Flag any unsupported claims.
2. **Citation Coverage**: Are all key claims properly cited? Note missing citations.
3. **Hallucination Detection**: Are there any potentially fabricated facts or references?
4. **Completeness**: Does the answer fully address the query? Note gaps.
5. **Quality Assessment**: Rate overall quality (high/medium/low) with reasoning.
6. **Recommended Improvements**: Suggest specific fixes.

Format your critique with clear headers for each section.
"""
