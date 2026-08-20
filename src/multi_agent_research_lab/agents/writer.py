"""Writer agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """You are a professional technical writer. Your job is to:
1. Synthesize research and analysis into a clear, coherent response
2. Include proper citations referencing the sources
3. Structure the content with appropriate headings
4. Make complex topics accessible to the target audience
5. Balance depth with readability

Always cite sources using [1], [2], etc. referencing the source list."""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self.llm = LLMClient(temperature=0.4)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        if not state.research_notes:
            logger.warning("Writer received empty research_notes")
            state.final_answer = "Insufficient data to generate final answer."
            return state

        logger.info("Writer starting synthesis")

        with trace_span("writer.synthesize") as span:
            source_text = self._format_sources_with_citations(state)
            response = self.llm.complete(
                system_prompt=WRITER_SYSTEM_PROMPT,
                user_prompt=self._build_writing_prompt(state, source_text),
            )
            state.final_answer = response.content
            span["token_count"] = response.output_tokens or 0

        # Record agent result
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "citations_count": state.final_answer.count("[") // 2,
                },
            )
        )

        state.add_trace_event(
            "writer",
            {
                "answer_length": len(state.final_answer),
                "citations_count": state.final_answer.count("[") // 2,
            },
        )

        logger.info(f"Writer completed: {len(state.final_answer)} chars")
        return state

    def _format_sources_with_citations(self, state: ResearchState) -> str:
        """Format sources with citation indices."""
        lines = ["References:\n"]
        for i, src in enumerate(state.sources, 1):
            lines.append(f"[{i}] {src.title}")
            if src.url:
                lines.append(f"    URL: {src.url}")
            if src.snippet:
                lines.append(f"    Content: {src.snippet[:200]}...")
            lines.append("")
        return "\n".join(lines)

    def _build_writing_prompt(self, state: ResearchState, source_text: str) -> str:
        """Build prompt for final answer generation."""
        audience = state.request.audience

        return f"""Query: {state.request.query}
Target Audience: {audience}

Research Notes:
{state.research_notes}

Analysis Notes:
{state.analysis_notes or "No analysis notes available."}

{source_text}

Task: Write a comprehensive response that:
1. Directly addresses the query
2. Synthesizes key findings from research and analysis
3. Includes proper citations [1], [2], etc. referencing the sources above
4. Is well-structured with clear headings
5. Is appropriate for: {audience}
6. Is concise but thorough (aim for 300-600 words)

Format the response with:
- A brief introduction
- Main body with logical sections
- A conclusion or summary
- Citations in brackets [n] throughout

Begin your response now:
"""
