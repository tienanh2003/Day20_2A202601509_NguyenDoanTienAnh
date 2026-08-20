"""Supervisor / router skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing logic:
    1. If no sources → route to researcher
    2. If no analysis_notes → route to analyst (after researcher)
    3. If no final_answer → route to writer
    4. If critic enabled and has answer → route to critic
    5. Otherwise → done
    """

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        settings = get_settings()
        max_iter = settings.max_iterations
        enable_critic = getattr(settings, "enable_critic", True)

        # Enforce max iterations
        if state.iteration >= max_iter:
            logger.warning(
                f"Max iterations ({max_iter}) reached. Forcing stop."
            )
            state.record_route("done")
            state.add_trace_event(
                "supervisor",
                {"action": "stop", "reason": "max_iterations", "iteration": state.iteration},
            )
            return state

        # Routing policy
        if not state.sources:
            next_route = AgentName.RESEARCHER
        elif not state.research_notes:
            next_route = AgentName.RESEARCHER  # Researcher also writes notes
        elif not state.analysis_notes:
            next_route = AgentName.ANALYST
        elif not state.final_answer:
            next_route = AgentName.WRITER
        elif enable_critic and not self._has_critic_review(state):
            # Optional: route to critic for review
            next_route = AgentName.CRITIC
        else:
            next_route = "done"

        state.record_route(next_route)
        state.add_trace_event(
            "supervisor",
            {
                "action": "route",
                "next": next_route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_research_notes": bool(state.research_notes),
                "has_analysis_notes": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
            },
        )

        logger.info(
            f"[Iter {state.iteration}] Supervisor routing to: {next_route}"
        )
        return state

    def _has_critic_review(self, state: ResearchState) -> bool:
        """Check if critic has already reviewed."""
        for result in state.agent_results:
            if result.agent == AgentName.CRITIC:
                return True
        return False
