"""LangGraph workflow skeleton."""

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


def _route_decision(state: ResearchState) -> Literal[
    "researcher", "analyst", "writer", "critic", "done"
]:
    """Route based on current state after supervisor decision."""
    history = state.route_history
    if not history:
        return "done"

    last_route = history[-1]
    if last_route == "done":
        return "done"

    # Map AgentName to node names
    mapping: dict[str, str] = {
        AgentName.RESEARCHER: "researcher",
        AgentName.ANALYST: "analyst",
        AgentName.WRITER: "writer",
        AgentName.CRITIC: "critic",
        "critic": "critic",
    }
    return mapping.get(last_route, last_route)


def _build_graph() -> StateGraph:
    """Build the LangGraph state machine."""
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("supervisor", _supervisor_node)
    graph.add_node("researcher", _researcher_node)
    graph.add_node("analyst", _analyst_node)
    graph.add_node("writer", _writer_node)
    graph.add_node("critic", _critic_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Conditional edges from supervisor
    graph.add_conditional_edges(
        "supervisor",
        _route_decision,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "critic": "critic",
            "done": END,
        },
    )

    # All worker nodes return to supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("critic", "supervisor")

    return graph


def _supervisor_node(state: ResearchState) -> dict[str, Any]:
    agent = SupervisorAgent()
    return agent.run(state).model_dump()


def _researcher_node(state: ResearchState) -> dict[str, Any]:
    agent = ResearcherAgent()
    return agent.run(state).model_dump()


def _analyst_node(state: ResearchState) -> dict[str, Any]:
    agent = AnalystAgent()
    return agent.run(state).model_dump()


def _writer_node(state: ResearchState) -> dict[str, Any]:
    agent = WriterAgent()
    return agent.run(state).model_dump()


def _critic_node(state: ResearchState) -> dict[str, Any]:
    agent = CriticAgent()
    return agent.run(state).model_dump()


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self, enable_critic: bool = True):
        self._graph: StateGraph[ResearchState] | None = None
        self.enable_critic = enable_critic

    def build(self) -> StateGraph[ResearchState]:
        """Create a LangGraph graph."""
        self._graph = _build_graph()
        return self._graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        with trace_span("workflow.run", {"query": state.request.query[:50]}):
            if self._graph is None:
                self.build()

            compiled = self._graph.compile()  # type: ignore[union-attr]

            # Run with trace
            with trace_span("graph.invoke"):
                result = compiled.invoke(state)

            return ResearchState(**result)
