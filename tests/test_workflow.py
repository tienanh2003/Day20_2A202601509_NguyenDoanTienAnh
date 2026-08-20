"""Tests for workflow module."""

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_workflow_builds() -> None:
    """MultiAgentWorkflow should build without errors."""
    workflow = MultiAgentWorkflow()
    graph = workflow.build()

    assert graph is not None


def test_workflow_run_returns_state() -> None:
    """MultiAgentWorkflow.run should return ResearchState."""
    workflow = MultiAgentWorkflow()
    state = ResearchState(request=ResearchQuery(query="Test query"))

    result = workflow.run(state)

    assert isinstance(result, ResearchState)
    assert result.request.query == "Test query"


def test_workflow_records_route_history() -> None:
    """Workflow should record route history."""
    workflow = MultiAgentWorkflow()
    state = ResearchState(request=ResearchQuery(query="Test"))

    result = workflow.run(state)

    assert len(result.route_history) > 0
    assert "supervisor" in result.route_history or result.route_history[0] in ["researcher", "done"]
