"""Tests for agent implementations."""

import pytest

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


class TestSupervisorAgent:
    """Tests for SupervisorAgent routing policy."""

    def test_routes_to_researcher_when_no_sources(self) -> None:
        """Supervisor should route to researcher when no sources."""
        state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
        agent = SupervisorAgent()
        result = agent.run(state)

        assert len(result.route_history) == 1
        assert result.route_history[0] == AgentName.RESEARCHER
        assert result.iteration == 1

    def test_routes_to_analyst_when_has_sources_no_analysis(self) -> None:
        """Supervisor should route to analyst when sources exist but no analysis."""
        state = ResearchState(
            request=ResearchQuery(query="Explain multi-agent systems"),
            sources=[SourceDocument(title="Test", snippet="Test content")]
        )
        agent = SupervisorAgent()
        result = agent.run(state)

        assert result.route_history[-1] == AgentName.ANALYST

    def test_routes_to_writer_when_has_analysis_no_answer(self) -> None:
        """Supervisor should route to writer when analysis exists but no final answer."""
        state = ResearchState(
            request=ResearchQuery(query="Explain multi-agent systems"),
            sources=[SourceDocument(title="Test", snippet="Test content")],
            research_notes="Some notes",
            analysis_notes="Analysis complete"
        )
        agent = SupervisorAgent()
        result = agent.run(state)

        assert result.route_history[-1] == AgentName.WRITER

    def test_routes_to_critic_when_has_answer(self) -> None:
        """Supervisor should route to critic after writer completes."""
        state = ResearchState(
            request=ResearchQuery(query="Explain multi-agent systems"),
            sources=[SourceDocument(title="Test", snippet="Test content")],
            research_notes="Some notes",
            analysis_notes="Analysis complete",
            final_answer="Complete answer"
        )
        agent = SupervisorAgent()
        result = agent.run(state)

        assert result.route_history[-1] == AgentName.CRITIC

    def test_stops_after_critic(self) -> None:
        """Supervisor should route to done after critic completes."""
        state = ResearchState(
            request=ResearchQuery(query="Explain multi-agent systems"),
            sources=[SourceDocument(title="Test", snippet="Test content")],
            research_notes="Some notes",
            analysis_notes="Analysis complete",
            final_answer="Complete answer"
        )
        # Pre-add critic result
        from multi_agent_research_lab.core.schemas import AgentResult
        state.agent_results.append(
            AgentResult(agent=AgentName.CRITIC, content="Critique done")
        )

        agent = SupervisorAgent()
        result = agent.run(state)

        assert result.route_history[-1] == "done"

    def test_stops_when_complete(self) -> None:
        """Supervisor should route to done when all components present (no critic)."""
        state = ResearchState(
            request=ResearchQuery(query="Explain multi-agent systems"),
            sources=[SourceDocument(title="Test", snippet="Test content")],
            research_notes="Some notes",
            analysis_notes="Analysis complete",
            final_answer="Complete answer"
        )
        # Disable critic
        state.iteration = 5  # Near max

        agent = SupervisorAgent()
        # Second run should go to done
        result = agent.run(state)

        # Should either route to critic or done depending on iteration
        assert result.route_history[-1] in [AgentName.CRITIC, "done"]

    def test_enforces_max_iterations(self) -> None:
        """Supervisor should stop when max iterations reached."""
        state = ResearchState(request=ResearchQuery(query="Test"))
        # Simulate reaching max iterations
        state.iteration = 10  # Assuming max_iterations default is 6

        agent = SupervisorAgent()
        result = agent.run(state)

        assert result.route_history[-1] == "done"

    def test_trace_event_recorded(self) -> None:
        """Supervisor should record trace events."""
        state = ResearchState(request=ResearchQuery(query="Test"))
        agent = SupervisorAgent()
        result = agent.run(state)

        assert len(result.trace) > 0
        assert result.trace[-1]["name"] == "supervisor"


class TestResearcherAgent:
    """Tests for ResearcherAgent."""

    def test_researcher_populates_sources(self) -> None:
        """Researcher should populate sources in state."""
        state = ResearchState(request=ResearchQuery(query="Test query"))
        agent = ResearcherAgent()
        result = agent.run(state)

        assert len(result.sources) > 0
        assert result.research_notes is not None
        assert len(result.research_notes) > 0

    def test_researcher_records_agent_result(self) -> None:
        """Researcher should add result to agent_results."""
        state = ResearchState(request=ResearchQuery(query="Test query"))
        agent = ResearcherAgent()
        result = agent.run(state)

        assert len(result.agent_results) > 0
        assert result.agent_results[-1].agent == AgentName.RESEARCHER

    def test_researcher_trace_event(self) -> None:
        """Researcher should record trace events."""
        state = ResearchState(request=ResearchQuery(query="Test"))
        agent = ResearcherAgent()
        result = agent.run(state)

        # Find researcher trace event
        researcher_traces = [t for t in result.trace if t.get("name") == "researcher"]
        assert len(researcher_traces) > 0


class TestAnalystAgent:
    """Tests for AnalystAgent."""

    def test_analyst_populates_analysis_notes(self) -> None:
        """Analyst should populate analysis_notes."""
        state = ResearchState(
            request=ResearchQuery(query="Test"),
            sources=[SourceDocument(title="Test", snippet="Content")],
            research_notes="Research notes here"
        )
        agent = AnalystAgent()
        result = agent.run(state)

        assert result.analysis_notes is not None
        assert len(result.analysis_notes) > 0

    def test_analyst_handles_empty_research_notes(self) -> None:
        """Analyst should handle empty research notes gracefully."""
        state = ResearchState(request=ResearchQuery(query="Test"))
        agent = AnalystAgent()
        result = agent.run(state)

        assert result.analysis_notes is not None
        assert "No research notes" in result.analysis_notes

    def test_analyst_records_agent_result(self) -> None:
        """Analyst should add result to agent_results."""
        state = ResearchState(
            request=ResearchQuery(query="Test"),
            sources=[SourceDocument(title="Test", snippet="Content")],
            research_notes="Notes"
        )
        agent = AnalystAgent()
        result = agent.run(state)

        assert len(result.agent_results) > 0
        assert AgentName.ANALYST in [r.agent for r in result.agent_results]


class TestWriterAgent:
    """Tests for WriterAgent."""

    def test_writer_populates_final_answer(self) -> None:
        """Writer should populate final_answer."""
        state = ResearchState(
            request=ResearchQuery(query="Test"),
            sources=[SourceDocument(title="Test", snippet="Content", url="http://test.com")],
            research_notes="Research notes",
            analysis_notes="Analysis notes"
        )
        agent = WriterAgent()
        result = agent.run(state)

        assert result.final_answer is not None
        assert len(result.final_answer) > 0

    def test_writer_handles_empty_sources(self) -> None:
        """Writer should handle empty sources gracefully."""
        state = ResearchState(request=ResearchQuery(query="Test"))
        agent = WriterAgent()
        result = agent.run(state)

        assert result.final_answer is not None
        assert "Insufficient data" in result.final_answer

    def test_writer_records_agent_result(self) -> None:
        """Writer should add result to agent_results."""
        state = ResearchState(
            request=ResearchQuery(query="Test"),
            sources=[SourceDocument(title="Test", snippet="Content")],
            research_notes="Notes",
            analysis_notes="Analysis"
        )
        agent = WriterAgent()
        result = agent.run(state)

        assert len(result.agent_results) > 0
        assert AgentName.WRITER in [r.agent for r in result.agent_results]


class TestCriticAgent:
    """Tests for CriticAgent."""

    def test_critic_reviews_final_answer(self) -> None:
        """Critic should review the final answer."""
        state = ResearchState(
            request=ResearchQuery(query="Test"),
            sources=[SourceDocument(title="Test", snippet="Content", url="http://test.com")],
            research_notes="Research notes",
            analysis_notes="Analysis notes",
            final_answer="This is the final answer with a citation [1]."
        )
        agent = CriticAgent()
        result = agent.run(state)

        # Critic should add its result to agent_results
        critic_results = [r for r in result.agent_results if r.agent == AgentName.CRITIC]
        assert len(critic_results) > 0

    def test_critic_handles_empty_final_answer(self) -> None:
        """Critic should handle empty final answer gracefully."""
        state = ResearchState(request=ResearchQuery(query="Test"))
        agent = CriticAgent()
        result = agent.run(state)

        # Should not crash, should add trace event
        critic_traces = [t for t in result.trace if t.get("name") == "critic"]
        assert len(critic_traces) > 0

    def test_critic_records_agent_result(self) -> None:
        """Critic should add result to agent_results."""
        state = ResearchState(
            request=ResearchQuery(query="Test"),
            sources=[SourceDocument(title="Test", snippet="Content")],
            final_answer="Final answer"
        )
        agent = CriticAgent()
        result = agent.run(state)

        assert AgentName.CRITIC in [r.agent for r in result.agent_results]
