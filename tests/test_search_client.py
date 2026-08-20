"""Tests for search client."""

from multi_agent_research_lab.services.search_client import SearchClient


def test_search_client_returns_mock_results() -> None:
    """SearchClient should return mock results when no API key configured."""
    client = SearchClient()
    results = client.search("test query", max_results=3)

    assert len(results) > 0
    assert all(hasattr(r, "title") for r in results)
    assert all(hasattr(r, "snippet") for r in results)


def test_search_client_respects_max_results() -> None:
    """SearchClient should respect max_results parameter."""
    client = SearchClient()
    results = client.search("test", max_results=2)

    assert len(results) <= 2


def test_search_client_sources_have_metadata() -> None:
    """Search results should include metadata."""
    client = SearchClient()
    results = client.search("test query")

    for result in results:
        assert hasattr(result, "metadata")
        assert "engine" in result.metadata
