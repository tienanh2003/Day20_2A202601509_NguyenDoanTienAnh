"""Search client abstraction for ResearcherAgent."""

import logging
from typing import Any

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Search client with mock fallback when no API key is configured."""

    def __init__(self) -> None:
        self._tavily_client: Any = None
        self._settings = get_settings()

    @property
    def tavily(self) -> Any:
        """Lazy-load Tavily client if available."""
        if self._tavily_client is None:
            try:
                from tavily import TavilyClient

                api_key = self._settings.tavily_api_key
                if api_key:
                    self._tavily_client = TavilyClient(api_key=api_key)
                    logger.info("Tavily search client initialized")
                else:
                    logger.warning("TAVILY_API_KEY not set, using mock search")
            except ImportError:
                logger.warning("tavily package not installed, using mock search")
        return self._tavily_client

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Uses Tavily if API key is configured, otherwise returns mock data.
        """
        if self.tavily:
            return self._search_tavily(query, max_results)
        return self._mock_search(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        """Search using Tavily API."""
        try:
            response = self.tavily.search(query=query, max_results=max_results)
            results = response.get("results", [])
            sources = []
            for item in results:
                sources.append(
                    SourceDocument(
                        title=item.get("title", "Untitled"),
                        url=item.get("url"),
                        snippet=item.get("content", "")[:500],
                        metadata={
                            "score": item.get("score", 0),
                            "engine": "tavily",
                        },
                    )
                )
            logger.info(f"Tavily returned {len(sources)} results")
            return sources
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Tavily search failed: {exc}, falling back to mock")
            return self._mock_search(query, max_results)

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Return mock search results for development/testing."""
        mock_sources = [
            SourceDocument(
                title=f"Overview: {query[:50]}",
                url="https://example.com/overview",
                snippet=(
                    f"This is a comprehensive overview of {query}. "
                    "Mock data for development without API keys."
                ),
                metadata={"score": 0.9, "engine": "mock"},
            ),
            SourceDocument(
                title=f"Latest Research: {query[:50]}",
                url="https://example.com/research",
                snippet=(
                    f"Recent academic research on {query} suggests that "
                    "this topic is rapidly evolving with new developments weekly."
                ),
                metadata={"score": 0.85, "engine": "mock"},
            ),
            SourceDocument(
                title=f"Best Practices: {query[:50]}",
                url="https://example.com/practices",
                snippet=(
                    f"Industry best practices for {query} include: "
                    "1) Start with clear objectives, 2) Use structured approaches, "
                    "3) Continuously iterate and improve."
                ),
                metadata={"score": 0.8, "engine": "mock"},
            ),
        ]
        return mock_sources[:max_results]
