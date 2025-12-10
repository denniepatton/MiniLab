"""
Web Search Tool using Tavily API.

Provides general web search capabilities for agents.
For academic literature, prefer PubMed tool.
"""

from __future__ import annotations

import os
from typing import Any, Optional
from pydantic import Field

from .base import Tool, ToolInput, ToolOutput, ToolError
from ..utils import console


class SearchInput(ToolInput):
    """Input for web search."""
    query: str = Field(..., description="The search query")
    max_results: int = Field(10, description="Maximum number of results to return")
    search_depth: str = Field("basic", description="Search depth: 'basic' or 'advanced'")
    include_domains: Optional[list[str]] = Field(None, description="Only search these domains")
    exclude_domains: Optional[list[str]] = Field(None, description="Exclude these domains")


class WebSearchOutput(ToolOutput):
    """Output for web search."""
    query: Optional[str] = None
    results: Optional[list[dict]] = None
    answer: Optional[str] = None  # Tavily can provide a direct answer


class WebSearchTool(Tool):
    """
    Web search using Tavily API.
    
    Requires TAVILY_API_KEY environment variable.
    For academic literature, prefer the PubMed tool.
    """
    
    name = "web_search"
    description = "Search the web for information (use pubmed for academic literature)"
    
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, **kwargs)
        self.api_key = os.getenv("TAVILY_API_KEY")
    
    def get_actions(self) -> dict[str, str]:
        return {
            "search": "Search the web for information",
        }
    
    def get_input_schema(self, action: str) -> type[ToolInput]:
        schemas = {
            "search": SearchInput,
        }
        if action not in schemas:
            raise ToolError(self.name, action, f"Unknown action: {action}")
        return schemas[action]
    
    async def execute(self, action: str, params: dict[str, Any]) -> WebSearchOutput:
        """Execute a web search action."""
        validated = self.validate_input(action, params)
        
        try:
            if action == "search":
                return await self._search(validated)
            else:
                raise ToolError(self.name, action, f"Unknown action: {action}")
        except Exception as e:
            return WebSearchOutput(success=False, error=f"Search failed: {e}")
    
    async def _search(self, params: SearchInput) -> WebSearchOutput:
        """Perform a web search using Tavily."""
        if not self.api_key:
            return WebSearchOutput(
                success=False,
                query=params.query,
                error="TAVILY_API_KEY not set in environment"
            )
        
        try:
            # Try to import tavily
            try:
                from tavily import TavilyClient
            except ImportError:
                return WebSearchOutput(
                    success=False,
                    query=params.query,
                    error="tavily-python not installed. Run: pip install tavily-python"
                )
            
            client = TavilyClient(api_key=self.api_key)
            
            # Build search parameters
            search_params = {
                "query": params.query,
                "max_results": params.max_results,
                "search_depth": params.search_depth,
            }
            
            if params.include_domains:
                search_params["include_domains"] = params.include_domains
            if params.exclude_domains:
                search_params["exclude_domains"] = params.exclude_domains
            
            # Execute search
            console.search_start(params.query, "web")
            response = client.search(**search_params)
            
            # Extract results
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title"),
                    "url": result.get("url"),
                    "content": result.get("content"),
                    "score": result.get("score"),
                })
            
            console.search_complete("web", len(results))
            return WebSearchOutput(
                success=True,
                query=params.query,
                results=results,
                answer=response.get("answer"),  # Tavily sometimes provides direct answers
                data=f"Found {len(results)} results"
            )
            
        except Exception as e:
            return WebSearchOutput(
                success=False,
                query=params.query,
                error=f"Tavily search failed: {e}"
            )
