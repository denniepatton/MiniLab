from __future__ import annotations

import os
from typing import Any, Dict, List
import httpx

from . import Tool


class WebSearchTool(Tool):
    """
    Web search tool using a search API (e.g., Tavily, SerpAPI, or similar).
    For now, this is a placeholder that would need API credentials.
    """

    def __init__(self, api_key: str | None = None):
        super().__init__(
            name="web_search",
            description="Search the web for recent scientific papers, news, and information"
        )
        # Example: using Tavily API (you'd need to install tavily-python or use HTTP)
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def execute(self, query: str, max_results: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execute a web search.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            Dict with 'results' list containing search results
        """
        if not self.api_key:
            # Fallback: return mock results or explain API needed
            return {
                "status": "error",
                "message": "Web search requires TAVILY_API_KEY or similar API credentials",
                "results": []
            }
        
        # Example Tavily API call (adjust based on actual API)
        try:
            response = await self.client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "advanced",
                },
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                "status": "success",
                "query": query,
                "results": data.get("results", []),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "results": []
            }


class ArxivSearchTool(Tool):
    """
    Search arXiv for scientific papers.
    Uses the arXiv API which is free and doesn't require authentication.
    """

    def __init__(self):
        super().__init__(
            name="arxiv_search",
            description="Search arXiv for scientific papers"
        )
        self.client = httpx.AsyncClient(timeout=30.0)
        self.base_url = "http://export.arxiv.org/api/query"

    async def execute(self, query: str, max_results: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Search arXiv for papers.
        
        Args:
            query: Search query (can use arXiv query syntax)
            max_results: Maximum number of results
            
        Returns:
            Dict with paper results including titles, authors, abstracts, URLs
        """
        try:
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            
            response = await self.client.get(self.base_url, params=params)
            response.raise_for_status()
            
            # Parse XML response (simplified - would need proper XML parsing)
            # For now, return raw response or use feedparser library
            return {
                "status": "success",
                "query": query,
                "raw_response": response.text,
                "note": "Full parsing requires feedparser library - install with: pip install feedparser"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "results": []
            }


class PubMedSearchTool(Tool):
    """
    Search PubMed for biomedical literature.
    Uses NCBI E-utilities API (free, no API key required for basic use).
    """

    def __init__(self, email: str | None = None):
        super().__init__(
            name="pubmed_search",
            description="Search PubMed for biomedical and life sciences literature"
        )
        self.email = email or os.environ.get("NCBI_EMAIL", "user@example.com")
        self.client = httpx.AsyncClient(timeout=30.0)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    async def execute(self, query: str, max_results: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Search PubMed for papers.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dict with paper results
        """
        try:
            # Step 1: Search for PMIDs
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "email": self.email,
            }
            
            search_response = await self.client.get(
                f"{self.base_url}/esearch.fcgi",
                params=search_params
            )
            search_response.raise_for_status()
            search_data = search_response.json()
            
            pmids = search_data.get("esearchresult", {}).get("idlist", [])
            
            if not pmids:
                return {
                    "status": "success",
                    "query": query,
                    "results": [],
                    "message": "No results found"
                }
            
            # Step 2: Fetch summaries for PMIDs
            summary_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json",
                "email": self.email,
            }
            
            summary_response = await self.client.get(
                f"{self.base_url}/esummary.fcgi",
                params=summary_params
            )
            summary_response.raise_for_status()
            summary_data = summary_response.json()
            
            results = []
            for pmid in pmids:
                if pmid in summary_data.get("result", {}):
                    paper = summary_data["result"][pmid]
                    results.append({
                        "pmid": pmid,
                        "title": paper.get("title", ""),
                        "authors": [a.get("name", "") for a in paper.get("authors", [])],
                        "journal": paper.get("fulljournalname", ""),
                        "pub_date": paper.get("pubdate", ""),
                        "doi": paper.get("elocationid", ""),
                    })
            
            return {
                "status": "success",
                "query": query,
                "results": results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "results": []
            }
