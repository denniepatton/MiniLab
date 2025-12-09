from __future__ import annotations

import os
from typing import Any, Dict, List
import httpx

from . import Tool


class WebSearchTool(Tool):
    """
    Web search tool that uses PubMed for scientific literature.
    PubMed is free and doesn't require API keys.
    """

    def __init__(self, api_key: str | None = None):
        super().__init__(
            name="web_search",
            description="""Search for scientific papers and biomedical literature via PubMed.
            
Actions:
- search: Search PubMed (requires: query, optional: max_results)

Example: {"tool": "web_search", "action": "search", "params": {"query": "Lu-177-PSMA prostate cancer biomarkers"}}

Returns paper titles, authors, journals, publication dates, and DOIs."""
        )
        # PubMed is our primary (and only) backend - no API key needed
        self._pubmed = PubMedSearchTool()

    async def execute(self, action: str = "search", query: str = "", max_results: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Execute a literature search via PubMed.
        
        Args:
            action: Action to perform (only "search" supported)
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            Dict with 'success', 'results', etc.
        """
        if action != "search":
            return {
                "success": False,
                "error": f"Unknown action: {action}. Use 'search' with a 'query' parameter.",
            }
        
        if not query:
            return {
                "success": False,
                "error": "Missing required parameter: 'query'",
            }
        
        # Use PubMed for all searches
        pubmed_result = await self._pubmed.execute(action="search", query=query, max_results=max_results)
        
        if pubmed_result.get("success"):
            return {
                "success": True,
                "source": "pubmed",
                "query": query,
                "results": pubmed_result.get("results", []),
            }
        else:
            return {
                "success": False,
                "error": pubmed_result.get("error", "PubMed search failed"),
                "source": "pubmed",
            }


class ArxivSearchTool(Tool):
    """
    Search arXiv for scientific papers.
    Uses the arXiv API which is free and doesn't require authentication.
    """

    def __init__(self):
        super().__init__(
            name="arxiv_search",
            description="""Search arXiv for scientific papers.

Actions:
- search: Search arXiv (requires: query, optional: max_results)

Example: {"tool": "arxiv_search", "action": "search", "params": {"query": "prostate cancer machine learning"}}"""
        )
        self.client = httpx.AsyncClient(timeout=30.0)
        self.base_url = "http://export.arxiv.org/api/query"

    async def execute(self, action: str = "search", query: str = "", max_results: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Search arXiv for papers.
        
        Args:
            action: Action to perform (only "search" supported)
            query: Search query (can use arXiv query syntax)
            max_results: Maximum number of results
            
        Returns:
            Dict with paper results including titles, authors, abstracts, URLs
        """
        if action != "search":
            return {"success": False, "error": f"Unknown action: {action}. Use 'search'."}
        
        if not query:
            return {"success": False, "error": "Missing required parameter: 'query'"}
        
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
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            results = []
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                link = entry.find("atom:link[@title='pdf']", ns)
                published = entry.find('atom:published', ns)
                authors = entry.findall('atom:author/atom:name', ns)
                
                results.append({
                    "title": title.text.strip() if title is not None else "",
                    "summary": summary.text.strip()[:500] if summary is not None else "",
                    "pdf_url": link.get('href') if link is not None else "",
                    "published": published.text[:10] if published is not None else "",
                    "authors": [a.text for a in authors[:5]],  # First 5 authors
                })
            
            return {
                "success": True,
                "query": query,
                "results": results,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


class PubMedSearchTool(Tool):
    """
    Search PubMed for biomedical literature.
    Uses NCBI E-utilities API (free, no API key required for basic use).
    """

    def __init__(self, email: str | None = None):
        super().__init__(
            name="pubmed_search",
            description="""Search PubMed for biomedical and life sciences literature.

Actions:
- search: Search PubMed (requires: query, optional: max_results)

Example: {"tool": "pubmed_search", "action": "search", "params": {"query": "PSMA prostate cancer therapy"}}"""
        )
        self.email = email or os.environ.get("NCBI_EMAIL", "user@example.com")
        self.client = httpx.AsyncClient(timeout=30.0)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    async def execute(self, action: str = "search", query: str = "", max_results: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Search PubMed for papers.
        
        Args:
            action: Action to perform (only "search" supported)
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dict with paper results
        """
        if action != "search":
            return {"success": False, "error": f"Unknown action: {action}. Use 'search'."}
        
        if not query:
            return {"success": False, "error": "Missing required parameter: 'query'"}
        
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
                    "success": True,
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
                        "authors": [a.get("name", "") for a in paper.get("authors", [])][:5],
                        "journal": paper.get("fulljournalname", ""),
                        "pub_date": paper.get("pubdate", ""),
                        "doi": paper.get("elocationid", ""),
                    })
            
            return {
                "success": True,
                "query": query,
                "results": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
