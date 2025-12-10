"""
arXiv Search Tool.

Provides access to arXiv preprints.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Optional
from urllib.parse import urlencode
import httpx
from pydantic import Field

from .base import Tool, ToolInput, ToolOutput, ToolError


class SearchInput(ToolInput):
    """Input for arXiv search."""
    query: str = Field(..., description="Search query (supports arXiv syntax)")
    max_results: int = Field(20, description="Maximum number of results")
    sort_by: str = Field("relevance", description="Sort by: 'relevance', 'lastUpdatedDate', or 'submittedDate'")
    sort_order: str = Field("descending", description="Sort order: 'ascending' or 'descending'")
    categories: Optional[list[str]] = Field(None, description="Filter by arXiv categories (e.g., ['cs.AI', 'q-bio.QM'])")


class FetchInput(ToolInput):
    """Input for fetching arXiv paper details."""
    arxiv_ids: list[str] = Field(..., description="List of arXiv IDs to fetch (e.g., ['2301.12345', '2302.67890'])")


class ArxivOutput(ToolOutput):
    """Output for arXiv operations."""
    query: Optional[str] = None
    arxiv_ids: Optional[list[str]] = None
    papers: Optional[list[dict]] = None
    count: Optional[int] = None


class ArxivTool(Tool):
    """
    arXiv preprint search.
    
    Free to use, no API key required.
    """
    
    name = "arxiv"
    description = "Search arXiv for preprints and technical papers"
    
    ARXIV_API = "http://export.arxiv.org/api/query"
    
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, **kwargs)
    
    def get_actions(self) -> dict[str, str]:
        return {
            "search": "Search arXiv for papers matching a query",
            "fetch": "Fetch detailed information for specific arXiv IDs",
        }
    
    def get_input_schema(self, action: str) -> type[ToolInput]:
        schemas = {
            "search": SearchInput,
            "fetch": FetchInput,
        }
        if action not in schemas:
            raise ToolError(self.name, action, f"Unknown action: {action}")
        return schemas[action]
    
    async def execute(self, action: str, params: dict[str, Any]) -> ArxivOutput:
        """Execute an arXiv action."""
        validated = self.validate_input(action, params)
        
        try:
            if action == "search":
                return await self._search(validated)
            elif action == "fetch":
                return await self._fetch(validated)
            else:
                raise ToolError(self.name, action, f"Unknown action: {action}")
        except Exception as e:
            return ArxivOutput(success=False, error=f"arXiv operation failed: {e}")
    
    async def _search(self, params: SearchInput) -> ArxivOutput:
        """Search arXiv for papers."""
        # Build query
        search_query = params.query
        
        # Add category filter if specified
        if params.categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in params.categories])
            search_query = f"({search_query}) AND ({cat_query})"
        
        # Build API parameters
        api_params = {
            "search_query": search_query,
            "start": 0,
            "max_results": params.max_results,
            "sortBy": params.sort_by,
            "sortOrder": params.sort_order,
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            url = f"{self.ARXIV_API}?{urlencode(api_params)}"
            response = await client.get(url)
            response.raise_for_status()
            
            papers = self._parse_arxiv_response(response.text)
            arxiv_ids = [p.get("arxiv_id") for p in papers if p.get("arxiv_id")]
            
            return ArxivOutput(
                success=True,
                query=params.query,
                arxiv_ids=arxiv_ids,
                papers=papers,
                count=len(papers),
                data=f"Found {len(papers)} papers"
            )
    
    async def _fetch(self, params: FetchInput) -> ArxivOutput:
        """Fetch details for specific arXiv IDs."""
        if not params.arxiv_ids:
            return ArxivOutput(success=True, arxiv_ids=[], papers=[], count=0)
        
        # Build ID list query
        id_list = ",".join(params.arxiv_ids)
        
        api_params = {
            "id_list": id_list,
            "max_results": len(params.arxiv_ids),
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            url = f"{self.ARXIV_API}?{urlencode(api_params)}"
            response = await client.get(url)
            response.raise_for_status()
            
            papers = self._parse_arxiv_response(response.text)
            
            return ArxivOutput(
                success=True,
                arxiv_ids=params.arxiv_ids,
                papers=papers,
                count=len(papers),
            )
    
    def _parse_arxiv_response(self, xml_text: str) -> list[dict]:
        """Parse arXiv API XML response."""
        papers = []
        
        # Define namespace
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        
        try:
            root = ET.fromstring(xml_text)
            
            for entry in root.findall("atom:entry", ns):
                paper = {}
                
                # arXiv ID (extract from URL)
                id_elem = entry.find("atom:id", ns)
                if id_elem is not None:
                    full_id = id_elem.text
                    # Extract ID from URL like http://arxiv.org/abs/2301.12345v1
                    if "/abs/" in full_id:
                        arxiv_id = full_id.split("/abs/")[-1]
                        # Remove version suffix for canonical ID
                        paper["arxiv_id"] = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                        paper["arxiv_id_versioned"] = arxiv_id
                
                # Title
                title_elem = entry.find("atom:title", ns)
                if title_elem is not None:
                    paper["title"] = " ".join(title_elem.text.split())  # Normalize whitespace
                
                # Abstract
                summary_elem = entry.find("atom:summary", ns)
                if summary_elem is not None:
                    paper["abstract"] = " ".join(summary_elem.text.split())
                
                # Authors
                authors = []
                for author_elem in entry.findall("atom:author", ns):
                    name_elem = author_elem.find("atom:name", ns)
                    if name_elem is not None:
                        authors.append(name_elem.text)
                paper["authors"] = authors
                
                # Published date
                published_elem = entry.find("atom:published", ns)
                if published_elem is not None:
                    paper["published"] = published_elem.text[:10]  # YYYY-MM-DD
                
                # Updated date
                updated_elem = entry.find("atom:updated", ns)
                if updated_elem is not None:
                    paper["updated"] = updated_elem.text[:10]
                
                # Categories
                categories = []
                for cat_elem in entry.findall("atom:category", ns):
                    term = cat_elem.get("term")
                    if term:
                        categories.append(term)
                paper["categories"] = categories
                paper["primary_category"] = categories[0] if categories else None
                
                # Links
                for link_elem in entry.findall("atom:link", ns):
                    rel = link_elem.get("rel")
                    href = link_elem.get("href")
                    if rel == "alternate":
                        paper["url"] = href
                    elif link_elem.get("title") == "pdf":
                        paper["pdf_url"] = href
                
                # DOI (if available)
                doi_elem = entry.find("arxiv:doi", ns)
                if doi_elem is not None:
                    paper["doi"] = doi_elem.text
                
                # Comment (often contains page count, conference info)
                comment_elem = entry.find("arxiv:comment", ns)
                if comment_elem is not None:
                    paper["comment"] = comment_elem.text
                
                papers.append(paper)
                
        except ET.ParseError:
            pass
        
        return papers
