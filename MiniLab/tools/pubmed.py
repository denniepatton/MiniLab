"""
PubMed Search Tool using NCBI E-utilities.

Provides comprehensive literature search capabilities.
Requires NCBI_EMAIL and optionally NCBI_API_KEY for higher rate limits.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any, Optional
from urllib.parse import urlencode
import httpx
from pydantic import Field

from .base import Tool, ToolInput, ToolOutput, ToolError
from ..utils import console


class SearchInput(ToolInput):
    """Input for PubMed search."""
    query: str = Field(..., description="Search query (supports PubMed syntax)")
    max_results: int = Field(20, description="Maximum number of results")
    sort: str = Field("relevance", description="Sort by: 'relevance', 'date', or 'citation'")
    min_date: Optional[str] = Field(None, description="Minimum date (YYYY/MM/DD)")
    max_date: Optional[str] = Field(None, description="Maximum date (YYYY/MM/DD)")


class FetchInput(ToolInput):
    """Input for fetching PubMed article details."""
    pmids: list[str] = Field(..., description="List of PubMed IDs to fetch")
    include_abstract: bool = Field(True, description="Include abstracts in results")


class CitedByInput(ToolInput):
    """Input for finding articles that cite a given article."""
    pmid: str = Field(..., description="PubMed ID to find citations for")
    max_results: int = Field(20, description="Maximum number of results")


class RelatedInput(ToolInput):
    """Input for finding related articles."""
    pmid: str = Field(..., description="PubMed ID to find related articles for")
    max_results: int = Field(20, description="Maximum number of results")


class PubMedOutput(ToolOutput):
    """Output for PubMed operations."""
    query: Optional[str] = None
    pmids: Optional[list[str]] = None
    articles: Optional[list[dict]] = None
    count: Optional[int] = None


class PubMedTool(Tool):
    """
    PubMed literature search using NCBI E-utilities.
    
    Environment variables:
    - NCBI_EMAIL: Required for E-utilities (identifies your application)
    - NCBI_API_KEY: Optional, provides higher rate limits (10 req/sec vs 3 req/sec)
    
    This is the preferred tool for academic literature searches.
    """
    
    name = "pubmed"
    description = "Search PubMed for scientific literature (preferred over web_search for academic papers)"
    
    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, **kwargs)
        self.email = os.getenv("NCBI_EMAIL")
        self.api_key = os.getenv("NCBI_API_KEY")
    
    def get_actions(self) -> dict[str, str]:
        return {
            "search": "Search PubMed for articles matching a query",
            "fetch": "Fetch detailed information for specific PubMed IDs",
            "cited_by": "Find articles that cite a given article",
            "related": "Find articles related to a given article",
        }
    
    def get_input_schema(self, action: str) -> type[ToolInput]:
        schemas = {
            "search": SearchInput,
            "fetch": FetchInput,
            "cited_by": CitedByInput,
            "related": RelatedInput,
        }
        if action not in schemas:
            raise ToolError(self.name, action, f"Unknown action: {action}")
        return schemas[action]
    
    def _get_common_params(self) -> dict[str, str]:
        """Get common parameters for E-utilities requests."""
        params = {"tool": "MiniLab"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params
    
    async def execute(self, action: str, params: dict[str, Any]) -> PubMedOutput:
        """Execute a PubMed action."""
        validated = self.validate_input(action, params)
        
        if not self.email:
            return PubMedOutput(
                success=False,
                error="NCBI_EMAIL not set in environment. Required for PubMed API."
            )
        
        try:
            if action == "search":
                return await self._search(validated)
            elif action == "fetch":
                return await self._fetch(validated)
            elif action == "cited_by":
                return await self._cited_by(validated)
            elif action == "related":
                return await self._related(validated)
            else:
                raise ToolError(self.name, action, f"Unknown action: {action}")
        except Exception as e:
            return PubMedOutput(success=False, error=f"PubMed operation failed: {e}")
    
    async def _search(self, params: SearchInput) -> PubMedOutput:
        """Search PubMed for articles."""
        # Build search parameters
        search_params = self._get_common_params()
        search_params["db"] = "pubmed"
        search_params["term"] = params.query
        search_params["retmax"] = str(params.max_results)
        search_params["retmode"] = "json"
        search_params["usehistory"] = "y"
        
        # Sort parameter
        if params.sort == "date":
            search_params["sort"] = "pub_date"
        elif params.sort == "citation":
            search_params["sort"] = "relevance"  # PubMed doesn't directly sort by citation count
        
        # Date filters
        if params.min_date:
            search_params["mindate"] = params.min_date.replace("/", "/")
        if params.max_date:
            search_params["maxdate"] = params.max_date.replace("/", "/")
        if params.min_date or params.max_date:
            search_params["datetype"] = "pdat"  # Publication date
        
        async with httpx.AsyncClient(timeout=30) as client:
            # Execute search
            console.search_start(params.query, "PubMed")
            search_url = f"{self.EUTILS_BASE}/esearch.fcgi?{urlencode(search_params)}"
            response = await client.get(search_url)
            response.raise_for_status()
            
            search_result = response.json()
            esearch = search_result.get("esearchresult", {})
            
            pmids = esearch.get("idlist", [])
            total_count = int(esearch.get("count", 0))
            
            if not pmids:
                return PubMedOutput(
                    success=True,
                    query=params.query,
                    pmids=[],
                    articles=[],
                    count=0,
                    data="No results found"
                )
            
            # Fetch details for found PMIDs
            fetch_result = await self._fetch(FetchInput(pmids=pmids, include_abstract=True))
            
            return PubMedOutput(
                success=True,
                query=params.query,
                pmids=pmids,
                articles=fetch_result.articles,
                count=total_count,
                data=f"Found {total_count} articles, returning {len(pmids)}"
            )
    
    async def _fetch(self, params: FetchInput) -> PubMedOutput:
        """Fetch detailed information for PubMed IDs."""
        if not params.pmids:
            return PubMedOutput(success=True, pmids=[], articles=[], count=0)
        
        fetch_params = self._get_common_params()
        fetch_params["db"] = "pubmed"
        fetch_params["id"] = ",".join(params.pmids)
        fetch_params["rettype"] = "abstract" if params.include_abstract else "docsum"
        fetch_params["retmode"] = "xml"
        
        async with httpx.AsyncClient(timeout=30) as client:
            fetch_url = f"{self.EUTILS_BASE}/efetch.fcgi?{urlencode(fetch_params)}"
            response = await client.get(fetch_url)
            response.raise_for_status()
            
            # Parse XML response
            articles = self._parse_pubmed_xml(response.text, params.include_abstract)
            
            return PubMedOutput(
                success=True,
                pmids=params.pmids,
                articles=articles,
                count=len(articles),
            )
    
    async def _cited_by(self, params: CitedByInput) -> PubMedOutput:
        """Find articles that cite a given article."""
        link_params = self._get_common_params()
        link_params["dbfrom"] = "pubmed"
        link_params["db"] = "pubmed"
        link_params["id"] = params.pmid
        link_params["linkname"] = "pubmed_pubmed_citedin"
        link_params["retmode"] = "json"
        
        async with httpx.AsyncClient(timeout=30) as client:
            link_url = f"{self.EUTILS_BASE}/elink.fcgi?{urlencode(link_params)}"
            response = await client.get(link_url)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract citing PMIDs
            pmids = []
            linksets = result.get("linksets", [])
            if linksets:
                linksetdbs = linksets[0].get("linksetdbs", [])
                if linksetdbs:
                    links = linksetdbs[0].get("links", [])
                    pmids = [str(l) for l in links[:params.max_results]]
            
            if not pmids:
                return PubMedOutput(
                    success=True,
                    pmids=[],
                    articles=[],
                    count=0,
                    data=f"No citing articles found for PMID {params.pmid}"
                )
            
            # Fetch details
            fetch_result = await self._fetch(FetchInput(pmids=pmids, include_abstract=True))
            
            return PubMedOutput(
                success=True,
                pmids=pmids,
                articles=fetch_result.articles,
                count=len(pmids),
                data=f"Found {len(pmids)} articles citing PMID {params.pmid}"
            )
    
    async def _related(self, params: RelatedInput) -> PubMedOutput:
        """Find articles related to a given article."""
        link_params = self._get_common_params()
        link_params["dbfrom"] = "pubmed"
        link_params["db"] = "pubmed"
        link_params["id"] = params.pmid
        link_params["linkname"] = "pubmed_pubmed"
        link_params["retmode"] = "json"
        
        async with httpx.AsyncClient(timeout=30) as client:
            link_url = f"{self.EUTILS_BASE}/elink.fcgi?{urlencode(link_params)}"
            response = await client.get(link_url)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract related PMIDs
            pmids = []
            linksets = result.get("linksets", [])
            if linksets:
                linksetdbs = linksets[0].get("linksetdbs", [])
                if linksetdbs:
                    links = linksetdbs[0].get("links", [])
                    pmids = [str(l) for l in links[:params.max_results]]
            
            if not pmids:
                return PubMedOutput(
                    success=True,
                    pmids=[],
                    articles=[],
                    count=0,
                    data=f"No related articles found for PMID {params.pmid}"
                )
            
            # Fetch details
            fetch_result = await self._fetch(FetchInput(pmids=pmids, include_abstract=True))
            
            return PubMedOutput(
                success=True,
                pmids=pmids,
                articles=fetch_result.articles,
                count=len(pmids),
                data=f"Found {len(pmids)} articles related to PMID {params.pmid}"
            )
    
    def _parse_pubmed_xml(self, xml_text: str, include_abstract: bool) -> list[dict]:
        """Parse PubMed XML response into article dictionaries."""
        articles = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article_elem in root.findall(".//PubmedArticle"):
                article = {}
                
                # PMID
                pmid_elem = article_elem.find(".//PMID")
                if pmid_elem is not None:
                    article["pmid"] = pmid_elem.text
                
                # Title
                title_elem = article_elem.find(".//ArticleTitle")
                if title_elem is not None:
                    article["title"] = "".join(title_elem.itertext())
                
                # Authors
                authors = []
                for author_elem in article_elem.findall(".//Author"):
                    last_name = author_elem.findtext("LastName", "")
                    initials = author_elem.findtext("Initials", "")
                    if last_name:
                        authors.append(f"{last_name} {initials}".strip())
                article["authors"] = authors
                
                # Journal
                journal_elem = article_elem.find(".//Journal/Title")
                if journal_elem is not None:
                    article["journal"] = journal_elem.text
                
                # Publication date
                pub_date = article_elem.find(".//PubDate")
                if pub_date is not None:
                    year = pub_date.findtext("Year", "")
                    month = pub_date.findtext("Month", "")
                    day = pub_date.findtext("Day", "")
                    article["date"] = f"{year} {month} {day}".strip()
                
                # DOI
                for article_id in article_elem.findall(".//ArticleId"):
                    if article_id.get("IdType") == "doi":
                        article["doi"] = article_id.text
                        break
                
                # Abstract
                if include_abstract:
                    abstract_texts = []
                    for abstract_elem in article_elem.findall(".//AbstractText"):
                        label = abstract_elem.get("Label", "")
                        text = "".join(abstract_elem.itertext())
                        if label:
                            abstract_texts.append(f"{label}: {text}")
                        else:
                            abstract_texts.append(text)
                    if abstract_texts:
                        article["abstract"] = " ".join(abstract_texts)
                
                # MeSH terms
                mesh_terms = []
                for mesh_elem in article_elem.findall(".//MeshHeading/DescriptorName"):
                    mesh_terms.append(mesh_elem.text)
                if mesh_terms:
                    article["mesh_terms"] = mesh_terms
                
                # Keywords
                keywords = []
                for keyword_elem in article_elem.findall(".//Keyword"):
                    keywords.append(keyword_elem.text)
                if keywords:
                    article["keywords"] = keywords
                
                # PubMed URL
                if article.get("pmid"):
                    article["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/"
                
                articles.append(article)
                
        except ET.ParseError as e:
            # Return what we have, logging the error
            pass
        
        return articles
