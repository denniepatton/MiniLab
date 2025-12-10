"""
Citation Tool for bibliography management.

Provides citation formatting and DOI/PMID lookup.
"""

from __future__ import annotations

import re
from typing import Any, Optional
import httpx
from pydantic import Field

from .base import Tool, ToolInput, ToolOutput, ToolError


class FetchDOIInput(ToolInput):
    """Input for fetching citation by DOI."""
    doi: str = Field(..., description="Digital Object Identifier (e.g., '10.1038/nature12373')")


class FetchPMIDInput(ToolInput):
    """Input for fetching citation by PMID."""
    pmid: str = Field(..., description="PubMed ID")


class FormatCitationInput(ToolInput):
    """Input for formatting a citation."""
    citation: dict = Field(..., description="Citation data dictionary")
    style: str = Field("nature", description="Citation style: 'nature', 'apa', 'mla', 'bibtex'")


class CreateManualInput(ToolInput):
    """Input for creating a manual citation entry."""
    title: str = Field(..., description="Article/book title")
    authors: list[str] = Field(..., description="List of author names")
    year: Optional[int] = Field(None, description="Publication year")
    journal: Optional[str] = Field(None, description="Journal name")
    volume: Optional[str] = Field(None, description="Volume number")
    pages: Optional[str] = Field(None, description="Page numbers")
    doi: Optional[str] = Field(None, description="DOI if available")
    pmid: Optional[str] = Field(None, description="PMID if available")
    url: Optional[str] = Field(None, description="URL if available")


class CitationOutput(ToolOutput):
    """Output for citation operations."""
    citation: Optional[dict] = None
    formatted: Optional[str] = None
    bibtex: Optional[str] = None


class CitationTool(Tool):
    """
    Citation management tool.
    
    Supports:
    - Fetching citation data from DOI or PMID
    - Formatting citations in various styles
    - Creating manual citation entries
    """
    
    name = "citation"
    description = "Manage citations: fetch from DOI/PMID, format in various styles"
    
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, **kwargs)
    
    def get_actions(self) -> dict[str, str]:
        return {
            "fetch_doi": "Fetch citation data from a DOI",
            "fetch_pmid": "Fetch citation data from a PubMed ID",
            "format": "Format a citation in a specific style",
            "create_manual": "Create a manual citation entry",
        }
    
    def get_input_schema(self, action: str) -> type[ToolInput]:
        schemas = {
            "fetch_doi": FetchDOIInput,
            "fetch_pmid": FetchPMIDInput,
            "format": FormatCitationInput,
            "create_manual": CreateManualInput,
        }
        if action not in schemas:
            raise ToolError(self.name, action, f"Unknown action: {action}")
        return schemas[action]
    
    async def execute(self, action: str, params: dict[str, Any]) -> CitationOutput:
        """Execute a citation action."""
        validated = self.validate_input(action, params)
        
        try:
            if action == "fetch_doi":
                return await self._fetch_doi(validated)
            elif action == "fetch_pmid":
                return await self._fetch_pmid(validated)
            elif action == "format":
                return await self._format(validated)
            elif action == "create_manual":
                return await self._create_manual(validated)
            else:
                raise ToolError(self.name, action, f"Unknown action: {action}")
        except Exception as e:
            return CitationOutput(success=False, error=f"Citation operation failed: {e}")
    
    async def _fetch_doi(self, params: FetchDOIInput) -> CitationOutput:
        """Fetch citation data from CrossRef using DOI."""
        doi = params.doi.strip()
        
        # Normalize DOI
        if doi.startswith("https://doi.org/"):
            doi = doi[16:]
        elif doi.startswith("http://doi.org/"):
            doi = doi[15:]
        elif doi.startswith("doi:"):
            doi = doi[4:]
        
        async with httpx.AsyncClient(timeout=30) as client:
            # Use CrossRef API
            url = f"https://api.crossref.org/works/{doi}"
            headers = {"Accept": "application/json"}
            
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                work = data.get("message", {})
                
                citation = {
                    "doi": doi,
                    "title": work.get("title", [""])[0],
                    "authors": [],
                    "journal": work.get("container-title", [""])[0],
                    "year": None,
                    "volume": work.get("volume"),
                    "issue": work.get("issue"),
                    "pages": work.get("page"),
                    "url": f"https://doi.org/{doi}",
                }
                
                # Parse authors
                for author in work.get("author", []):
                    given = author.get("given", "")
                    family = author.get("family", "")
                    if family:
                        citation["authors"].append(f"{family}, {given}".strip(", "))
                
                # Parse date
                date_parts = work.get("published-print", {}).get("date-parts", [[]])
                if not date_parts[0]:
                    date_parts = work.get("published-online", {}).get("date-parts", [[]])
                if date_parts[0]:
                    citation["year"] = date_parts[0][0]
                
                return CitationOutput(
                    success=True,
                    citation=citation,
                    formatted=self._format_nature(citation),
                )
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return CitationOutput(
                        success=False,
                        error=f"DOI not found: {doi}"
                    )
                raise
    
    async def _fetch_pmid(self, params: FetchPMIDInput) -> CitationOutput:
        """Fetch citation data from PubMed using PMID."""
        from .pubmed import PubMedTool, FetchInput
        
        # Use PubMed tool to fetch
        pubmed = PubMedTool(self.agent_id)
        result = await pubmed._fetch(FetchInput(pmids=[params.pmid], include_abstract=False))
        
        if not result.success or not result.articles:
            return CitationOutput(
                success=False,
                error=f"PMID not found: {params.pmid}"
            )
        
        article = result.articles[0]
        
        citation = {
            "pmid": params.pmid,
            "title": article.get("title", ""),
            "authors": article.get("authors", []),
            "journal": article.get("journal", ""),
            "year": None,
            "doi": article.get("doi"),
            "url": article.get("url"),
        }
        
        # Parse year from date
        date = article.get("date", "")
        if date:
            year_match = re.search(r"\d{4}", date)
            if year_match:
                citation["year"] = int(year_match.group())
        
        return CitationOutput(
            success=True,
            citation=citation,
            formatted=self._format_nature(citation),
        )
    
    async def _format(self, params: FormatCitationInput) -> CitationOutput:
        """Format a citation in a specific style."""
        citation = params.citation
        
        if params.style == "nature":
            formatted = self._format_nature(citation)
        elif params.style == "apa":
            formatted = self._format_apa(citation)
        elif params.style == "mla":
            formatted = self._format_mla(citation)
        elif params.style == "bibtex":
            formatted = self._format_bibtex(citation)
        else:
            return CitationOutput(
                success=False,
                error=f"Unknown citation style: {params.style}"
            )
        
        return CitationOutput(
            success=True,
            citation=citation,
            formatted=formatted,
            bibtex=self._format_bibtex(citation) if params.style != "bibtex" else None,
        )
    
    async def _create_manual(self, params: CreateManualInput) -> CitationOutput:
        """Create a manual citation entry."""
        citation = {
            "title": params.title,
            "authors": params.authors,
            "year": params.year,
            "journal": params.journal,
            "volume": params.volume,
            "pages": params.pages,
            "doi": params.doi,
            "pmid": params.pmid,
            "url": params.url,
        }
        
        return CitationOutput(
            success=True,
            citation=citation,
            formatted=self._format_nature(citation),
            bibtex=self._format_bibtex(citation),
        )
    
    def _format_nature(self, citation: dict) -> str:
        """Format citation in Nature style."""
        parts = []
        
        # Authors
        authors = citation.get("authors", [])
        if authors:
            if len(authors) > 3:
                parts.append(f"{authors[0]} et al.")
            else:
                parts.append(", ".join(authors))
        
        # Title
        title = citation.get("title", "")
        if title:
            parts.append(title)
        
        # Journal, volume, pages (year)
        journal_parts = []
        if citation.get("journal"):
            journal_parts.append(f"*{citation['journal']}*")
        if citation.get("volume"):
            journal_parts.append(f"**{citation['volume']}**")
        if citation.get("pages"):
            journal_parts.append(citation["pages"])
        
        if journal_parts:
            journal_str = " ".join(journal_parts)
            if citation.get("year"):
                journal_str += f" ({citation['year']})"
            parts.append(journal_str)
        elif citation.get("year"):
            parts.append(f"({citation['year']})")
        
        result = ". ".join(parts)
        
        # Add DOI link
        if citation.get("doi"):
            result += f" https://doi.org/{citation['doi']}"
        elif citation.get("url"):
            result += f" {citation['url']}"
        
        return result
    
    def _format_apa(self, citation: dict) -> str:
        """Format citation in APA style."""
        parts = []
        
        # Authors
        authors = citation.get("authors", [])
        if authors:
            formatted_authors = []
            for author in authors[:6]:
                if ", " in author:
                    parts_split = author.split(", ")
                    formatted_authors.append(f"{parts_split[0]}, {parts_split[1][0]}.")
                else:
                    formatted_authors.append(author)
            
            if len(authors) > 6:
                formatted_authors.append("...")
                formatted_authors.append(authors[-1])
            
            parts.append(", ".join(formatted_authors[:-1]) + ", & " + formatted_authors[-1] if len(formatted_authors) > 1 else formatted_authors[0])
        
        # Year
        if citation.get("year"):
            parts.append(f"({citation['year']})")
        
        # Title
        if citation.get("title"):
            parts.append(citation["title"])
        
        # Journal
        if citation.get("journal"):
            journal_str = f"*{citation['journal']}*"
            if citation.get("volume"):
                journal_str += f", *{citation['volume']}*"
            if citation.get("pages"):
                journal_str += f", {citation['pages']}"
            parts.append(journal_str)
        
        result = " ".join(parts)
        
        if citation.get("doi"):
            result += f" https://doi.org/{citation['doi']}"
        
        return result
    
    def _format_mla(self, citation: dict) -> str:
        """Format citation in MLA style."""
        parts = []
        
        # Authors
        authors = citation.get("authors", [])
        if authors:
            if len(authors) == 1:
                parts.append(authors[0])
            elif len(authors) == 2:
                parts.append(f"{authors[0]}, and {authors[1]}")
            else:
                parts.append(f"{authors[0]}, et al")
        
        # Title
        if citation.get("title"):
            parts.append(f'"{citation["title"]}"')
        
        # Journal
        if citation.get("journal"):
            parts.append(f"*{citation['journal']}*")
        
        # Volume, year, pages
        details = []
        if citation.get("volume"):
            details.append(f"vol. {citation['volume']}")
        if citation.get("year"):
            details.append(str(citation["year"]))
        if citation.get("pages"):
            details.append(f"pp. {citation['pages']}")
        
        if details:
            parts.append(", ".join(details))
        
        return ". ".join(parts) + "."
    
    def _format_bibtex(self, citation: dict) -> str:
        """Format citation as BibTeX entry."""
        # Generate key
        first_author = citation.get("authors", ["unknown"])[0]
        if ", " in first_author:
            last_name = first_author.split(", ")[0]
        else:
            last_name = first_author.split()[-1] if first_author else "unknown"
        
        year = citation.get("year", "")
        title_word = citation.get("title", "untitled").split()[0].lower() if citation.get("title") else "untitled"
        key = f"{last_name.lower()}{year}{title_word}"
        key = re.sub(r"[^a-z0-9]", "", key)
        
        lines = [f"@article{{{key},"]
        
        if citation.get("authors"):
            authors_str = " and ".join(citation["authors"])
            lines.append(f'  author = {{{authors_str}}},')
        
        if citation.get("title"):
            lines.append(f'  title = {{{citation["title"]}}},')
        
        if citation.get("journal"):
            lines.append(f'  journal = {{{citation["journal"]}}},')
        
        if citation.get("year"):
            lines.append(f'  year = {{{citation["year"]}}},')
        
        if citation.get("volume"):
            lines.append(f'  volume = {{{citation["volume"]}}},')
        
        if citation.get("pages"):
            lines.append(f'  pages = {{{citation["pages"]}}},')
        
        if citation.get("doi"):
            lines.append(f'  doi = {{{citation["doi"]}}},')
        
        if citation.get("pmid"):
            lines.append(f'  pmid = {{{citation["pmid"]}}},')
        
        lines.append("}")
        
        return "\n".join(lines)
