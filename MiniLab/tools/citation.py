"""
Citation Tool - Fetch and format academic citations with DOI links

This tool allows agents to:
1. Fetch citation metadata from DOI
2. Format citations in various styles
3. Generate clickable DOI links
4. Build bibliographies
"""

from __future__ import annotations

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import urllib.parse


@dataclass
class Citation:
    """Structured citation data."""
    
    title: str
    authors: List[str]
    year: Optional[int] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    pmid: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "journal": self.journal,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "doi": self.doi,
            "url": self.url,
            "pmid": self.pmid,
        }


class CitationTool:
    """
    Tool for fetching and formatting academic citations.
    
    Note: This is a basic implementation. For production use, integrate with
    APIs like CrossRef, PubMed, or services like Zotero.
    """
    
    def __init__(self):
        self.citations: Dict[str, Citation] = {}  # Cache by DOI
    
    async def execute(self, action: str, **params) -> Dict[str, Any]:
        """
        Execute a citation operation.
        
        Actions:
            - fetch_from_doi: Fetch citation from DOI
            - fetch_from_pmid: Fetch citation from PubMed ID
            - format_citation: Format a citation in specified style
            - format_bibliography: Format multiple citations as bibliography
            - create_manual: Create a citation manually
        """
        try:
            if action == "fetch_from_doi":
                return await self._fetch_from_doi(params.get("doi"))
            
            elif action == "fetch_from_pmid":
                return await self._fetch_from_pmid(params.get("pmid"))
            
            elif action == "format_citation":
                return self._format_citation(
                    params.get("doi"),
                    params.get("style", "apa")
                )
            
            elif action == "format_bibliography":
                return self._format_bibliography(
                    params.get("dois", []),
                    params.get("style", "apa")
                )
            
            elif action == "create_manual":
                return self._create_manual_citation(params)
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    async def _fetch_from_doi(self, doi: str) -> Dict[str, Any]:
        """
        Fetch citation from DOI.
        
        Note: This is a placeholder. In production, use CrossRef API:
        https://api.crossref.org/works/{doi}
        """
        if not doi:
            return {
                "success": False,
                "error": "DOI is required",
            }
        
        # Clean DOI
        doi = doi.strip()
        if doi.startswith("http"):
            # Extract DOI from URL
            match = re.search(r"10\.\d{4,}/[^\s]+", doi)
            if match:
                doi = match.group(0)
        
        # Check cache
        if doi in self.citations:
            citation = self.citations[doi]
            return {
                "success": True,
                "citation": citation.to_dict(),
                "formatted_apa": self._format_apa(citation),
                "doi_link": f"https://doi.org/{doi}",
                "source": "cache",
            }
        
        # In a real implementation, fetch from CrossRef API here
        # For now, return a placeholder
        return {
            "success": False,
            "error": "Citation fetching not yet implemented. Use 'create_manual' action to add citations manually.",
            "doi": doi,
            "doi_link": f"https://doi.org/{doi}",
            "instructions": {
                "action": "create_manual",
                "params": {
                    "doi": doi,
                    "title": "Article Title",
                    "authors": ["Last, F. M.", "Last2, F. M."],
                    "year": 2024,
                    "journal": "Journal Name",
                    "volume": "1",
                    "issue": "1",
                    "pages": "1-10",
                },
            },
        }
    
    async def _fetch_from_pmid(self, pmid: str) -> Dict[str, Any]:
        """
        Fetch citation from PubMed ID.
        
        Note: This is a placeholder. In production, use PubMed E-utilities:
        https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
        """
        if not pmid:
            return {
                "success": False,
                "error": "PMID is required",
            }
        
        pmid = pmid.strip()
        
        return {
            "success": False,
            "error": "PubMed fetching not yet implemented. Use 'create_manual' action to add citations manually.",
            "pmid": pmid,
            "pubmed_link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "instructions": {
                "action": "create_manual",
                "params": {
                    "pmid": pmid,
                    "title": "Article Title",
                    "authors": ["Last, F. M.", "Last2, F. M."],
                    "year": 2024,
                    "journal": "Journal Name",
                    "volume": "1",
                    "issue": "1",
                    "pages": "1-10",
                },
            },
        }
    
    def _create_manual_citation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a citation manually from provided metadata."""
        required = ["title", "authors"]
        missing = [field for field in required if not params.get(field)]
        
        if missing:
            return {
                "success": False,
                "error": f"Missing required fields: {', '.join(missing)}",
            }
        
        citation = Citation(
            title=params["title"],
            authors=params["authors"] if isinstance(params["authors"], list) else [params["authors"]],
            year=params.get("year"),
            journal=params.get("journal"),
            volume=params.get("volume"),
            issue=params.get("issue"),
            pages=params.get("pages"),
            doi=params.get("doi"),
            url=params.get("url"),
            pmid=params.get("pmid"),
        )
        
        # Cache by DOI if available
        if citation.doi:
            self.citations[citation.doi] = citation
        
        return {
            "success": True,
            "citation": citation.to_dict(),
            "formatted_apa": self._format_apa(citation),
            "doi_link": f"https://doi.org/{citation.doi}" if citation.doi else None,
            "pubmed_link": f"https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/" if citation.pmid else None,
        }
    
    def _format_citation(self, doi: str, style: str = "apa") -> Dict[str, Any]:
        """Format a citation in specified style."""
        if doi not in self.citations:
            return {
                "success": False,
                "error": f"Citation not found: {doi}. Fetch it first.",
            }
        
        citation = self.citations[doi]
        
        if style == "apa":
            formatted = self._format_apa(citation)
        elif style == "mla":
            formatted = self._format_mla(citation)
        elif style == "chicago":
            formatted = self._format_chicago(citation)
        else:
            return {
                "success": False,
                "error": f"Unknown citation style: {style}. Supported: apa, mla, chicago",
            }
        
        return {
            "success": True,
            "formatted": formatted,
            "doi_link": f"https://doi.org/{citation.doi}" if citation.doi else None,
        }
    
    def _format_bibliography(self, dois: List[str], style: str = "apa") -> Dict[str, Any]:
        """Format multiple citations as a bibliography."""
        if not dois:
            return {
                "success": False,
                "error": "No DOIs provided",
            }
        
        formatted_citations = []
        missing_dois = []
        
        for doi in dois:
            if doi in self.citations:
                citation = self.citations[doi]
                if style == "apa":
                    formatted = self._format_apa(citation)
                elif style == "mla":
                    formatted = self._format_mla(citation)
                elif style == "chicago":
                    formatted = self._format_chicago(citation)
                else:
                    formatted = str(citation)
                
                formatted_citations.append(formatted)
            else:
                missing_dois.append(doi)
        
        if missing_dois:
            return {
                "success": False,
                "error": f"Some citations not found: {', '.join(missing_dois)}",
                "formatted_count": len(formatted_citations),
            }
        
        # Sort alphabetically by first author's last name
        formatted_citations.sort()
        
        bibliography = "\n\n".join(formatted_citations)
        
        return {
            "success": True,
            "bibliography": bibliography,
            "count": len(formatted_citations),
            "style": style,
        }
    
    def _format_apa(self, citation: Citation) -> str:
        """Format citation in APA style."""
        # Authors
        if len(citation.authors) == 1:
            authors = citation.authors[0]
        elif len(citation.authors) == 2:
            authors = f"{citation.authors[0]} & {citation.authors[1]}"
        else:
            authors = ", ".join(citation.authors[:-1]) + f", & {citation.authors[-1]}"
        
        # Year
        year = f"({citation.year})" if citation.year else "(n.d.)"
        
        # Title
        title = citation.title.rstrip(".")
        
        # Journal info
        journal_info = ""
        if citation.journal:
            journal_info = f"*{citation.journal}*"
            if citation.volume:
                journal_info += f", *{citation.volume}*"
            if citation.issue:
                journal_info += f"({citation.issue})"
            if citation.pages:
                journal_info += f", {citation.pages}"
        
        # DOI or URL
        link = ""
        if citation.doi:
            link = f"https://doi.org/{citation.doi}"
        elif citation.url:
            link = citation.url
        
        # Assemble
        parts = [authors, year, title]
        if journal_info:
            parts.append(journal_info)
        if link:
            parts.append(link)
        
        return f"{parts[0]} {parts[1]}. {parts[2]}. " + ". ".join(parts[3:]) + "."
    
    def _format_mla(self, citation: Citation) -> str:
        """Format citation in MLA style."""
        # First author (Last, First)
        if citation.authors:
            first_author = citation.authors[0]
            if len(citation.authors) > 1:
                authors = f"{first_author}, et al."
            else:
                authors = first_author
        else:
            authors = "Unknown"
        
        # Title
        title = f'"{citation.title}"'
        
        # Journal
        journal = f"*{citation.journal}*" if citation.journal else ""
        
        # Volume/Issue
        vol_issue = ""
        if citation.volume:
            vol_issue = f"vol. {citation.volume}"
            if citation.issue:
                vol_issue += f", no. {citation.issue}"
        
        # Year
        year = str(citation.year) if citation.year else "n.d."
        
        # Pages
        pages = f"pp. {citation.pages}" if citation.pages else ""
        
        # DOI
        doi = f"https://doi.org/{citation.doi}" if citation.doi else ""
        
        # Assemble
        parts = [p for p in [authors, title, journal, vol_issue, year, pages, doi] if p]
        return ", ".join(parts) + "."
    
    def _format_chicago(self, citation: Citation) -> str:
        """Format citation in Chicago style."""
        # Authors
        if len(citation.authors) == 1:
            authors = citation.authors[0]
        elif len(citation.authors) == 2:
            authors = f"{citation.authors[0]} and {citation.authors[1]}"
        elif len(citation.authors) == 3:
            authors = f"{citation.authors[0]}, {citation.authors[1]}, and {citation.authors[2]}"
        else:
            authors = f"{citation.authors[0]} et al."
        
        # Title
        title = f'"{citation.title}"'
        
        # Journal
        journal = f"*{citation.journal}*" if citation.journal else ""
        
        # Volume/Issue
        vol_issue = ""
        if citation.volume:
            vol_issue = citation.volume
            if citation.issue:
                vol_issue += f", no. {citation.issue}"
        
        # Year and pages
        year = f"({citation.year})" if citation.year else ""
        pages = citation.pages if citation.pages else ""
        
        # DOI
        doi = f"https://doi.org/{citation.doi}" if citation.doi else ""
        
        # Assemble
        parts = [p for p in [authors, title, journal, vol_issue, year, pages] if p]
        result = ". ".join(parts) + "."
        if doi:
            result += f" {doi}."
        
        return result
