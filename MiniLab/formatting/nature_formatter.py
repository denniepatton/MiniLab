"""
Nature journal formatter for publication-ready documents.

Implements Nature Review Article formatting with:
- Arial font (sans-serif)
- Nature-compliant reference formatting
- Proper document structure (Abstract, Keywords, Sections, References)
- Figure/table legend formatting
- PDF generation with reportlab
"""

import re
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass


@dataclass
class NatureReference:
    """A single reference in Nature format."""
    number: int
    authors: List[str]
    year: int
    title: str
    journal: str
    volume: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    arxiv_id: Optional[str] = None
    
    def to_nature_string(self) -> str:
        """Render as Nature-formatted reference string."""
        # Format: Authors et al. (Year). Title. Journal Volume, Pages. DOI.
        
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        
        parts = [f"{authors_str} ({self.year})"]
        parts.append(f"{self.title}")
        
        journal_part = self.journal
        if self.volume:
            journal_part += f" {self.volume}"
        if self.pages:
            journal_part += f", {self.pages}"
        parts.append(f"<i>{journal_part}</i>")
        
        if self.doi:
            parts.append(f"https://doi.org/{self.doi}")
        elif self.pmid:
            parts.append(f"PMID: {self.pmid}")
        elif self.arxiv_id:
            parts.append(f"arXiv:{self.arxiv_id}")
        
        return " ".join(parts)


class NatureFormatter:
    """
    Format documents for Nature Review Articles.
    
    Handles:
    - Document structure (abstract, keywords, sections, references)
    - Citation integration (numbered, linked to reference list)
    - Figure/table legend formatting
    - Typography (Arial font, proper sizing and spacing)
    - PDF output with reportlab
    """
    
    # Nature Review Article requirements
    ABSTRACT_MIN_WORDS = 150
    ABSTRACT_MAX_WORDS = 250
    KEYWORDS_MIN = 5
    KEYWORDS_MAX = 8
    
    # Typography
    FONT_NAME = "Helvetica"  # Arial equivalent in reportlab
    FONT_SIZE_BODY = 12
    FONT_SIZE_HEADING1 = 18
    FONT_SIZE_HEADING2 = 14
    FONT_SIZE_HEADING3 = 12
    FONT_SIZE_CAPTION = 10
    
    LINE_SPACING = 1.5  # 1.5x spacing
    MARGIN_TOP = 72  # 1 inch
    MARGIN_BOTTOM = 72
    MARGIN_LEFT = 72
    MARGIN_RIGHT = 72
    
    def __init__(self):
        """Initialize formatter."""
        self.references: Dict[int, NatureReference] = {}
        self.citation_count = 0
    
    def parse_markdown_to_nature(self, markdown_text: str) -> Dict[str, Any]:
        """
        Parse a markdown literature review and extract Nature-formatted components.
        
        Args:
            markdown_text: Markdown text (from agent-generated literature review)
            
        Returns:
            Dict with components: abstract, keywords, sections, references
        """
        sections = {}
        references = {}
        citation_map = {}  # Maps original citation format to number
        
        # Extract abstract (if marked, or first paragraph)
        abstract_match = re.search(r"(?:^## Abstract|^# Abstract|^Abstract)\n\n(.*?)(?=\n## |\n# |\n\n)", markdown_text, re.MULTILINE | re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else ""
        
        # Extract keywords
        keywords_match = re.search(r"(?:^## Keywords|^Keywords)[\s:]*\n(.*?)(?=\n## |\n# |\n\n)", markdown_text, re.MULTILINE | re.DOTALL)
        keywords = []
        if keywords_match:
            keywords_text = keywords_match.group(1).strip()
            # Parse as comma-separated or bullet list
            keywords = [k.strip().lstrip("- ") for k in keywords_text.split(',') if k.strip()]
            if not keywords:
                keywords = [k.strip().lstrip("- ") for k in keywords_text.split('\n') if k.strip()]
        
        # Parse sections (h2 headers)
        section_pattern = r"^## (.*?)\n(.*?)(?=^## |\Z)"
        for match in re.finditer(section_pattern, markdown_text, re.MULTILINE | re.DOTALL):
            section_title = match.group(1).strip()
            section_content = match.group(2).strip()
            
            # Skip abstract/keywords sections
            if section_title.lower() in ["abstract", "keywords"]:
                continue
            
            sections[section_title] = section_content
        
        # Extract and number citations
        # Look for patterns like [Ref 1], [1], (Smith et al. 2020), etc.
        citation_patterns = [
            r"\[(\d+)\]",  # [1]
            r"\[Ref (\d+)\]",  # [Ref 1]
        ]
        
        citation_count = 1
        for pattern in citation_patterns:
            for match in re.finditer(pattern, markdown_text):
                original = match.group(0)
                if original not in citation_map:
                    citation_map[original] = f"[{citation_count}]"
                    citation_count += 1
        
        # Extract references from bibliography section
        bib_match = re.search(r"(?:^## References?|^References?)\n\n(.*?)(?=\Z)", markdown_text, re.MULTILINE | re.DOTALL)
        if bib_match:
            bib_text = bib_match.group(1).strip()
            
            # Parse numbered list format: 1. Author. Year. Title. Journal. DOI.
            ref_pattern = r"^(\d+)\.\s*(.*?)(?=^\d+\.|$)"
            for match in re.finditer(ref_pattern, bib_text, re.MULTILINE | re.DOTALL):
                ref_num = int(match.group(1))
                ref_text = match.group(2).strip()
                
                # Parse reference components
                ref_obj = self._parse_reference_string(ref_text, ref_num)
                if ref_obj:
                    references[ref_num] = ref_obj
        
        return {
            "abstract": abstract,
            "keywords": keywords,
            "sections": sections,
            "references": references,
            "citation_map": citation_map,
        }
    
    def _parse_reference_string(self, ref_text: str, number: int) -> Optional[NatureReference]:
        """Parse a reference string into components."""
        # Simple parser - handles: Author. (Year). Title. Journal Volume, Pages. DOI.
        
        try:
            # Extract year in parentheses
            year_match = re.search(r"\((\d{4})\)", ref_text)
            year = int(year_match.group(1)) if year_match else 0
            
            # Extract DOI
            doi_match = re.search(r"https://doi\.org/([\d.]+/.*?)(?:\s|$)", ref_text)
            doi = doi_match.group(1) if doi_match else None
            
            # Extract PMID
            pmid_match = re.search(r"PMID:\s*(\d+)", ref_text)
            pmid = pmid_match.group(1) if pmid_match else None
            
            # Extract arXiv
            arxiv_match = re.search(r"arXiv:(\d+\.\d+)", ref_text)
            arxiv_id = arxiv_match.group(1) if arxiv_match else None
            
            # Extract authors (first part before year)
            authors_match = re.match(r"^(.*?)\s*\(\d{4}\)", ref_text)
            authors_text = authors_match.group(1) if authors_match else ""
            authors = [a.strip() for a in authors_text.split(",") if a.strip()]
            
            # Extract title (between year and journal)
            title_match = re.search(r"\(\d{4}\)\.\s*(.*?)\.\s*[<i>]*", ref_text)
            title = title_match.group(1) if title_match else ""
            
            # Extract journal (after title, before volume/pages)
            journal_match = re.search(r"<i>(.*?)</i>|([A-Z][A-Za-z\s&]+?)(?:\s+\d+|,|\.|$)", ref_text)
            journal = journal_match.group(1) or journal_match.group(2) if journal_match else ""
            
            # Extract volume and pages
            volume_match = re.search(r"<i>.*?(\d+)</i>", ref_text)
            volume = volume_match.group(1) if volume_match else None
            
            pages_match = re.search(r",\s*(\d+[-–]\d+)", ref_text)
            pages = pages_match.group(1) if pages_match else None
            
            if not (authors or title or journal):
                return None
            
            return NatureReference(
                number=number,
                authors=authors,
                year=year,
                title=title,
                journal=journal,
                volume=volume,
                pages=pages,
                doi=doi,
                pmid=pmid,
                arxiv_id=arxiv_id,
            )
        except Exception:
            return None
    
    def validate_document_structure(self, parsed: Dict[str, Any]) -> List[str]:
        """
        Validate that parsed document meets Nature standards.
        
        Returns:
            List of issues found (empty if valid)
        """
        issues = []
        
        # Check abstract
        if not parsed.get("abstract"):
            issues.append("Missing abstract")
        elif len(parsed["abstract"].split()) < self.ABSTRACT_MIN_WORDS:
            issues.append(f"Abstract too short ({len(parsed['abstract'].split())} words, min {self.ABSTRACT_MIN_WORDS})")
        elif len(parsed["abstract"].split()) > self.ABSTRACT_MAX_WORDS:
            issues.append(f"Abstract too long ({len(parsed['abstract'].split())} words, max {self.ABSTRACT_MAX_WORDS})")
        
        # Check keywords
        keywords = parsed.get("keywords", [])
        if len(keywords) < self.KEYWORDS_MIN:
            issues.append(f"Too few keywords ({len(keywords)}, min {self.KEYWORDS_MIN})")
        elif len(keywords) > self.KEYWORDS_MAX:
            issues.append(f"Too many keywords ({len(keywords)}, max {self.KEYWORDS_MAX})")
        
        # Check sections
        if not parsed.get("sections"):
            issues.append("No main content sections found")
        
        # Check references
        if not parsed.get("references"):
            issues.append("No references found")
        elif len(parsed["references"]) < 10:
            issues.append(f"Too few references ({len(parsed['references'])}, minimum for comprehensive review is 10)")
        
        return issues
    
    def generate_pdf(self, parsed: Dict[str, Any], output_path: Path, title: str = "Literature Review") -> None:
        """
        Generate a PDF from parsed Nature-formatted content.
        
        Args:
            parsed: Output from parse_markdown_to_nature()
            output_path: Path to write PDF
            title: Document title
            
        Raises:
            RuntimeError: If reportlab is not available
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.pdfgen import canvas
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
            from reportlab.lib import colors
        except ImportError:
            raise RuntimeError(
                "reportlab is required for PDF generation. "
                "Install with: pip install reportlab>=4.0"
            )
        
        # Create PDF
        pdf = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            topMargin=self.MARGIN_TOP,
            bottomMargin=self.MARGIN_BOTTOM,
            leftMargin=self.MARGIN_LEFT,
            rightMargin=self.MARGIN_RIGHT,
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Create Nature-specific styles
        title_style = ParagraphStyle(
            "NatureTitle",
            parent=styles["Heading1"],
            fontName=self.FONT_NAME,
            fontSize=self.FONT_SIZE_HEADING1,
            textColor=colors.HexColor("#000000"),
            spaceAfter=12,
            alignment=1,  # Center
        )
        
        heading_style = ParagraphStyle(
            "NatureHeading",
            parent=styles["Heading2"],
            fontName=self.FONT_NAME,
            fontSize=self.FONT_SIZE_HEADING2,
            textColor=colors.HexColor("#000000"),
            spaceAfter=6,
            spaceBefore=12,
        )
        
        body_style = ParagraphStyle(
            "NatureBody",
            parent=styles["Normal"],
            fontName=self.FONT_NAME,
            fontSize=self.FONT_SIZE_BODY,
            leading=self.FONT_SIZE_BODY * self.LINE_SPACING,
            alignment=4,  # Justified
            spaceAfter=6,
        )
        
        # Title
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Abstract
        if parsed.get("abstract"):
            story.append(Paragraph("<b>Abstract</b>", heading_style))
            story.append(Paragraph(parsed["abstract"], body_style))
            story.append(Spacer(1, 0.1*inch))
        
        # Keywords
        if parsed.get("keywords"):
            keywords_str = ", ".join(parsed["keywords"])
            story.append(Paragraph(f"<b>Keywords:</b> {keywords_str}", body_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Main sections
        for section_title, section_content in parsed.get("sections", {}).items():
            story.append(Paragraph(section_title, heading_style))
            story.append(Paragraph(section_content, body_style))
            story.append(Spacer(1, 0.1*inch))
        
        # References
        if parsed.get("references"):
            story.append(PageBreak())
            story.append(Paragraph("References", heading_style))
            story.append(Spacer(1, 0.1*inch))
            
            for ref_num, ref in sorted(parsed["references"].items()):
                ref_text = ref.to_nature_string()
                story.append(Paragraph(f"<b>{ref_num}.</b> {ref_text}", body_style))
                story.append(Spacer(1, 0.05*inch))
        
        # Build PDF
        pdf.build(story)
