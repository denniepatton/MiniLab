from __future__ import annotations

import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import random

from MiniLab.storage.state_store import StateStore, Citation
from MiniLab.tools.web_search import PubMedSearchTool, ArxivSearchTool


class DailyDigest:
    """
    Generates daily paper recommendations and literature scanning.
    Carroll (the librarian) uses this to suggest relevant papers.
    """

    def __init__(
        self,
        state_store: StateStore,
        topics: List[str] | None = None,
        arxiv_categories: List[str] | None = None,
    ):
        self.state_store = state_store
        self.topics = topics or []
        self.arxiv_categories = arxiv_categories or ["cs.LG", "q-bio", "stat.ML"]
        
        self.pubmed_tool = PubMedSearchTool()
        self.arxiv_tool = ArxivSearchTool()

    async def generate_daily_recommendations(
        self,
        project_id: Optional[str] = None,
        num_papers: int = 3,
    ) -> Dict[str, any]:
        """
        Generate daily paper recommendations.
        
        Args:
            project_id: Optional project context for personalized recommendations
            num_papers: Number of papers to recommend
            
        Returns:
            Dict with recommended papers and connection notes
        """
        recommendations = []
        
        # Load project context if provided
        project = None
        if project_id:
            project = self.state_store.load_project(project_id)
        
        # Get recent papers from key topics
        for topic in self.topics[:2]:  # Limit to prevent too many API calls
            try:
                # Search PubMed
                pubmed_result = await self.pubmed_tool.execute(
                    query=f"{topic} AND (\"last 7 days\"[PDat])",
                    max_results=5
                )
                
                if pubmed_result["status"] == "success":
                    for paper in pubmed_result.get("results", [])[:2]:
                        recommendations.append({
                            "source": "pubmed",
                            "topic": topic,
                            "title": paper["title"],
                            "authors": paper["authors"],
                            "journal": paper["journal"],
                            "pub_date": paper["pub_date"],
                            "pmid": paper["pmid"],
                            "doi": paper.get("doi", ""),
                            "connections": self._find_connections(paper, project) if project else []
                        })
                
                # Also check arXiv
                arxiv_result = await self.arxiv_tool.execute(
                    query=topic,
                    max_results=3
                )
                
                # Note: Would need proper parsing of arXiv results
                
            except Exception as e:
                print(f"Error fetching papers for topic {topic}: {e}")
        
        # Randomly select recommendations if we have more than requested
        if len(recommendations) > num_papers:
            recommendations = random.sample(recommendations, num_papers)
        
        return {
            "date": datetime.now().isoformat(),
            "project_id": project_id,
            "recommendations": recommendations,
            "topics_scanned": self.topics,
        }

    def _find_connections(self, paper: dict, project) -> List[str]:
        """
        Find connections between a new paper and existing project knowledge.
        This is a simple keyword-based approach; could be enhanced with embeddings.
        """
        connections = []
        paper_text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
        
        # Check against project citations
        for citation in project.citations.values():
            citation_text = (citation.title + " " + (citation.abstract or "")).lower()
            
            # Simple keyword overlap check
            paper_words = set(paper_text.split())
            citation_words = set(citation_text.split())
            
            # Find overlapping meaningful words (> 4 chars)
            overlap = paper_words & citation_words
            meaningful_overlap = [w for w in overlap if len(w) > 4]
            
            if len(meaningful_overlap) > 3:
                connections.append(
                    f"Related to {citation.key}: shared concepts {', '.join(list(meaningful_overlap)[:3])}"
                )
        
        # Check against project ideas
        for idea in project.ideas:
            idea_text = (idea["title"] + " " + idea["description"]).lower()
            idea_words = set(idea_text.split())
            paper_words = set(paper_text.split())
            
            overlap = paper_words & idea_words
            meaningful_overlap = [w for w in overlap if len(w) > 4]
            
            if len(meaningful_overlap) > 2:
                connections.append(
                    f"Relevant to idea '{idea['title']}': {', '.join(list(meaningful_overlap)[:2])}"
                )
        
        return connections

    async def scan_literature_updates(
        self,
        lookback_days: int = 7,
    ) -> Dict[str, any]:
        """
        Scan for new literature across all tracked topics.
        Returns a summary of new publications.
        """
        results = {
            "scan_date": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "topics": {},
            "total_papers": 0,
        }
        
        for topic in self.topics:
            try:
                pubmed_result = await self.pubmed_tool.execute(
                    query=f"{topic} AND (\"last {lookback_days} days\"[PDat])",
                    max_results=10
                )
                
                if pubmed_result["status"] == "success":
                    papers = pubmed_result.get("results", [])
                    results["topics"][topic] = {
                        "count": len(papers),
                        "top_papers": papers[:3],  # Top 3 for summary
                    }
                    results["total_papers"] += len(papers)
                    
            except Exception as e:
                results["topics"][topic] = {
                    "error": str(e)
                }
        
        return results

    def format_recommendation_email(self, recommendations: dict) -> str:
        """
        Format recommendations as a readable email/message.
        """
        lines = [
            "=" * 60,
            f"Daily Literature Digest - {datetime.now().strftime('%B %d, %Y')}",
            "=" * 60,
            "",
        ]
        
        if recommendations["project_id"]:
            lines.append(f"Project: {recommendations['project_id']}")
            lines.append("")
        
        lines.append(f"Today's Recommendations ({len(recommendations['recommendations'])} papers):")
        lines.append("")
        
        for i, paper in enumerate(recommendations["recommendations"], 1):
            lines.append(f"{i}. {paper['title']}")
            lines.append(f"   Authors: {', '.join(paper['authors'][:3])}")
            lines.append(f"   Journal: {paper.get('journal', 'N/A')}")
            lines.append(f"   Date: {paper.get('pub_date', 'N/A')}")
            
            if paper.get("doi"):
                lines.append(f"   DOI: {paper['doi']}")
            
            if paper.get("connections"):
                lines.append(f"   Connections to your work:")
                for conn in paper["connections"]:
                    lines.append(f"     • {conn}")
            
            lines.append("")
        
        lines.append("=" * 60)
        lines.append("Scanned topics: " + ", ".join(recommendations["topics_scanned"]))
        lines.append("=" * 60)
        
        return "\n".join(lines)
