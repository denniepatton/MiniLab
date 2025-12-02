from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib


@dataclass
class Citation:
    """Represents a scientific citation with metadata."""
    key: str  # unique identifier (e.g., "Smith2020" or DOI)
    title: str
    authors: List[str]
    year: int
    journal: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    added_date: str = field(default_factory=lambda: datetime.now().isoformat())
    zotero_key: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Citation:
        return cls(**data)


@dataclass
class ConceptLink:
    """Represents a connection between two concepts or papers."""
    source: str
    target: str
    relation_type: str  # e.g., "cites", "extends", "contradicts", "inspired_by"
    description: str = ""
    strength: float = 1.0  # 0-1, for graph weighting
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ConceptLink:
        return cls(**data)


@dataclass
class ProjectState:
    """State for a specific research project."""
    project_id: str
    name: str
    description: str
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Citations specific to this project
    citations: Dict[str, Citation] = field(default_factory=dict)
    
    # Concept graph edges
    concept_links: List[ConceptLink] = field(default_factory=list)
    
    # Agent memory/notes specific to this project
    agent_notes: Dict[str, List[str]] = field(default_factory=dict)
    
    # Discussion history (meeting summaries, decisions)
    meeting_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Ideas, hypotheses, to-dos
    ideas: List[Dict[str, Any]] = field(default_factory=list)
    
    # Any custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_citation(self, citation: Citation):
        """Add or update a citation."""
        self.citations[citation.key] = citation
        self.last_modified = datetime.now().isoformat()

    def add_concept_link(self, link: ConceptLink):
        """Add a concept link to the graph."""
        self.concept_links.append(link)
        self.last_modified = datetime.now().isoformat()

    def add_agent_note(self, agent_id: str, note: str):
        """Add a note from a specific agent."""
        if agent_id not in self.agent_notes:
            self.agent_notes[agent_id] = []
        self.agent_notes[agent_id].append(note)
        self.last_modified = datetime.now().isoformat()

    def add_meeting_record(self, meeting_type: str, participants: List[str], 
                          summary: str, details: Dict[str, Any]):
        """Record a meeting."""
        self.meeting_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": meeting_type,
            "participants": participants,
            "summary": summary,
            "details": details,
        })
        self.last_modified = datetime.now().isoformat()

    def add_idea(self, title: str, description: str, source_agent: str, 
                 related_citations: List[str] = None):
        """Record an idea or hypothesis."""
        self.ideas.append({
            "id": hashlib.md5(f"{title}{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            "title": title,
            "description": description,
            "source_agent": source_agent,
            "related_citations": related_citations or [],
            "created_date": datetime.now().isoformat(),
            "status": "proposed",  # proposed, exploring, validated, rejected
        })
        self.last_modified = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "created_date": self.created_date,
            "last_modified": self.last_modified,
            "citations": {k: v.to_dict() for k, v in self.citations.items()},
            "concept_links": [link.to_dict() for link in self.concept_links],
            "agent_notes": self.agent_notes,
            "meeting_history": self.meeting_history,
            "ideas": self.ideas,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProjectState:
        """Deserialize from dictionary."""
        citations = {k: Citation.from_dict(v) for k, v in data.get("citations", {}).items()}
        concept_links = [ConceptLink.from_dict(link) for link in data.get("concept_links", [])]
        
        return cls(
            project_id=data["project_id"],
            name=data["name"],
            description=data["description"],
            created_date=data.get("created_date", datetime.now().isoformat()),
            last_modified=data.get("last_modified", datetime.now().isoformat()),
            citations=citations,
            concept_links=concept_links,
            agent_notes=data.get("agent_notes", {}),
            meeting_history=data.get("meeting_history", []),
            ideas=data.get("ideas", []),
            metadata=data.get("metadata", {}),
        )


class StateStore:
    """
    Manages persistent state for MiniLab projects.
    Stores project states, citations, and knowledge graphs on disk.
    """

    def __init__(self, storage_dir: pathlib.Path | str = None):
        if storage_dir is None:
            # Default to ~/.minilab/projects
            storage_dir = pathlib.Path.home() / ".minilab" / "projects"
        self.storage_dir = pathlib.Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Global bibliography (across all projects)
        self.global_bibliography_path = self.storage_dir.parent / "global_bibliography.json"
        self.global_bibliography: Dict[str, Citation] = self._load_global_bibliography()

    def _load_global_bibliography(self) -> Dict[str, Citation]:
        """Load the global bibliography file."""
        if self.global_bibliography_path.exists():
            with open(self.global_bibliography_path, 'r') as f:
                data = json.load(f)
                return {k: Citation.from_dict(v) for k, v in data.items()}
        return {}

    def _save_global_bibliography(self):
        """Save the global bibliography file."""
        data = {k: v.to_dict() for k, v in self.global_bibliography.items()}
        with open(self.global_bibliography_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_project_path(self, project_id: str) -> pathlib.Path:
        """Get the file path for a project."""
        return self.storage_dir / f"{project_id}.json"

    def load_project(self, project_id: str) -> Optional[ProjectState]:
        """Load a project state from disk."""
        path = self.get_project_path(project_id)
        if not path.exists():
            return None
        
        with open(path, 'r') as f:
            data = json.load(f)
            return ProjectState.from_dict(data)

    def save_project(self, project: ProjectState):
        """Save a project state to disk."""
        path = self.get_project_path(project.project_id)
        with open(path, 'w') as f:
            json.dump(project.to_dict(), f, indent=2)
        
        # Also update global bibliography with any new citations
        for citation in project.citations.values():
            if citation.key not in self.global_bibliography:
                self.global_bibliography[citation.key] = citation
        self._save_global_bibliography()

    def create_project(self, project_id: str, name: str, description: str) -> ProjectState:
        """Create a new project."""
        project = ProjectState(
            project_id=project_id,
            name=name,
            description=description,
        )
        self.save_project(project)
        return project

    def list_projects(self) -> List[Dict[str, str]]:
        """List all projects."""
        projects = []
        for path in self.storage_dir.glob("*.json"):
            project = self.load_project(path.stem)
            if project:
                projects.append({
                    "project_id": project.project_id,
                    "name": project.name,
                    "description": project.description,
                    "last_modified": project.last_modified,
                })
        return projects

    def delete_project(self, project_id: str):
        """Delete a project."""
        path = self.get_project_path(project_id)
        if path.exists():
            path.unlink()

    def search_citations(self, query: str, project_id: Optional[str] = None) -> List[Citation]:
        """
        Search citations by keyword in title, authors, or abstract.
        If project_id is provided, search only that project; otherwise search globally.
        """
        query_lower = query.lower()
        results = []
        
        if project_id:
            project = self.load_project(project_id)
            if project:
                citations = project.citations.values()
            else:
                citations = []
        else:
            citations = self.global_bibliography.values()
        
        for citation in citations:
            if (query_lower in citation.title.lower() or
                any(query_lower in author.lower() for author in citation.authors) or
                (citation.abstract and query_lower in citation.abstract.lower()) or
                any(query_lower in tag.lower() for tag in citation.tags)):
                results.append(citation)
        
        return results

    def export_bibliography(self, project_id: str, format: str = "bibtex") -> str:
        """
        Export bibliography in various formats.
        Currently only supports basic representations; full BibTeX would require additional libraries.
        """
        project = self.load_project(project_id)
        if not project:
            return ""
        
        if format == "bibtex":
            # Simplified BibTeX export
            entries = []
            for citation in project.citations.values():
                authors_str = " and ".join(citation.authors)
                entry = f"""@article{{{citation.key},
  title={{{citation.title}}},
  author={{{authors_str}}},
  year={{{citation.year}}},
  journal={{{citation.journal or ""}}},
  doi={{{citation.doi or ""}}}
}}"""
                entries.append(entry)
            return "\n\n".join(entries)
        
        elif format == "json":
            return json.dumps(
                {k: v.to_dict() for k, v in project.citations.items()},
                indent=2
            )
        
        else:
            raise ValueError(f"Unsupported format: {format}")
