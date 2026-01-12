"""
ProjectWriter: Single source of truth for project outputs.

Mirrors VSCode-style architecture where:
- Python code manages structure and metadata
- Agents provide content but don't create arbitrary files
- All dates come from the system, never the LLM
- Duplicate/redundant files are prevented

Canonical Document Philosophy:
- ALWAYS created: project_specification.md, session_summary.md, checkpoints/
- CONDITIONALLY created: literature/, analysis/, figures/, outputs/ (only if workflow runs)
- NEVER created: executive_summary.md, brief_bibliography.md, etc.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..context import ContextManager


class ProjectWriter:
    """
    Centralized output management for MiniLab projects.
    
    This class owns the project file structure and ensures:
    - Consistent file organization
    - Correct timestamps (from system, not LLM)
    - No duplicate/redundant files
    - Proper metadata in all documents
    
    Canonical Document Rules:
    - ALWAYS: project_specification.md (consultation), session_summary.md (orchestrator)
    - CONDITIONAL: literature/* (only if lit review), analysis/* (only if analysis)
    - FORBIDDEN: Files that duplicate canonical outputs
    """
    
    # Canonical project structure - maps file pattern to creating workflow
    # ALWAYS created regardless of workflow:
    ALWAYS_FILES = {
        "project_specification.md": "consultation",
        "checkpoints/": "system",
    }
    
    # CONDITIONALLY created based on workflow:
    CONDITIONAL_FILES = {
        "literature/references.md": "literature_review",
        "literature/literature_summary.md": "literature_review",
        "analysis/": "execute_analysis",
        "figures/": "execute_analysis",
        "outputs/summary_report.md": "writeup_results",
        "data_manifest.md": "system",  # Only if data exists
    }
    
    # Combined for backward compatibility
    STRUCTURE = {**ALWAYS_FILES, **CONDITIONAL_FILES}
    
    # Files that should NEVER be created by agents (use canonical versions instead)
    FORBIDDEN_FILES = [
        "executive_summary.md",      # Use literature_summary.md
        "brief_bibliography.md",     # Use references.md
        "search_summary.md",         # Goes in transcript
        "literature_search_summary.md",
        "session_summary.md",        # Only orchestrator creates this
    ]
    
    def __init__(
        self, 
        project_path: Path,
        project_name: str,
        session_date: Optional[datetime] = None,
        context_manager: Optional[ContextManager] = None,
        auto_index: bool = True,
        max_index_bytes: int = 2_000_000,
    ):
        """
        Initialize ProjectWriter.
        
        Args:
            project_path: Path to project directory (e.g., Sandbox/my_project)
            project_name: Human-readable project name
            session_date: Session start date (defaults to now)
        """
        self.project_path = Path(project_path)
        self.project_name = project_name
        self.session_date = session_date or datetime.now()
        self._created_files: list[str] = []

        self.context_manager = context_manager
        self.auto_index = auto_index
        self.max_index_bytes = max_index_bytes
        
        # Ensure project directory exists
        self.project_path.mkdir(parents=True, exist_ok=True)

    def _should_index(self, path: Path) -> bool:
        if not self.auto_index or not self.context_manager:
            return False
        if not path.exists() or not path.is_file():
            return False
        try:
            if path.stat().st_size > self.max_index_bytes:
                return False
        except Exception:
            return False
        ext = path.suffix.lower()
        return ext in {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml", ".csv"} or ext == ""

    def _maybe_index(self, path: Path) -> None:
        if not self._should_index(path):
            return
        try:
            # ContextManager stores are keyed by Sandbox/<project_slug> directory name.
            self.context_manager.index_file(project_name=self.project_path.name, file_path=path)
        except Exception:
            pass
    
    @property
    def date_string(self) -> str:
        """Get formatted date string for documents."""
        return self.session_date.strftime("%B %d, %Y")
    
    @property
    def date_iso(self) -> str:
        """Get ISO formatted date."""
        return self.session_date.strftime("%Y-%m-%d")
    
    @property
    def timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _ensure_dir(self, filepath: Path) -> None:
        """Ensure directory exists for filepath."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def _add_header(self, title: str, author: str = "MiniLab") -> str:
        """Create standard document header."""
        return f"""# {title}

**Project:** {self.project_name}  
**Date:** {self.date_string}  
**Generated by:** {author}

---

"""
    
    def write_project_specification(self, content: str) -> Path:
        """Write project specification from consultation."""
        filepath = self.project_path / "project_specification.md"
        
        full_content = self._add_header("Project Specification", "MiniLab Agent Bohr")
        full_content += content
        
        filepath.write_text(full_content)
        self._created_files.append(str(filepath))
        self._maybe_index(filepath)
        return filepath
    
    def write_data_manifest(self, manifest: dict[str, Any]) -> Optional[Path]:
        """
        Write data manifest ONLY if data files exist.
        
        Returns None if no data files found.
        """
        files = manifest.get("files", [])
        if not files:
            return None  # Don't create file if no data
        
        filepath = self.project_path / "data_manifest.md"
        
        content = self._add_header("Data Manifest", "MiniLab")
        content += f"## Summary\n\n"
        content += f"- **Total Files:** {len(files)}\n"
        content += f"- **Total Rows:** {manifest.get('summary', {}).get('total_rows', 0):,}\n\n"
        
        content += "## Files\n\n"
        for f in files:
            content += f"### {f.get('name', 'Unknown')}\n\n"
            content += f"- **Path:** `{f.get('path', 'N/A')}`\n"
            content += f"- **Rows:** {f.get('rows', '?'):,}\n"
            content += f"- **Columns:** {f.get('columns', '?')}\n"
            if f.get("column_names"):
                cols = ", ".join(f["column_names"][:10])
                if len(f["column_names"]) > 10:
                    cols += f" (+{len(f['column_names'])-10} more)"
                content += f"- **Column Names:** {cols}\n"
            content += "\n"
        
        filepath.write_text(content)
        self._created_files.append(str(filepath))
        self._maybe_index(filepath)
        return filepath
    
    def write_literature_summary(self, content: str) -> Path:
        """Write literature review summary."""
        lit_dir = self.project_path / "literature"
        lit_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = lit_dir / "literature_summary.md"
        
        full_content = self._add_header("Literature Review", "MiniLab Agent Gould")
        full_content += content
        
        filepath.write_text(full_content)
        self._created_files.append(str(filepath))
        self._maybe_index(filepath)
        return filepath
    
    def write_bibliography(self, content: str) -> Path:
        """Write bibliography/references."""
        lit_dir = self.project_path / "literature"
        lit_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = lit_dir / "references.md"
        
        full_content = self._add_header("Bibliography", "MiniLab Agent Gould")
        full_content += content
        
        filepath.write_text(full_content)
        self._created_files.append(str(filepath))
        self._maybe_index(filepath)
        return filepath
    
    def write_literature_review_pdf(self, markdown_content: str, output_name: str = "literature_review.pdf") -> Optional[Path]:
        """
        Generate Nature-formatted PDF of literature review.
        
        This is MANDATORY for literature review outputs - no fallbacks.
        
        Args:
            markdown_content: The markdown content of the literature review
            output_name: Name of PDF file (relative to literature/)
            
        Returns:
            Path to generated PDF, or None if generation fails with error
            
        Raises:
            RuntimeError: If reportlab is not installed
        """
        from MiniLab.formats import NatureFormatter
        
        lit_dir = self.project_path / "literature"
        lit_dir.mkdir(parents=True, exist_ok=True)
        output_path = lit_dir / output_name
        
        formatter = NatureFormatter()
        
        # Parse markdown to Nature format
        parsed = formatter.parse_markdown_to_nature(markdown_content)
        
        # Validate document structure
        issues = formatter.validate_document_structure(parsed)
        if issues:
            # Log issues but continue (critical review should have caught these)
            print(f"Warning: Literature review has structural issues: {issues}")
        
        # Generate PDF (raises RuntimeError if reportlab unavailable)
        formatter.generate_pdf(
            parsed, 
            output_path,
            title=f"Literature Review: {self.project_name}"
        )
        
        self._created_files.append(str(output_path))
        return output_path
    
    def write_session_summary(
        self, 
        summary: str,
        session_id: str,
        started_at: str,
        completed_workflows: list[str],
        token_usage: dict[str, Any],
    ) -> Path:
        """
        Write session summary (orchestrator only).
        
        This is the ONLY place session_summary.md should be created.
        """
        filepath = self.project_path / "session_summary.md"
        
        content = self._add_header("Session Summary", "MiniLab Agent Bohr")
        
        # Session metadata
        content += f"**Session ID:** {session_id}  \n"
        content += f"**Started:** {started_at}  \n"
        content += f"**Completed:** {self.timestamp}  \n\n"
        
        # Token usage
        if token_usage:
            content += "## Resource Usage\n\n"
            content += f"- **Tokens Used:** {token_usage.get('total_used', 0):,}\n"
            if token_usage.get('budget'):
                content += f"- **Token Budget:** {token_usage['budget']:,}\n"
                content += f"- **Budget Used:** {token_usage.get('percentage_used', 0):.1f}%\n"
            content += f"- **Estimated Cost:** ${token_usage.get('estimated_cost', 0):.2f}\n\n"
        
        # Completed workflows
        if completed_workflows:
            content += "## Completed Workflows\n\n"
            for wf in completed_workflows:
                content += f"- {wf.replace('_', ' ').title()}\n"
            content += "\n"
        
        # Main summary
        content += "## Summary\n\n"
        content += summary
        
        filepath.write_text(content)
        self._created_files.append(str(filepath))
        self._maybe_index(filepath)
        return filepath

    def write_token_accounting(
        self,
        *,
        token_usage: dict[str, Any],
        transactions: list[Any],
        aggregates: dict[str, Any],
    ) -> Path:
        """Persist authoritative token accounting artifacts for calibration/debugging."""
        out_dir = self.project_path / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / "token_accounting.json"

        def _tx_to_dict(t: Any) -> dict[str, Any]:
            return {
                "timestamp": getattr(t, "timestamp", None).isoformat() if getattr(t, "timestamp", None) else None,
                "agent_id": getattr(t, "agent_id", None),
                "workflow": getattr(t, "workflow", None),
                "trigger": getattr(t, "trigger", None),
                "operation": getattr(t, "operation", None),
                "input_tokens": getattr(t, "input_tokens", 0),
                "output_tokens": getattr(t, "output_tokens", 0),
                "total_tokens": getattr(t, "total_tokens", 0),
                "balance_after": getattr(t, "balance_after", None),
            }

        payload = {
            "token_usage": token_usage,
            "transactions": [_tx_to_dict(t) for t in transactions],
            "aggregates": aggregates,
        }

        filepath.write_text(json.dumps(payload, indent=2, default=str))
        self._created_files.append(str(filepath))
        self._maybe_index(filepath)
        return filepath
    
    def write_summary_report(self, content: str) -> Path:
        """Write final summary report."""
        outputs_dir = self.project_path / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = outputs_dir / "summary_report.md"
        
        full_content = self._add_header("Summary Report", "MiniLab")
        full_content += content
        
        filepath.write_text(full_content)
        self._created_files.append(str(filepath))
        self._maybe_index(filepath)
        return filepath

    def write_summary_report_pdf(self, *, markdown_path: Optional[Path] = None) -> Optional[Path]:
        """Generate a PDF version of the canonical summary report.

        Returns None if PDF generation is unavailable.
        """
        try:
            from .pdf import PdfMetadata, markdown_file_to_pdf
        except Exception:
            return None

        md_path = Path(markdown_path) if markdown_path else (self.project_path / "outputs" / "summary_report.md")
        if not md_path.exists():
            return None

        pdf_path = md_path.with_suffix(".pdf")
        try:
            meta = PdfMetadata(
                title="Summary Report",
                project=self.project_name,
                date=self.date_string,
                generated_by="MiniLab",
            )
            out = markdown_file_to_pdf(markdown_path=md_path, output_path=pdf_path, meta=meta)
            self._created_files.append(str(out))
            return out
        except Exception:
            return None

    def write_task_graph_visuals(self, *, task_graph: Any, render_png: bool = True) -> dict[str, Path]:
        """Write TaskGraph DOT (and optionally PNG) into checkpoints/.

        This is used by the orchestrator so users always get a visual plan artifact.
        """
        try:
            from .task_graph import export_task_graph_visuals
        except Exception:
            return {}

        out_dir = self.project_path / "checkpoints"
        try:
            artifacts = export_task_graph_visuals(task_graph, out_dir=out_dir, render_png=render_png)
        except Exception:
            return {}

        for p in artifacts.values():
            self._created_files.append(str(p))
        return artifacts
    
    def append_to_file(self, relative_path: str, content: str) -> Path:
        """
        Append content to an existing file.
        
        Use for building up documents incrementally (e.g., references).
        """
        filepath = self.project_path / relative_path
        self._ensure_dir(filepath)
        
        existing = filepath.read_text() if filepath.exists() else ""
        filepath.write_text(existing + content)

        self._created_files.append(str(filepath))
        self._maybe_index(filepath)
        
        return filepath
    
    def get_created_files(self) -> list[str]:
        """Get list of files created by this writer."""
        return self._created_files.copy()
    
    def validate_agent_file(self, filename: str) -> tuple[bool, str]:
        """
        Check if an agent should be allowed to create this file.
        
        Returns (allowed, reason).
        """
        basename = Path(filename).name
        
        for forbidden in self.FORBIDDEN_FILES:
            if forbidden in basename.lower():
                return False, f"File '{basename}' should not be created by agents. Use the canonical structure."
        
        return True, "OK"
    
    def get_canonical_path(self, file_type: str) -> Path:
        """
        Get the canonical path for a type of output.
        
        Args:
            file_type: One of 'bibliography', 'literature_summary', 'project_spec', etc.
        
        Returns:
            Canonical Path for that file type
        """
        mappings = {
            "bibliography": self.project_path / "literature" / "references.md",
            "references": self.project_path / "literature" / "references.md",
            "literature_summary": self.project_path / "literature" / "literature_summary.md",
            "project_spec": self.project_path / "project_specification.md",
            "summary_report": self.project_path / "outputs" / "summary_report.md",
            "session_summary": self.project_path / "session_summary.md",
        }
        
        return mappings.get(file_type, self.project_path / file_type)
