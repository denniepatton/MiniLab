"""
Project SSOT (Single Source of Truth) for MiniLab.

This module provides a centralized, authoritative representation of project state.
It replaces the scattered checkpoint files, session.json, task graphs, and various
workflow state files with a single coherent state object.

Design Philosophy:
- ONE authoritative state file per project (project_state.json)
- All components read from and write to this SSOT
- Agents receive consistent context derived from SSOT
- Human-readable transcript.md is the only other output

The SSOT contains:
- Project metadata (name, created, status)
- User request and approved scope
- Task plan and progress
- Access policy (what data/paths are in-scope)
- Token budget and usage
- Agent activity log (lightweight, not full conversations)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, List


class ProjectStatus(str, Enum):
    """Project lifecycle status."""
    PLANNING = "planning"           # Initial consultation
    IN_PROGRESS = "in_progress"     # Active execution
    PAUSED = "paused"               # User paused
    BUDGET_EXHAUSTED = "budget_exhausted"  # Out of tokens
    COMPLETED = "completed"         # Successfully finished
    FAILED = "failed"               # Error occurred


class TaskStatus(str, Enum):
    """Status of a planned task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class TaskPlan:
    """A planned task in the project."""
    id: str
    title: str
    description: str
    module: str  # Which workflow module handles this
    assigned_agent: str  # Primary agent
    status: TaskStatus = TaskStatus.PENDING
    estimated_tokens: int = 0
    actual_tokens: int = 0
    dependencies: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)  # Paths to created files
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "module": self.module,
            "assigned_agent": self.assigned_agent,
            "status": self.status.value,
            "estimated_tokens": self.estimated_tokens,
            "actual_tokens": self.actual_tokens,
            "dependencies": self.dependencies,
            "outputs": self.outputs,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskPlan":
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            module=data["module"],
            assigned_agent=data.get("assigned_agent", "bohr"),
            status=TaskStatus(data.get("status", "pending")),
            estimated_tokens=data.get("estimated_tokens", 0),
            actual_tokens=data.get("actual_tokens", 0),
            dependencies=data.get("dependencies", []),
            outputs=data.get("outputs", []),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
        )


@dataclass
class AccessPolicy:
    """
    Intent-derived access policy for the project.
    
    This specifies what data and paths are in-scope for this project,
    derived from the user's request during consultation.
    """
    # Paths that can be read (relative to workspace root)
    readable_paths: list[str] = field(default_factory=list)
    
    # Whether ReadData/ is in-scope (only if user mentions data analysis)
    readdata_allowed: bool = False
    
    # Specific data files mentioned by user
    data_files_in_scope: list[str] = field(default_factory=list)
    
    # Project's own sandbox path (always writable)
    project_path: str = ""
    
    def allows_read(self, path: str) -> bool:
        """Check if reading this path is allowed."""
        # Project's own path always allowed
        if self.project_path and path.startswith(self.project_path):
            return True
        
        # Check if ReadData is allowed
        if "ReadData" in path and not self.readdata_allowed:
            return False
        
        # Check specific allowlist
        for allowed in self.readable_paths:
            if path.startswith(allowed):
                return True
        
        return False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "readable_paths": self.readable_paths,
            "readdata_allowed": self.readdata_allowed,
            "data_files_in_scope": self.data_files_in_scope,
            "project_path": self.project_path,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AccessPolicy":
        return cls(
            readable_paths=data.get("readable_paths", []),
            readdata_allowed=data.get("readdata_allowed", False),
            data_files_in_scope=data.get("data_files_in_scope", []),
            project_path=data.get("project_path", ""),
        )
    
    @classmethod
    def from_user_request(cls, request: str, project_path: str) -> "AccessPolicy":
        """
        Derive access policy from user request.
        
        This is called during consultation to determine what data
        the user actually wants to work with.
        """
        request_lower = request.lower()
        
        policy = cls(project_path=project_path)
        
        # Check if user mentions data analysis
        data_keywords = [
            "data", "dataset", "csv", "analyze", "analysis",
            "readdata", "explore", "exploratory", "files",
        ]
        if any(kw in request_lower for kw in data_keywords):
            policy.readdata_allowed = True
            policy.readable_paths.append("ReadData/")
        
        # Check for specific paths mentioned
        if "readdata/" in request_lower:
            policy.readdata_allowed = True
            policy.readable_paths.append("ReadData/")
        
        # Literature reviews don't need ReadData
        lit_keywords = ["literature", "review", "papers", "pubmed", "arxiv", "research"]
        if any(kw in request_lower for kw in lit_keywords) and not policy.readdata_allowed:
            # Pure lit review - no data access
            pass
        
        return policy


@dataclass 
class BudgetState:
    """Token budget state for the project."""
    total_budget: int = 0
    tokens_used: int = 0
    estimated_remaining_work: int = 0
    
    # Per-module tracking
    module_usage: dict[str, int] = field(default_factory=dict)
    
    # Warnings issued
    warnings_issued: list[int] = field(default_factory=list)  # Percentage thresholds
    
    @property
    def remaining(self) -> int:
        return max(0, self.total_budget - self.tokens_used)
    
    @property
    def percent_used(self) -> float:
        if self.total_budget == 0:
            return 0.0
        return (self.tokens_used / self.total_budget) * 100
    
    @property
    def is_exhausted(self) -> bool:
        return self.tokens_used >= self.total_budget
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "tokens_used": self.tokens_used,
            "estimated_remaining_work": self.estimated_remaining_work,
            "module_usage": self.module_usage,
            "warnings_issued": self.warnings_issued,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetState":
        return cls(
            total_budget=data.get("total_budget", 0),
            tokens_used=data.get("tokens_used", 0),
            estimated_remaining_work=data.get("estimated_remaining_work", 0),
            module_usage=data.get("module_usage", {}),
            warnings_issued=data.get("warnings_issued", []),
        )


@dataclass
class ProjectSSOT:
    """
    Single Source of Truth for project state.
    
    This is the authoritative record of everything about a project.
    All components should read from and update this state.
    """
    # Identity
    project_name: str
    project_path: Path
    session_id: str
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Status
    status: ProjectStatus = ProjectStatus.PLANNING
    
    # User request
    user_request: str = ""
    scope_summary: str = ""  # Bohr's understanding
    scope_confirmed: bool = False
    
    # Task plan
    tasks: list[TaskPlan] = field(default_factory=list)
    current_task_id: Optional[str] = None
    
    # Access policy
    access_policy: AccessPolicy = field(default_factory=AccessPolicy)
    
    # Budget
    budget: BudgetState = field(default_factory=BudgetState)
    
    # Outputs
    deliverables: list[str] = field(default_factory=list)  # Final output files
    
    # Resume support
    continuation_plan: Optional[str] = None  # What to do if resumed
    
    def __post_init__(self):
        if isinstance(self.project_path, str):
            self.project_path = Path(self.project_path)
    
    @property
    def ssot_path(self) -> Path:
        """Path to the SSOT JSON file."""
        return self.project_path / "project_state.json"
    
    @property
    def transcript_path(self) -> Path:
        """Path to the human-readable transcript."""
        return self.project_path / "transcript.md"
    
    def get_current_task(self) -> Optional[TaskPlan]:
        """Get the currently active task."""
        if not self.current_task_id:
            return None
        for task in self.tasks:
            if task.id == self.current_task_id:
                return task
        return None
    
    def get_pending_tasks(self) -> list[TaskPlan]:
        """Get tasks that are ready to execute."""
        pending = []
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                # Check if dependencies are met
                deps_met = all(d in completed_ids for d in task.dependencies)
                if deps_met:
                    pending.append(task)
        
        return pending
    
    def start_task(self, task_id: str) -> Optional[TaskPlan]:
        """Mark a task as started."""
        for task in self.tasks:
            if task.id == task_id:
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now().isoformat()
                self.current_task_id = task_id
                self.touch()
                return task
        return None
    
    def complete_task(self, task_id: str, outputs: list[str] = None, tokens_used: int = 0) -> Optional[TaskPlan]:
        """Mark a task as completed."""
        for task in self.tasks:
            if task.id == task_id:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now().isoformat()
                task.actual_tokens = tokens_used
                if outputs:
                    task.outputs = outputs
                if self.current_task_id == task_id:
                    self.current_task_id = None
                self.touch()
                return task
        return None
    
    def fail_task(self, task_id: str, error: str) -> Optional[TaskPlan]:
        """Mark a task as failed."""
        for task in self.tasks:
            if task.id == task_id:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now().isoformat()
                task.error = error
                if self.current_task_id == task_id:
                    self.current_task_id = None
                self.touch()
                return task
        return None
    
    def touch(self) -> None:
        """Update the last-modified timestamp."""
        self.updated_at = datetime.now().isoformat()
    
    def update_budget(self, tokens_used: int, module: Optional[str] = None) -> None:
        """Update budget tracking."""
        self.budget.tokens_used = tokens_used
        if module:
            self.budget.module_usage[module] = self.budget.module_usage.get(module, 0) + tokens_used
        self.touch()
    
    def get_context_for_agents(self) -> str:
        """
        Generate context string for agent system prompts.
        
        This provides agents with consistent, up-to-date project state.
        """
        lines = [
            "## Project State (from SSOT)",
            "",
            f"**Project:** {self.project_name}",
            f"**Status:** {self.status.value}",
            "",
            "### User Request",
            self.user_request,
            "",
            "### Scope",
            self.scope_summary or "(Not yet confirmed)",
            "",
        ]
        
        # Budget status
        if self.budget.total_budget > 0:
            lines.extend([
                "### Budget Status",
                f"- **Used:** {self.budget.tokens_used:,} / {self.budget.total_budget:,} ({self.budget.percent_used:.1f}%)",
                f"- **Remaining:** {self.budget.remaining:,} tokens",
                "",
            ])
        
        # Task progress
        if self.tasks:
            completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
            lines.extend([
                "### Task Progress",
                f"- **Completed:** {completed}/{len(self.tasks)} tasks",
                "",
            ])
            
            # Current task
            current = self.get_current_task()
            if current:
                lines.extend([
                    f"**Current Task:** {current.title}",
                    f"- Module: {current.module}",
                    f"- Estimated: {current.estimated_tokens:,} tokens",
                    "",
                ])
            
            # Upcoming tasks
            pending = self.get_pending_tasks()
            if pending:
                lines.append("**Upcoming Tasks:**")
                for t in pending[:3]:
                    lines.append(f"- {t.title} ({t.module})")
                lines.append("")
        
        # Access policy
        lines.extend([
            "### Access Policy",
            f"- **ReadData allowed:** {'Yes' if self.access_policy.readdata_allowed else 'No'}",
            f"- **Project path:** {self.access_policy.project_path}",
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version": 1,
            "project_name": self.project_name,
            "project_path": str(self.project_path),
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status.value,
            "user_request": self.user_request,
            "scope_summary": self.scope_summary,
            "scope_confirmed": self.scope_confirmed,
            "tasks": [t.to_dict() for t in self.tasks],
            "current_task_id": self.current_task_id,
            "access_policy": self.access_policy.to_dict(),
            "budget": self.budget.to_dict(),
            "deliverables": self.deliverables,
            "continuation_plan": self.continuation_plan,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectSSOT":
        """Deserialize from dictionary."""
        ssot = cls(
            project_name=data["project_name"],
            project_path=Path(data["project_path"]),
            session_id=data["session_id"],
        )
        ssot.created_at = data.get("created_at", ssot.created_at)
        ssot.updated_at = data.get("updated_at", ssot.updated_at)
        ssot.status = ProjectStatus(data.get("status", "planning"))
        ssot.user_request = data.get("user_request", "")
        ssot.scope_summary = data.get("scope_summary", "")
        ssot.scope_confirmed = data.get("scope_confirmed", False)
        ssot.tasks = [TaskPlan.from_dict(t) for t in data.get("tasks", [])]
        ssot.current_task_id = data.get("current_task_id")
        ssot.access_policy = AccessPolicy.from_dict(data.get("access_policy", {}))
        ssot.budget = BudgetState.from_dict(data.get("budget", {}))
        ssot.deliverables = data.get("deliverables", [])
        ssot.continuation_plan = data.get("continuation_plan")
        return ssot
    
    def save(self) -> Path:
        """Save SSOT to disk."""
        self.touch()
        self.project_path.mkdir(parents=True, exist_ok=True)
        
        with open(self.ssot_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        return self.ssot_path
    
    @classmethod
    def load(cls, project_path: Path) -> "ProjectSSOT":
        """Load SSOT from disk."""
        ssot_path = project_path / "project_state.json"
        
        if not ssot_path.exists():
            raise FileNotFoundError(f"No SSOT found at {ssot_path}")
        
        with open(ssot_path) as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def create(
        cls,
        project_name: str,
        project_path: Path,
        user_request: str,
        session_id: Optional[str] = None,
    ) -> "ProjectSSOT":
        """
        Create a new SSOT for a project.
        
        This is called during session start to initialize project state.
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        ssot = cls(
            project_name=project_name,
            project_path=project_path,
            session_id=session_id,
            user_request=user_request,
        )
        
        # Derive initial access policy from request
        ssot.access_policy = AccessPolicy.from_user_request(
            user_request, 
            f"Sandbox/{project_name}"
        )
        
        return ssot


def get_ssot(project_path: Path) -> Optional[ProjectSSOT]:
    """
    Get SSOT for a project, loading from disk if it exists.
    
    Returns None if no SSOT exists.
    """
    try:
        return ProjectSSOT.load(project_path)
    except FileNotFoundError:
        return None
