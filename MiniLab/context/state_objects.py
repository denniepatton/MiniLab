"""
State Objects for structured context management.

These canonical state objects replace raw conversation history
and provide structured, compressible context for agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import json


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class Priority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Task:
    """A single task in the project."""
    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    assigned_to: Optional[str] = None  # Agent ID
    depends_on: list[str] = field(default_factory=list)  # Task IDs
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    notes: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)  # File paths
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "assigned_to": self.assigned_to,
            "depends_on": self.depends_on,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "notes": self.notes,
            "outputs": self.outputs,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Task:
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            status=TaskStatus(data.get("status", "pending")),
            priority=Priority(data.get("priority", "medium")),
            assigned_to=data.get("assigned_to"),
            depends_on=data.get("depends_on", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            notes=data.get("notes", []),
            outputs=data.get("outputs", []),
        )
    
    def to_summary(self) -> str:
        """Compact string representation for context."""
        status_emoji = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.IN_PROGRESS: "🔄",
            TaskStatus.BLOCKED: "🚫",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
        }
        return f"{status_emoji[self.status]} [{self.id}] {self.title} ({self.assigned_to or 'unassigned'})"


@dataclass
class TaskState:
    """
    Rolling task state - the short-term memory of current work.
    
    Target: ~1000 tokens, refreshed via summarization.
    """
    current_task: Optional[Task] = None
    recent_decisions: list[str] = field(default_factory=list)  # Last 5 decisions
    key_facts: list[str] = field(default_factory=list)  # Important discoveries
    constraints: list[str] = field(default_factory=list)  # Active constraints
    blockers: list[str] = field(default_factory=list)  # Current blockers
    next_steps: list[str] = field(default_factory=list)  # Immediate next actions
    
    MAX_DECISIONS = 5
    MAX_FACTS = 10
    MAX_CONSTRAINTS = 5
    
    def add_decision(self, decision: str) -> None:
        """Add a decision, maintaining max count."""
        self.recent_decisions.append(decision)
        if len(self.recent_decisions) > self.MAX_DECISIONS:
            self.recent_decisions = self.recent_decisions[-self.MAX_DECISIONS:]
    
    def add_fact(self, fact: str) -> None:
        """Add a key fact, maintaining max count."""
        self.key_facts.append(fact)
        if len(self.key_facts) > self.MAX_FACTS:
            self.key_facts = self.key_facts[-self.MAX_FACTS:]
    
    def to_context(self) -> str:
        """Generate context string (~1000 tokens target)."""
        lines = ["## Current Task State"]
        
        if self.current_task:
            lines.append(f"\n**Current Task:** {self.current_task.to_summary()}")
            lines.append(f"Description: {self.current_task.description[:200]}...")
        
        if self.recent_decisions:
            lines.append("\n**Recent Decisions:**")
            for d in self.recent_decisions:
                lines.append(f"- {d[:100]}")
        
        if self.key_facts:
            lines.append("\n**Key Facts:**")
            for f in self.key_facts:
                lines.append(f"- {f[:100]}")
        
        if self.constraints:
            lines.append("\n**Constraints:**")
            for c in self.constraints:
                lines.append(f"- {c[:80]}")
        
        if self.blockers:
            lines.append("\n**Blockers:**")
            for b in self.blockers:
                lines.append(f"- ⚠️ {b[:80]}")
        
        if self.next_steps:
            lines.append("\n**Next Steps:**")
            for i, s in enumerate(self.next_steps[:5], 1):
                lines.append(f"{i}. {s[:80]}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            "current_task": self.current_task.to_dict() if self.current_task else None,
            "recent_decisions": self.recent_decisions,
            "key_facts": self.key_facts,
            "constraints": self.constraints,
            "blockers": self.blockers,
            "next_steps": self.next_steps,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> TaskState:
        return cls(
            current_task=Task.from_dict(data["current_task"]) if data.get("current_task") else None,
            recent_decisions=data.get("recent_decisions", []),
            key_facts=data.get("key_facts", []),
            constraints=data.get("constraints", []),
            blockers=data.get("blockers", []),
            next_steps=data.get("next_steps", []),
        )


@dataclass
class ConversationSummary:
    """
    Compressed summary of conversation history.
    
    Used instead of raw dialogue to reduce token usage.
    """
    summary: str  # High-level summary
    key_exchanges: list[dict] = field(default_factory=list)  # Important Q&A pairs
    action_items: list[str] = field(default_factory=list)  # Things to do
    agreements: list[str] = field(default_factory=list)  # Things agreed upon
    open_questions: list[str] = field(default_factory=list)  # Unresolved questions
    
    def to_context(self) -> str:
        """Generate context string."""
        lines = ["## Conversation Summary", "", self.summary]
        
        if self.agreements:
            lines.append("\n**Agreements:**")
            for a in self.agreements:
                lines.append(f"- {a}")
        
        if self.action_items:
            lines.append("\n**Action Items:**")
            for a in self.action_items:
                lines.append(f"- [ ] {a}")
        
        if self.open_questions:
            lines.append("\n**Open Questions:**")
            for q in self.open_questions:
                lines.append(f"- {q}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "key_exchanges": self.key_exchanges,
            "action_items": self.action_items,
            "agreements": self.agreements,
            "open_questions": self.open_questions,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> ConversationSummary:
        return cls(
            summary=data.get("summary", ""),
            key_exchanges=data.get("key_exchanges", []),
            action_items=data.get("action_items", []),
            agreements=data.get("agreements", []),
            open_questions=data.get("open_questions", []),
        )


@dataclass
class DataFile:
    """Description of a data file in the project."""
    path: str
    filename: str
    description: str
    file_type: str  # csv, json, etc.
    size_bytes: int
    rows: Optional[int] = None
    columns: Optional[list[str]] = None
    sample_data: Optional[str] = None  # First few rows as string
    notes: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "filename": self.filename,
            "description": self.description,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "columns": self.columns,
            "sample_data": self.sample_data,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> DataFile:
        return cls(**data)
    
    def to_summary(self) -> str:
        """Compact string representation."""
        cols = f", {len(self.columns)} cols" if self.columns else ""
        rows = f"{self.rows} rows" if self.rows else ""
        return f"- **{self.filename}**: {self.description} ({rows}{cols})"


@dataclass
class DataManifest:
    """
    Manifest of all data files in the project.
    
    Human-readable summary of available data.
    """
    project_name: str
    data_directory: str
    files: list[DataFile] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    notes: str = ""
    
    def to_context(self) -> str:
        """Generate context string for agents."""
        lines = [
            "## Data Manifest",
            f"Project: {self.project_name}",
            f"Data Directory: {self.data_directory}",
            f"Files: {len(self.files)}",
            "",
        ]
        
        for f in self.files:
            lines.append(f.to_summary())
        
        if self.notes:
            lines.append(f"\n**Notes:** {self.notes}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "data_directory": self.data_directory,
            "files": [f.to_dict() for f in self.files],
            "created_at": self.created_at.isoformat(),
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> DataManifest:
        return cls(
            project_name=data["project_name"],
            data_directory=data["data_directory"],
            files=[DataFile.from_dict(f) for f in data.get("files", [])],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            notes=data.get("notes", ""),
        )
    
    def save(self, path: str) -> None:
        """Save manifest to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> DataManifest:
        """Load manifest from file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class PlanSection:
    """A section of a working or execution plan."""
    title: str
    content: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "dependencies": self.dependencies,
            "outputs": self.outputs,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> PlanSection:
        return cls(
            title=data["title"],
            content=data["content"],
            status=TaskStatus(data.get("status", "pending")),
            assigned_to=data.get("assigned_to"),
            dependencies=data.get("dependencies", []),
            outputs=data.get("outputs", []),
        )


@dataclass
class WorkingPlan:
    """
    High-level working plan created by Planning Committee.
    
    This is the meta-plan that guides the project.
    """
    project_name: str
    objective: str
    hypothesis: Optional[str] = None
    approach: str = ""
    sections: list[PlanSection] = field(default_factory=list)
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    contributors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    
    def to_context(self) -> str:
        """Generate context string for agents."""
        lines = [
            "## Working Plan",
            f"**Project:** {self.project_name} (v{self.version})",
            f"**Objective:** {self.objective}",
        ]
        
        if self.hypothesis:
            lines.append(f"**Hypothesis:** {self.hypothesis}")
        
        if self.approach:
            lines.append(f"\n**Approach:**\n{self.approach}")
        
        if self.sections:
            lines.append("\n**Sections:**")
            for s in self.sections:
                status = "✅" if s.status == TaskStatus.COMPLETED else "⏳"
                lines.append(f"\n### {status} {s.title}")
                lines.append(s.content[:500] + "..." if len(s.content) > 500 else s.content)
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "objective": self.objective,
            "hypothesis": self.hypothesis,
            "approach": self.approach,
            "sections": [s.to_dict() for s in self.sections],
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "contributors": self.contributors,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> WorkingPlan:
        return cls(
            project_name=data["project_name"],
            objective=data["objective"],
            hypothesis=data.get("hypothesis"),
            approach=data.get("approach", ""),
            sections=[PlanSection.from_dict(s) for s in data.get("sections", [])],
            version=data.get("version", 1),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
            created_by=data.get("created_by", ""),
            contributors=data.get("contributors", []),
            notes=data.get("notes", []),
        )
    
    def save(self, path: str) -> None:
        """Save plan to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> WorkingPlan:
        """Load plan from file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class ExecutionStep:
    """A single step in the execution plan."""
    id: str
    description: str
    code_file: Optional[str] = None
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    error_message: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "code_file": self.code_file,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "status": self.status.value,
            "error_message": self.error_message,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> ExecutionStep:
        return cls(
            id=data["id"],
            description=data["description"],
            code_file=data.get("code_file"),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            status=TaskStatus(data.get("status", "pending")),
            error_message=data.get("error_message"),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
        )


@dataclass
class ExecutionPlan:
    """
    Concrete execution plan created by Dayhoff.
    
    Translates the WorkingPlan into specific code steps.
    """
    project_name: str
    working_plan_version: int
    steps: list[ExecutionStep] = field(default_factory=list)
    environment_requirements: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_context(self) -> str:
        """Generate context string for agents."""
        lines = [
            "## Execution Plan",
            f"Project: {self.project_name}",
            f"Based on Working Plan v{self.working_plan_version}",
            f"Steps: {len(self.steps)}",
            "",
        ]
        
        for step in self.steps:
            status = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.BLOCKED: "🚫",
            }.get(step.status, "?")
            
            lines.append(f"{status} **Step {step.id}:** {step.description}")
            if step.code_file:
                lines.append(f"   Code: {step.code_file}")
            if step.outputs:
                lines.append(f"   Outputs: {', '.join(step.outputs)}")
            if step.error_message:
                lines.append(f"   ⚠️ Error: {step.error_message}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "working_plan_version": self.working_plan_version,
            "steps": [s.to_dict() for s in self.steps],
            "environment_requirements": self.environment_requirements,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> ExecutionPlan:
        return cls(
            project_name=data["project_name"],
            working_plan_version=data["working_plan_version"],
            steps=[ExecutionStep.from_dict(s) for s in data.get("steps", [])],
            environment_requirements=data.get("environment_requirements", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )
    
    def save(self, path: str) -> None:
        """Save plan to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> ExecutionPlan:
        """Load plan from file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
    
    def get_next_step(self) -> Optional[ExecutionStep]:
        """Get the next pending step."""
        for step in self.steps:
            if step.status == TaskStatus.PENDING:
                return step
        return None
    
    def get_current_step(self) -> Optional[ExecutionStep]:
        """Get the currently in-progress step."""
        for step in self.steps:
            if step.status == TaskStatus.IN_PROGRESS:
                return step
        return None


@dataclass
class ProjectState:
    """
    Complete project state - the master state object.
    
    Aggregates all state objects for a project.
    """
    project_name: str
    project_dir: str
    workflow: str  # Current workflow type
    current_module: str  # Current workflow module
    
    # State objects
    data_manifest: Optional[DataManifest] = None
    working_plan: Optional[WorkingPlan] = None
    execution_plan: Optional[ExecutionPlan] = None
    task_state: TaskState = field(default_factory=TaskState)
    conversation_summary: ConversationSummary = field(default_factory=lambda: ConversationSummary(summary=""))
    
    # Tasks
    tasks: list[Task] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "project_dir": self.project_dir,
            "workflow": self.workflow,
            "current_module": self.current_module,
            "data_manifest": self.data_manifest.to_dict() if self.data_manifest else None,
            "working_plan": self.working_plan.to_dict() if self.working_plan else None,
            "execution_plan": self.execution_plan.to_dict() if self.execution_plan else None,
            "task_state": self.task_state.to_dict(),
            "conversation_summary": self.conversation_summary.to_dict(),
            "tasks": [t.to_dict() for t in self.tasks],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> ProjectState:
        return cls(
            project_name=data["project_name"],
            project_dir=data["project_dir"],
            workflow=data["workflow"],
            current_module=data["current_module"],
            data_manifest=DataManifest.from_dict(data["data_manifest"]) if data.get("data_manifest") else None,
            working_plan=WorkingPlan.from_dict(data["working_plan"]) if data.get("working_plan") else None,
            execution_plan=ExecutionPlan.from_dict(data["execution_plan"]) if data.get("execution_plan") else None,
            task_state=TaskState.from_dict(data["task_state"]) if data.get("task_state") else TaskState(),
            conversation_summary=ConversationSummary.from_dict(data["conversation_summary"]) if data.get("conversation_summary") else ConversationSummary(summary=""),
            tasks=[Task.from_dict(t) for t in data.get("tasks", [])],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )
    
    def save(self, path: Optional[str] = None) -> None:
        """Save state to file."""
        if path is None:
            path = f"{self.project_dir}/project_state.json"
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> ProjectState:
        """Load state from file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
