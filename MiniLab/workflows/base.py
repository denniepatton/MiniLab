"""
Base workflow module definition.

Provides abstract base class for all workflow modules with:
- Required inputs/outputs specification
- Primary agents assignment
- Execution protocol
- Checkpoint/restore capability
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import json
from pathlib import Path
from datetime import datetime


class WorkflowStatus(Enum):
    """Status of a workflow execution."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowResult:
    """Result from a workflow module execution."""
    status: WorkflowStatus
    outputs: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)  # File paths created
    summary: str = ""
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowCheckpoint:
    """Checkpoint for workflow state persistence."""
    workflow_name: str
    status: WorkflowStatus
    step_index: int
    state: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "workflow_name": self.workflow_name,
            "status": self.status.value,
            "step_index": self.step_index,
            "state": self.state,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowCheckpoint":
        return cls(
            workflow_name=data["workflow_name"],
            status=WorkflowStatus(data["status"]),
            step_index=data["step_index"],
            state=data["state"],
            timestamp=data.get("timestamp", ""),
        )


class WorkflowModule(ABC):
    """
    Abstract base class for workflow modules.
    
    Each workflow module encapsulates a distinct phase of analysis:
    - CONSULTATION: Initial user discussion to clarify goals
    - LITERATURE REVIEW: Background research and context gathering
    - PLANNING COMMITTEE: Multi-agent deliberation on approach
    - EXECUTE ANALYSIS: Implementation and computation
    - WRITE-UP RESULTS: Documentation and reporting
    - CRITICAL REVIEW: Quality assessment and iteration
    """
    
    # Module identification
    name: str = "base"
    description: str = "Base workflow module"
    
    # Input/output specifications
    required_inputs: list[str] = []
    optional_inputs: list[str] = []
    expected_outputs: list[str] = []
    
    # Agent assignments
    primary_agents: list[str] = []
    supporting_agents: list[str] = []
    
    def __init__(
        self,
        agents: dict,  # Dict of agent_name -> Agent instance
        context_manager: Any,  # ContextManager instance
        project_path: Path,
    ):
        """
        Initialize workflow module.
        
        Args:
            agents: Dictionary mapping agent names to Agent instances
            context_manager: ContextManager for RAG retrieval
            project_path: Path to project directory (in Sandbox/)
        """
        self.agents = agents
        self.context_manager = context_manager
        self.project_path = project_path
        self._status = WorkflowStatus.NOT_STARTED
        self._current_step = 0
        self._state: dict[str, Any] = {}
    
    @property
    def status(self) -> WorkflowStatus:
        return self._status
    
    def validate_inputs(self, inputs: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate that all required inputs are present.
        
        Returns:
            Tuple of (is_valid, list of missing inputs)
        """
        missing = []
        for req in self.required_inputs:
            if req not in inputs or inputs[req] is None:
                missing.append(req)
        return len(missing) == 0, missing
    
    @abstractmethod
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[WorkflowCheckpoint] = None,
    ) -> WorkflowResult:
        """
        Execute the workflow module.
        
        Args:
            inputs: Dictionary of input parameters
            checkpoint: Optional checkpoint to resume from
            
        Returns:
            WorkflowResult with outputs and status
        """
        pass
    
    def checkpoint(self) -> WorkflowCheckpoint:
        """
        Create a checkpoint of current workflow state.
        
        Returns:
            WorkflowCheckpoint that can be used to resume
        """
        return WorkflowCheckpoint(
            workflow_name=self.name,
            status=self._status,
            step_index=self._current_step,
            state=self._state.copy(),
        )
    
    def restore(self, checkpoint: WorkflowCheckpoint) -> None:
        """
        Restore workflow state from checkpoint.
        
        Args:
            checkpoint: Previously saved checkpoint
        """
        if checkpoint.workflow_name != self.name:
            raise ValueError(
                f"Checkpoint workflow '{checkpoint.workflow_name}' "
                f"does not match module '{self.name}'"
            )
        self._status = checkpoint.status
        self._current_step = checkpoint.step_index
        self._state = checkpoint.state.copy()
    
    def save_checkpoint(self, path: Optional[Path] = None) -> Path:
        """
        Save checkpoint to disk.
        
        Args:
            path: Optional custom path, defaults to project checkpoints dir
            
        Returns:
            Path to saved checkpoint file
        """
        if path is None:
            checkpoint_dir = self.project_path / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = checkpoint_dir / f"{self.name}_checkpoint.json"
        
        checkpoint = self.checkpoint()
        with open(path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, default=str)
        
        return path
    
    def load_checkpoint(self, path: Path) -> WorkflowCheckpoint:
        """
        Load checkpoint from disk.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Loaded WorkflowCheckpoint
        """
        with open(path) as f:
            data = json.load(f)
        return WorkflowCheckpoint.from_dict(data)
    
    async def _run_agent_task(
        self,
        agent_name: str,
        task: str,
        context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Run a task with a specific agent.
        
        Args:
            agent_name: Name of agent to use
            task: Task description
            context: Optional additional context
            
        Returns:
            Agent's response and any outputs
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not available in workflow")
        
        agent = self.agents[agent_name]
        
        # Build context from RAG if context_manager available
        rag_context = ""
        if self.context_manager:
            relevant = self.context_manager.retrieve(
                project_name=self.project_path.name,
                query=task,
                k=10,
            )
            if relevant:
                rag_context = "\n\n".join([
                    f"[{chunk.source}]: {chunk.content}"
                    for chunk in relevant
                ])
        
        # Combine contexts
        full_context = context or {}
        if rag_context:
            full_context["relevant_context"] = rag_context
        
        # Always include project path so agents know where to write files
        full_context["project_path"] = str(self.project_path)
        full_context["project_name"] = self.project_path.name
        
        # Execute agent task
        result = await agent.execute_task(
            task=task,
            context=full_context,
            project_name=self.project_path.name,
        )
        
        # Index result for future RAG retrieval
        if self.context_manager and result.get("response"):
            import hashlib
            doc_id = hashlib.md5(f"{self.name}/{agent_name}/{task[:50]}".encode()).hexdigest()[:12]
            self.context_manager.index_document(
                project_name=self.project_path.name,
                doc_id=doc_id,
                content=result["response"],
                metadata={
                    "agent": agent_name,
                    "task": task[:100],
                    "source": f"{self.name}/{agent_name}",
                },
            )
        
        return result
    
    def _log_step(self, message: str) -> None:
        """Log a workflow step for tracking."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{self.name}] {message}"
        
        # Append to workflow log
        log_path = self.project_path / "logs" / f"{self.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(log_entry + "\n")
