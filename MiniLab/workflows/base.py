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
    
    # Budget allocation is now managed by BudgetManager singleton
    # These are kept as fallback defaults only
    @classmethod
    def get_budget_allocation(cls, workflow_name: str) -> float:
        """Get budget allocation from BudgetManager (single source of truth)."""
        try:
            from ..config.budget_manager import get_budget_manager
            wb = get_budget_manager().get_workflow_budget(workflow_name)
            if wb:
                return wb.allocation_percent
        except Exception:
            pass
        # Fallback defaults
        defaults = {
            "consultation": 0.05,
            "literature_review": 0.20,
            "planning_committee": 0.15,
            "execute_analysis": 0.35,
            "writeup_results": 0.15,
            "critical_review": 0.10,
        }
        return defaults.get(workflow_name, 0.10)
    
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
        self._workflow_budget: Optional[int] = None
        self._workflow_start_tokens: int = 0
    
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
    
    def _init_budget_tracking(self, session_budget: Optional[int] = None) -> None:
        """
        Initialize budget tracking for this workflow.
        
        Uses the centralized BudgetManager for dynamic allocation.
        """
        try:
            from ..core import get_token_account
            from ..config.budget_manager import get_budget_manager
            
            account = get_token_account()
            self._workflow_start_tokens = account.total_used
            
            # Get budget from centralized manager
            budget_mgr = get_budget_manager()
            wb = budget_mgr.get_workflow_budget(self.name)
            if wb:
                self._workflow_budget = wb.allocated_tokens
            elif session_budget:
                # Fallback to percentage-based allocation
                allocation = self.BUDGET_ALLOCATION.get(self.name, 0.15)
                self._workflow_budget = int(session_budget * allocation)
        except ImportError:
            pass
    
    def _check_workflow_budget(self) -> tuple[bool, float]:
        """
        Check if workflow is within its budget allocation.
        
        Returns:
            Tuple of (within_budget, percentage_of_workflow_budget_used)
        """
        try:
            from ..core import get_token_account
            account = get_token_account()
            
            workflow_used = account.total_used - self._workflow_start_tokens
            
            if self._workflow_budget:
                pct = (workflow_used / self._workflow_budget) * 100
                return pct < 100, pct
            
            return True, 0.0
        except ImportError:
            return True, 0.0
    
    def _get_budget_guidance(self) -> str:
        """
        Get budget guidance message for agents.
        
        Returns guidance text based on current budget status.
        Agents receive this information but decide how to act on it.
        """
        try:
            from ..config.budget_manager import get_budget_manager
            budget_mgr = get_budget_manager()
            return budget_mgr.get_workflow_guidance(self.name)
        except ImportError:
            pass
        
        # Fallback to simple percentage-based guidance
        within_budget, pct = self._check_workflow_budget()
        
        if pct >= 80:
            return f"\n⚠️ BUDGET: {pct:.0f}% of workflow allocation used. Prioritize core deliverables.\n"
        elif pct >= 60:
            return f"\n📊 Budget: {pct:.0f}% of workflow allocation used.\n"
        return ""
    
    def _record_workflow_usage(self) -> None:
        """Record final token usage for this workflow to the budget manager."""
        try:
            from ..core import get_token_account
            from ..config.budget_manager import get_budget_manager
            
            account = get_token_account()
            workflow_used = account.total_used - self._workflow_start_tokens
            
            budget_mgr = get_budget_manager()
            budget_mgr.record_usage(self.name, workflow_used)
        except ImportError:
            pass
    
    async def execute_autonomous(
        self,
        inputs: dict[str, Any],
        lead_agent: str,
        objective: str,
        supporting_context: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Execute workflow in autonomous mode - agent decides approach.
        
        Unlike structured execute(), this gives the lead agent full control
        to decide how to accomplish the objective. The agent can:
        - Skip, combine, or reorder steps as it sees fit
        - Consult colleagues when it decides to
        - Allocate its own time/tokens within budget
        - Decide when the task is complete
        
        This is the PREFERRED execution mode for maximum flexibility.
        
        Args:
            inputs: Workflow inputs (data paths, prior results, etc.)
            lead_agent: Agent who takes ownership of the workflow
            objective: High-level objective (what, not how)
            supporting_context: Additional context from prior workflows
            
        Returns:
            WorkflowResult with outputs determined by the agent
        """
        if lead_agent not in self.agents:
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=f"Lead agent '{lead_agent}' not available",
            )
        
        self._status = WorkflowStatus.IN_PROGRESS
        self._init_budget_tracking()
        self._log_step(f"Starting autonomous execution with {lead_agent}")
        
        # Build autonomous task prompt
        budget_guidance = self._get_budget_guidance()
        
        task_prompt = f"""## Autonomous Workflow Execution

You are the lead agent for the **{self.name}** workflow phase.

### Your Objective
{objective}

### Available Context
{supporting_context or 'No prior context provided.'}

### Available Inputs
```json
{json.dumps(inputs, indent=2, default=str)}
```

### Your Authority
As lead agent, you have FULL AUTONOMY to decide:
- How to approach this objective
- Which steps to take and in what order
- When to consult colleagues (you can call them directly)
- When to use tools vs. reason independently
- When the objective is sufficiently achieved

### Project Location
All outputs should go to: {self.project_path}

### Budget Context
{budget_guidance}

### Expected Deliverables
This workflow typically produces: {', '.join(self.expected_outputs) or 'results relevant to objective'}

### Execution Guidelines
1. Assess the objective and plan your approach
2. Execute your plan, adapting as you learn
3. Use tools to create real artifacts (code, reports, data)
4. Consult colleagues when their expertise adds value
5. Wrap up when you've achieved the objective or hit budget limits

Begin your work now. When complete, summarize your deliverables."""
        
        try:
            agent = self.agents[lead_agent]
            
            result = await agent.execute(
                task=task_prompt,
                project_name=self.project_path.name,
            )
            
            self._record_workflow_usage()
            
            # Extract artifacts from agent's outputs
            artifacts = result.outputs if hasattr(result, 'outputs') else []
            
            if result.status.value == "completed":
                return WorkflowResult(
                    status=WorkflowStatus.COMPLETED,
                    outputs={"agent_result": result.result},
                    artifacts=artifacts,
                    summary=result.result or "Workflow completed",
                )
            else:
                return WorkflowResult(
                    status=WorkflowStatus.FAILED,
                    error=result.error or "Autonomous execution did not complete",
                    summary=result.result or "",
                )
        
        except Exception as e:
            self._record_workflow_usage()
            self._log_step(f"Error in autonomous execution: {e}")
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=str(e),
            )
