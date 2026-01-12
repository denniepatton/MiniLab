"""
Base module definition for MiniLab.

Provides abstract base class for all modules with:
- Required inputs/outputs specification
- Primary agents assignment
- Execution protocol
- Checkpoint/restore capability

Terminology (aligned with minilab_outline.md):
- Task: A project-DAG node representing a user-meaningful milestone
- Module: A reusable procedure that composes tools and possibly agents
- Tool: An atomic, side-effectful capability with typed I/O
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import json
from pathlib import Path
from datetime import datetime


class ModuleStatus(Enum):
    """Status of a module execution."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ModuleType(Enum):
    """Type of module pattern."""
    LINEAR = "linear"  # Fixed ordered sequence of steps
    SUBGRAPH = "subgraph"  # Includes retries, branching, verification hooks


@dataclass
class ModuleResult:
    """Result from a module execution."""
    status: ModuleStatus
    outputs: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)  # File paths created
    summary: str = ""
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleCheckpoint:
    """Checkpoint for module state persistence."""
    module_name: str
    status: ModuleStatus
    step_index: int
    state: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "module_name": self.module_name,
            "status": self.status.value,
            "step_index": self.step_index,
            "state": self.state,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModuleCheckpoint":
        return cls(
            module_name=data["module_name"],
            status=ModuleStatus(data["status"]),
            step_index=data["step_index"],
            state=data["state"],
            timestamp=data.get("timestamp", ""),
        )


class Module(ABC):
    """
    Abstract base class for modules.
    
    A Module is a reusable procedure that composes tools (and possibly 
    multiple agents) to achieve a bounded subgoal. Modules can be:
    - LINEAR: A fixed ordered sequence of steps
    - SUBGRAPH: Includes retries, branching, and verification hooks
    
    Module Types (from outline):
    
    Coordination Modules:
    - CONSULTATION: Initial scope confirmation
    - TEAM_DISCUSSION: Multi-agent feedback
    - ONE_ON_ONE: Deep dive with specific expert
    - PLANNING: Full plan production
    - CORE_INPUT: Core subgroup answer
    
    Evidence & Writing Modules:
    - EVIDENCE_GATHERING: Search + evidence packets
    - WRITE_ARTIFACT: Mandatory write gateway
    - BUILD_REPORT: Assemble narrative outputs
    
    Execution & Verification Modules:
    - GENERATE_CODE: Produce runnable scripts
    - RUN_CHECKS: Tests/lint/smoke checks
    - SANITY_CHECK_DATA: Data validation
    - INTERPRET_STATS: Statistical interpretation
    - INTERPRET_PLOT: Visual plot interpretation
    - CITATION_CHECK: Citation integrity
    - FORMATTING_CHECK: Rubric compliance
    
    External Expert Module:
    - CONSULT_EXTERNAL_EXPERT: Strict-contract specialist consultation
    """
    
    # Module identification
    name: str = "base"
    description: str = "Base module"
    module_type: ModuleType = ModuleType.LINEAR
    
    # Input/output specifications
    required_inputs: list[str] = []
    optional_inputs: list[str] = []
    expected_outputs: list[str] = []
    
    # Agent assignments
    primary_agents: list[str] = []
    supporting_agents: list[str] = []
    
    # Budget guidance - informational hints, not hard limits
    @classmethod
    def get_budget_allocation(cls, module_name: str) -> float:
        """Get suggested budget allocation percentage (guidance only)."""
        defaults = {
            # Coordination modules
            "consultation": 0.05,
            "team_discussion": 0.10,
            "one_on_one": 0.05,
            "planning": 0.10,
            "core_input": 0.05,
            # Evidence & writing modules
            "evidence_gathering": 0.15,
            "write_artifact": 0.02,
            "build_report": 0.15,
            # Execution & verification modules
            "generate_code": 0.15,
            "run_checks": 0.03,
            "sanity_check_data": 0.05,
            "interpret_stats": 0.05,
            "interpret_plot": 0.03,
            "citation_check": 0.03,
            "formatting_check": 0.03,
            # Review modules
            "critical_review": 0.10,
            # External
            "consult_external_expert": 0.05,
            # Legacy names (for compatibility during transition)
            "literature_review": 0.20,
            "planning_committee": 0.15,
            "execute_analysis": 0.35,
            "writeup_results": 0.15,
        }
        return defaults.get(module_name, 0.10)
    
    def __init__(
        self,
        agents: dict,  # Dict of agent_name -> Agent instance
        context_manager: Any,  # ContextManager instance
        project_path: Path,
    ):
        """
        Initialize module.
        
        Args:
            agents: Dictionary mapping agent names to Agent instances
            context_manager: ContextManager for RAG retrieval
            project_path: Path to project directory (in Sandbox/)
        """
        self.agents = agents
        self.context_manager = context_manager
        self.project_path = project_path
        self._status = ModuleStatus.NOT_STARTED
        self._current_step = 0
        self._state: dict[str, Any] = {}
        self._module_budget: Optional[int] = None
        self._module_start_tokens: int = 0
    
    @property
    def status(self) -> ModuleStatus:
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
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """
        Execute the module.
        
        Args:
            inputs: Dictionary of input parameters
            checkpoint: Optional checkpoint to resume from
            
        Returns:
            ModuleResult with outputs and status
        """
        pass
    
    def checkpoint(self) -> ModuleCheckpoint:
        """
        Create a checkpoint of current module state.
        
        Returns:
            ModuleCheckpoint that can be used to resume
        """
        return ModuleCheckpoint(
            module_name=self.name,
            status=self._status,
            step_index=self._current_step,
            state=self._state.copy(),
        )
    
    def restore(self, checkpoint: ModuleCheckpoint) -> None:
        """
        Restore module state from checkpoint.
        
        Args:
            checkpoint: Previously saved checkpoint
        """
        if checkpoint.module_name != self.name:
            raise ValueError(
                f"Checkpoint module '{checkpoint.module_name}' "
                f"does not match module '{self.name}'"
            )
        self._status = checkpoint.status
        self._current_step = checkpoint.step_index
        self._state = checkpoint.state.copy()
    
    def save_checkpoint(self, path: Optional[Path] = None) -> Path:
        """
        Save checkpoint to disk.
        
        Args:
            path: Optional custom path, defaults to project logs dir
            
        Returns:
            Path to saved checkpoint file
        """
        if path is None:
            checkpoint_dir = self.project_path / "logs" / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = checkpoint_dir / f"{self.name}_checkpoint.json"
        
        checkpoint = self.checkpoint()
        with open(path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, default=str)
        
        return path
    
    def load_checkpoint(self, path: Path) -> ModuleCheckpoint:
        """
        Load checkpoint from disk.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Loaded ModuleCheckpoint
        """
        with open(path) as f:
            data = json.load(f)
        return ModuleCheckpoint.from_dict(data)
    
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
            raise ValueError(f"Agent '{agent_name}' not available in module")
        
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
        """Log a module step for tracking."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{self.name}] {message}"
        
        # Append to module log
        log_path = self.project_path / "logs" / f"{self.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(log_entry + "\n")
    
    def _init_budget_tracking(self, session_budget: Optional[int] = None) -> None:
        """
        Initialize budget tracking for this module.
        
        Uses percentage-based guidance (not hard limits).
        """
        try:
            from ..core import get_token_account
            
            account = get_token_account()
            self._module_start_tokens = account.total_used
            
            # Calculate suggested budget based on session budget
            if session_budget:
                allocation = self.get_budget_allocation(self.name)
                self._module_budget = int(session_budget * allocation)
        except ImportError:
            pass
    
    def _check_module_budget(self) -> tuple[bool, float]:
        """
        Check if module is within its budget allocation.
        
        Returns:
            Tuple of (within_budget, percentage_of_module_budget_used)
        """
        try:
            from ..core import get_token_account
            account = get_token_account()
            
            module_used = account.total_used - self._module_start_tokens
            
            if self._module_budget:
                pct = (module_used / self._module_budget) * 100
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
            ctx = budget_mgr.get_context()
            return ctx.to_prompt_text()
        except Exception:
            pass
        
        # Fallback to simple percentage-based guidance
        within_budget, pct = self._check_module_budget()
        
        if pct >= 80:
            return f"\n⚠️ BUDGET: {pct:.0f}% of module allocation used. Prioritize core deliverables.\n"
        elif pct >= 60:
            return f"\n📊 Budget: {pct:.0f}% of module allocation used.\n"
        return ""
    
    def _record_module_usage(self) -> None:
        """Record final token usage for this module to the budget manager."""
        try:
            from ..core import get_token_account
            from ..config.budget_manager import get_budget_manager
            
            account = get_token_account()
            module_used = account.total_used - self._module_start_tokens
            
            budget_mgr = get_budget_manager()
            budget_mgr.record_usage(self.name, module_used)
        except ImportError:
            pass
    
    async def execute_autonomous(
        self,
        inputs: dict[str, Any],
        lead_agent: str,
        objective: str,
        supporting_context: Optional[str] = None,
    ) -> ModuleResult:
        """
        Execute module in autonomous mode - agent decides approach.
        
        Unlike structured execute(), this gives the lead agent full control
        to decide how to accomplish the objective. The agent can:
        - Skip, combine, or reorder steps as it sees fit
        - Consult colleagues when it decides to
        - Allocate its own time/tokens within budget
        - Decide when the task is complete
        
        This is the PREFERRED execution mode for maximum flexibility.
        
        Args:
            inputs: Module inputs (data paths, prior results, etc.)
            lead_agent: Agent who takes ownership of the module
            objective: High-level objective (what, not how)
            supporting_context: Additional context from prior modules
            
        Returns:
            ModuleResult with outputs determined by the agent
        """
        if lead_agent not in self.agents:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Lead agent '{lead_agent}' not available",
            )
        
        self._status = ModuleStatus.IN_PROGRESS
        self._init_budget_tracking()
        self._log_step(f"Starting autonomous execution with {lead_agent}")
        
        # Build autonomous task prompt
        budget_guidance = self._get_budget_guidance()
        
        task_prompt = f"""## Autonomous Module Execution

You are the lead agent for the **{self.name}** module.

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
This module typically produces: {', '.join(self.expected_outputs) or 'results relevant to objective'}

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
            
            self._record_module_usage()
            
            # Extract artifacts from agent's outputs
            artifacts = result.outputs if hasattr(result, 'outputs') else []
            
            if result.status.value == "completed":
                return ModuleResult(
                    status=ModuleStatus.COMPLETED,
                    outputs={"agent_result": result.result},
                    artifacts=artifacts,
                    summary=result.result or "Module completed",
                    metadata={
                        **({"stop_reason": result.stop_reason} if getattr(result, "stop_reason", None) else {}),
                    },
                )

            # Budget exhaustion is a controlled stop
            if result.status.value == "budget_exhausted":
                return ModuleResult(
                    status=ModuleStatus.PAUSED,
                    outputs={"agent_result": result.result},
                    artifacts=artifacts,
                    summary=result.result or "Paused due to budget exhaustion",
                    metadata={"stop_reason": getattr(result, "stop_reason", "budget_exceeded")},
                )

            if result.status.value == "paused":
                return ModuleResult(
                    status=ModuleStatus.PAUSED,
                    outputs={"agent_result": result.result},
                    artifacts=artifacts,
                    summary=result.result or "Module paused",
                    metadata={
                        **({"stop_reason": result.stop_reason} if getattr(result, "stop_reason", None) else {}),
                    },
                )

            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=result.error or "Autonomous execution did not complete",
                summary=result.result or "",
                metadata={
                    **({"stop_reason": result.stop_reason} if getattr(result, "stop_reason", None) else {}),
                },
            )
        
        except Exception as e:
            self._record_module_usage()
            self._log_step(f"Error in autonomous execution: {e}")
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=str(e),
            )


# Backward compatibility aliases (will be removed after full migration)
WorkflowModule = Module
WorkflowResult = ModuleResult
WorkflowCheckpoint = ModuleCheckpoint
WorkflowStatus = ModuleStatus
