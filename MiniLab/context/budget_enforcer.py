"""
BudgetEnforcer: Runtime budget enforcement with guardrails.

Provides:
- Token budget tracking at multiple granularities (run, workflow, task)
- Per-step limits (MAX_PROMPT_TOKENS_PER_STEP, MAX_TOOL_OUTPUT_CHARS)
- Automatic budget warnings and degradation
- Integration with OrchestratorRuntime
"""

from __future__ import annotations

__all__ = [
    "BudgetScope",
    "DegradationMode",
    "BudgetConfig",
    "ScopedBudget",
    "BudgetEnforcer",
]

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional
from pydantic import BaseModel, Field

from MiniLab.core.token_account import TokenAccount


class BudgetScope(str, Enum):
    """Scope levels for budget enforcement."""
    RUN = "run"
    WORKFLOW = "workflow"
    TASK = "task"
    STEP = "step"


class DegradationMode(str, Enum):
    """Mode degradation levels when budget is constrained."""
    NORMAL = "normal"
    CONSERVATIVE = "conservative"  # Reduce context, use cheaper models
    MINIMAL = "minimal"  # Essential operations only
    EMERGENCY = "emergency"  # Final outputs only


class BudgetConfig(BaseModel):
    """Configuration for budget enforcement."""

    # Global limits
    max_tokens_total: int = Field(default=100_000, description="Total budget for run")

    # Per-step limits
    max_prompt_tokens_per_step: int = Field(default=8_000, description="Max prompt per LLM call")
    max_completion_tokens_per_step: int = Field(default=4_000, description="Max completion per call")
    max_tool_output_chars: int = Field(default=50_000, description="Truncate tool outputs")

    # Allocation percentages (must sum to <= 100)
    planning_budget_pct: float = Field(default=15.0, description="Budget for planning phase")
    execution_budget_pct: float = Field(default=70.0, description="Budget for task execution")
    verification_budget_pct: float = Field(default=10.0, description="Budget for verification")
    reserve_budget_pct: float = Field(default=5.0, description="Emergency reserve")

    # Warning thresholds
    warning_thresholds: list[float] = Field(
        default=[0.5, 0.75, 0.9],
        description="Issue warnings at these fractions"
    )

    # Degradation thresholds
    conservative_threshold: float = Field(default=0.7, description="Enter conservative at this usage")
    minimal_threshold: float = Field(default=0.9, description="Enter minimal at this usage")
    emergency_threshold: float = Field(default=0.95, description="Enter emergency at this usage")

    model_config = {"extra": "forbid"}


@dataclass
class ScopedBudget:
    """Budget allocation for a specific scope."""
    scope: BudgetScope
    scope_id: str
    allocated: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return self.allocated - self.used

    @property
    def usage_fraction(self) -> float:
        if self.allocated == 0:
            return 0.0
        return self.used / self.allocated

    def can_spend(self, tokens: int) -> bool:
        return self.used + tokens <= self.allocated

    def spend(self, tokens: int) -> None:
        self.used += tokens


class BudgetEnforcer:
    """
    Enforces token budgets across multiple scopes.
    
    Integrates with TokenAccount for actual tracking while providing:
    - Hierarchical budget allocation (run > workflow > task)
    - Automatic degradation mode management
    - Per-step limit enforcement
    - Callback notifications
    """

    def __init__(
        self,
        config: BudgetConfig,
        token_account: Optional[TokenAccount] = None,
    ):
        """
        Initialize budget enforcer.
        
        Args:
            config: Budget configuration
            token_account: Optional existing TokenAccount (uses singleton if not provided)
        """
        self.config = config
        self.account = token_account or TokenAccount()

        # Set the global budget in the account
        self.account.set_budget(config.max_tokens_total)

        # Scoped budgets
        self._run_budget: Optional[ScopedBudget] = None
        self._workflow_budgets: dict[str, ScopedBudget] = {}
        self._task_budgets: dict[str, ScopedBudget] = {}

        # Current degradation mode
        self._mode: DegradationMode = DegradationMode.NORMAL

        # Callbacks
        self._on_mode_change: list[Callable[[DegradationMode, DegradationMode], None]] = []
        self._on_warning: list[Callable[[BudgetScope, str, float], None]] = []
        self._on_exceeded: list[Callable[[BudgetScope, str], None]] = []

        # Warnings already issued
        self._warnings_issued: set[tuple[str, float]] = set()

    def initialize_run(self, run_id: str) -> ScopedBudget:
        """Initialize budget for a new run."""
        self._run_budget = ScopedBudget(
            scope=BudgetScope.RUN,
            scope_id=run_id,
            allocated=self.config.max_tokens_total,
        )
        self._workflow_budgets.clear()
        self._task_budgets.clear()
        self._warnings_issued.clear()
        self._mode = DegradationMode.NORMAL
        return self._run_budget

    def allocate_workflow(self, workflow_id: str, budget_pct: float) -> ScopedBudget:
        """
        Allocate budget for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            budget_pct: Percentage of run budget to allocate
            
        Returns:
            ScopedBudget for the workflow
        """
        if not self._run_budget:
            raise RuntimeError("Must initialize run budget first")

        allocated = int(self._run_budget.remaining * (budget_pct / 100))
        budget = ScopedBudget(
            scope=BudgetScope.WORKFLOW,
            scope_id=workflow_id,
            allocated=allocated,
        )
        self._workflow_budgets[workflow_id] = budget
        return budget

    def allocate_task(
        self,
        task_id: str,
        workflow_id: str,
        budget_tokens: Optional[int] = None
    ) -> ScopedBudget:
        """
        Allocate budget for a task.
        
        Args:
            task_id: Task identifier
            workflow_id: Parent workflow identifier
            budget_tokens: Explicit token budget (defaults to fair share)
            
        Returns:
            ScopedBudget for the task
        """
        workflow = self._workflow_budgets.get(workflow_id)
        if not workflow:
            raise RuntimeError(f"Unknown workflow: {workflow_id}")

        if budget_tokens is None:
            # Default to 10% of workflow remaining
            budget_tokens = int(workflow.remaining * 0.1)

        budget = ScopedBudget(
            scope=BudgetScope.TASK,
            scope_id=task_id,
            allocated=min(budget_tokens, workflow.remaining),
        )
        self._task_budgets[task_id] = budget
        return budget

    def check_step(self, prompt_tokens: int) -> bool:
        """
        Check if a step is within per-step limits.
        
        Args:
            prompt_tokens: Estimated prompt tokens
            
        Returns:
            True if step is allowed
        """
        if prompt_tokens > self.config.max_prompt_tokens_per_step:
            return False
        return True

    def truncate_tool_output(self, output: str) -> str:
        """Truncate tool output to configured limit."""
        max_chars = self.config.max_tool_output_chars

        if len(output) <= max_chars:
            return output

        # Calculate sizes accounting for the truncation message
        truncation_msg = f"\n\n... [TRUNCATED: {len(output) - max_chars:,} chars omitted] ...\n\n"
        msg_len = len(truncation_msg)

        # Split remaining budget between head and tail
        available = max_chars - msg_len
        if available <= 0:
            # max_chars is too small, just truncate hard
            return output[:max_chars]

        head_size = available * 2 // 3  # Give more to the beginning
        tail_size = available - head_size

        return output[:head_size] + truncation_msg + output[-tail_size:]

    def can_spend(
        self,
        tokens: int,
        task_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> bool:
        """
        Check if spending is allowed at all relevant scopes.
        
        Args:
            tokens: Tokens to spend
            task_id: Optional task scope
            workflow_id: Optional workflow scope
            
        Returns:
            True if spending is allowed
        """
        # Check run budget
        if self._run_budget and not self._run_budget.can_spend(tokens):
            return False

        # Check workflow budget
        if workflow_id and workflow_id in self._workflow_budgets:
            if not self._workflow_budgets[workflow_id].can_spend(tokens):
                return False

        # Check task budget
        if task_id and task_id in self._task_budgets:
            if not self._task_budgets[task_id].can_spend(tokens):
                return False

        return True

    def spend(
        self,
        tokens: int,
        task_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> None:
        """
        Record token spending at all relevant scopes.
        
        Args:
            tokens: Tokens spent
            task_id: Optional task scope
            workflow_id: Optional workflow scope
        """
        # Update run budget
        if self._run_budget:
            self._run_budget.spend(tokens)
            self._check_warnings(self._run_budget)
            self._update_mode()

        # Update workflow budget
        if workflow_id and workflow_id in self._workflow_budgets:
            self._workflow_budgets[workflow_id].spend(tokens)
            self._check_warnings(self._workflow_budgets[workflow_id])

        # Update task budget
        if task_id and task_id in self._task_budgets:
            self._task_budgets[task_id].spend(tokens)

    def _check_warnings(self, budget: ScopedBudget) -> None:
        """Issue warnings if thresholds are crossed."""
        for threshold in self.config.warning_thresholds:
            key = (budget.scope_id, threshold)
            if key not in self._warnings_issued:
                if budget.usage_fraction >= threshold:
                    self._warnings_issued.add(key)
                    for callback in self._on_warning:
                        callback(budget.scope, budget.scope_id, budget.usage_fraction)

    def _update_mode(self) -> None:
        """Update degradation mode based on run budget usage."""
        if not self._run_budget:
            return

        usage = self._run_budget.usage_fraction
        new_mode = self._mode

        if usage >= self.config.emergency_threshold:
            new_mode = DegradationMode.EMERGENCY
        elif usage >= self.config.minimal_threshold:
            new_mode = DegradationMode.MINIMAL
        elif usage >= self.config.conservative_threshold:
            new_mode = DegradationMode.CONSERVATIVE
        else:
            new_mode = DegradationMode.NORMAL

        if new_mode != self._mode:
            old_mode = self._mode
            self._mode = new_mode
            for callback in self._on_mode_change:
                callback(old_mode, new_mode)

    @property
    def mode(self) -> DegradationMode:
        """Current degradation mode."""
        return self._mode

    @property
    def total_used(self) -> int:
        """Total tokens used in run."""
        return self._run_budget.used if self._run_budget else 0

    @property
    def total_remaining(self) -> int:
        """Total tokens remaining in run."""
        return self._run_budget.remaining if self._run_budget else 0

    def get_workflow_status(self, workflow_id: str) -> Optional[dict[str, Any]]:
        """Get budget status for a workflow."""
        budget = self._workflow_budgets.get(workflow_id)
        if not budget:
            return None
        return {
            "workflow_id": workflow_id,
            "allocated": budget.allocated,
            "used": budget.used,
            "remaining": budget.remaining,
            "usage_pct": budget.usage_fraction * 100,
        }

    def get_summary(self) -> dict[str, Any]:
        """Get complete budget summary."""
        return {
            "mode": self._mode.value,
            "run": {
                "allocated": self._run_budget.allocated if self._run_budget else 0,
                "used": self._run_budget.used if self._run_budget else 0,
                "remaining": self._run_budget.remaining if self._run_budget else 0,
                "usage_pct": (self._run_budget.usage_fraction * 100) if self._run_budget else 0,
            },
            "workflows": {
                wf_id: {
                    "allocated": b.allocated,
                    "used": b.used,
                    "remaining": b.remaining,
                    "usage_pct": b.usage_fraction * 100,
                }
                for wf_id, b in self._workflow_budgets.items()
            },
            "config": {
                "max_tokens_total": self.config.max_tokens_total,
                "max_prompt_per_step": self.config.max_prompt_tokens_per_step,
                "max_tool_output_chars": self.config.max_tool_output_chars,
            },
        }

    def on_mode_change(
        self,
        callback: Callable[[DegradationMode, DegradationMode], None]
    ) -> None:
        """Register callback for mode changes."""
        self._on_mode_change.append(callback)

    def on_warning(
        self,
        callback: Callable[[BudgetScope, str, float], None]
    ) -> None:
        """Register callback for budget warnings."""
        self._on_warning.append(callback)
