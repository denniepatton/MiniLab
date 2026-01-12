"""
Budget Isolation - Controlled budget allocation for colleague calls.

Provides mechanisms to give colleagues their own budget slice rather than
sharing the parent's pool. This enables:
- Fair budget distribution among agents
- Preventing one colleague from consuming entire budget
- Tracking per-colleague token usage
- Soft limits with warnings vs hard caps

Mirrors the concept of process resource isolation but for token budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from ..core.token_account import TokenAccount


class BudgetMode(Enum):
    """How budget is allocated to colleagues."""
    SHARED = "shared"           # Share parent's remaining budget (default)
    ISOLATED = "isolated"       # Get a fixed slice, cannot exceed
    PROPORTIONAL = "proportional"  # Get percentage of parent's remaining
    UNLIMITED = "unlimited"     # No budget constraints (for testing)


class BudgetEnforcementLevel(Enum):
    """How strictly budget limits are enforced."""
    SOFT = "soft"              # Warn but allow overage
    FIRM = "firm"              # Warn heavily, complete current task
    HARD = "hard"              # Stop immediately when exceeded


@dataclass
class BudgetSlice:
    """
    A budget allocation for a colleague call.
    
    Tracks allocated vs used tokens and enforces limits.
    """
    # Allocation
    allocated_tokens: int
    mode: BudgetMode = BudgetMode.ISOLATED
    enforcement: BudgetEnforcementLevel = BudgetEnforcementLevel.FIRM
    
    # Usage tracking
    used_input_tokens: int = 0
    used_output_tokens: int = 0
    
    # Metadata
    parent_agent_id: str = ""
    colleague_id: str = ""
    purpose: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    # State
    is_active: bool = True
    warning_issued: bool = False
    overage_allowed: bool = False
    
    @property
    def total_used(self) -> int:
        """Total tokens used."""
        return self.used_input_tokens + self.used_output_tokens
    
    @property
    def remaining(self) -> int:
        """Remaining tokens in allocation."""
        return max(0, self.allocated_tokens - self.total_used)
    
    @property
    def usage_percent(self) -> float:
        """Percentage of allocation used."""
        if self.allocated_tokens <= 0:
            return 100.0
        return (self.total_used / self.allocated_tokens) * 100
    
    @property
    def is_exceeded(self) -> bool:
        """Check if budget is exceeded."""
        return self.total_used > self.allocated_tokens
    
    @property
    def is_near_limit(self) -> bool:
        """Check if near budget limit (>80%)."""
        return self.usage_percent >= 80
    
    def debit(self, input_tokens: int, output_tokens: int) -> tuple[bool, str]:
        """
        Debit tokens from the slice.
        
        Returns:
            Tuple of (allowed, message)
        """
        self.used_input_tokens += input_tokens
        self.used_output_tokens += output_tokens
        
        if self.mode == BudgetMode.UNLIMITED:
            return True, ""
        
        if self.is_exceeded:
            if self.enforcement == BudgetEnforcementLevel.HARD:
                return False, f"Budget exceeded ({self.total_used:,}/{self.allocated_tokens:,} tokens)"
            
            if not self.warning_issued:
                self.warning_issued = True
                if self.enforcement == BudgetEnforcementLevel.FIRM:
                    return True, f"⚠️ Budget exceeded - complete current task and wrap up"
                else:
                    return True, f"⚠️ Budget exceeded - consider wrapping up"
            
            self.overage_allowed = True
            return True, ""
        
        if self.is_near_limit and not self.warning_issued:
            self.warning_issued = True
            pct = self.usage_percent
            return True, f"Budget at {pct:.0f}% - plan to wrap up soon"
        
        return True, ""
    
    def get_context_message(self) -> str:
        """Get budget context for injection into prompts."""
        if self.mode == BudgetMode.UNLIMITED:
            return ""
        
        pct = self.usage_percent
        remaining = self.remaining
        
        lines = [f"Budget: {pct:.0f}% used ({self.total_used:,}/{self.allocated_tokens:,} tokens)"]
        lines.append(f"Remaining: ~{remaining:,} tokens")
        
        if pct >= 90:
            lines.append("⚠️ Budget nearly exhausted. Finish current task and wrap up.")
        elif pct >= 75:
            lines.append("Budget constrained. Complete core deliverable without exploration.")
        elif pct >= 50:
            lines.append("Budget moderate. Focus on essential tasks.")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            "allocated_tokens": self.allocated_tokens,
            "mode": self.mode.value,
            "enforcement": self.enforcement.value,
            "used_input_tokens": self.used_input_tokens,
            "used_output_tokens": self.used_output_tokens,
            "parent_agent_id": self.parent_agent_id,
            "colleague_id": self.colleague_id,
            "purpose": self.purpose,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
            "warning_issued": self.warning_issued,
            "overage_allowed": self.overage_allowed,
        }


class BudgetAllocator:
    """
    Manages budget allocation for colleague calls.
    
    Tracks active slices and coordinates with the parent TokenAccount.
    """
    
    # Default allocations by colleague call mode
    MODE_ALLOCATIONS = {
        "quick": 5_000,      # Quick consultations
        "focused": 15_000,   # Focused work
        "detailed": 40_000,  # Detailed analysis
    }
    
    # Default proportional allocations (percentage of remaining)
    MODE_PROPORTIONS = {
        "quick": 0.05,
        "focused": 0.15,
        "detailed": 0.30,
    }
    
    def __init__(self):
        self._active_slices: dict[str, BudgetSlice] = {}  # slice_id -> BudgetSlice
        self._slice_stack: list[str] = []  # Stack for nested colleague calls
        self._completed_slices: list[BudgetSlice] = []
        
        # Callbacks
        self.on_slice_created: Optional[Callable[[BudgetSlice], None]] = None
        self.on_slice_exhausted: Optional[Callable[[BudgetSlice], None]] = None
        self.on_warning: Optional[Callable[[str, str], None]] = None  # (agent_id, message)
    
    def _generate_slice_id(self, parent_id: str, colleague_id: str) -> str:
        """Generate unique slice ID."""
        timestamp = datetime.now().strftime("%H%M%S")
        return f"{parent_id}>{colleague_id}@{timestamp}"
    
    def create_slice(
        self,
        parent_agent_id: str,
        colleague_id: str,
        mode: str = "focused",
        purpose: str = "",
        budget_mode: BudgetMode = BudgetMode.ISOLATED,
        enforcement: BudgetEnforcementLevel = BudgetEnforcementLevel.FIRM,
        custom_allocation: Optional[int] = None,
        parent_account: Optional[Any] = None,  # TokenAccount
    ) -> BudgetSlice:
        """
        Create a budget slice for a colleague call.
        
        Args:
            parent_agent_id: ID of the calling agent
            colleague_id: ID of the colleague being called
            mode: Call mode ("quick", "focused", "detailed")
            purpose: Description of what the colleague will do
            budget_mode: How to allocate (SHARED, ISOLATED, PROPORTIONAL)
            enforcement: How strictly to enforce
            custom_allocation: Override default allocation
            parent_account: Parent's TokenAccount for proportional mode
            
        Returns:
            BudgetSlice for the colleague
        """
        # Determine allocation
        if custom_allocation is not None:
            allocated = custom_allocation
        elif budget_mode == BudgetMode.PROPORTIONAL and parent_account:
            proportion = self.MODE_PROPORTIONS.get(mode, 0.15)
            remaining = parent_account.budget - parent_account.total_used if parent_account.budget else 100_000
            allocated = int(remaining * proportion)
        elif budget_mode == BudgetMode.SHARED and parent_account:
            # Shared mode: slice gets parent's remaining budget
            allocated = parent_account.budget - parent_account.total_used if parent_account.budget else 100_000
        elif budget_mode == BudgetMode.UNLIMITED:
            allocated = 1_000_000  # Effectively unlimited
        else:
            # Default isolated allocation based on mode
            allocated = self.MODE_ALLOCATIONS.get(mode, 15_000)
        
        slice_id = self._generate_slice_id(parent_agent_id, colleague_id)
        
        budget_slice = BudgetSlice(
            allocated_tokens=allocated,
            mode=budget_mode,
            enforcement=enforcement,
            parent_agent_id=parent_agent_id,
            colleague_id=colleague_id,
            purpose=purpose,
        )
        
        self._active_slices[slice_id] = budget_slice
        self._slice_stack.append(slice_id)
        
        if self.on_slice_created:
            self.on_slice_created(budget_slice)
        
        return budget_slice
    
    def get_active_slice(self, colleague_id: Optional[str] = None) -> Optional[BudgetSlice]:
        """
        Get the active budget slice.
        
        Args:
            colleague_id: Optional specific colleague to look up
            
        Returns:
            Active BudgetSlice or None
        """
        if not self._slice_stack:
            return None
        
        if colleague_id:
            for slice_id in reversed(self._slice_stack):
                if slice_id in self._active_slices:
                    s = self._active_slices[slice_id]
                    if s.colleague_id == colleague_id:
                        return s
            return None
        
        # Return top of stack
        current_id = self._slice_stack[-1]
        return self._active_slices.get(current_id)
    
    def close_slice(self, colleague_id: str) -> Optional[BudgetSlice]:
        """
        Close a budget slice when colleague finishes.
        
        Args:
            colleague_id: ID of the finishing colleague
            
        Returns:
            The closed BudgetSlice or None
        """
        # Find and remove from stack
        for i, slice_id in enumerate(self._slice_stack):
            if slice_id in self._active_slices:
                s = self._active_slices[slice_id]
                if s.colleague_id == colleague_id:
                    s.is_active = False
                    self._completed_slices.append(s)
                    del self._active_slices[slice_id]
                    self._slice_stack.pop(i)
                    return s
        
        return None
    
    def debit(
        self,
        input_tokens: int,
        output_tokens: int,
        agent_id: str,
    ) -> tuple[bool, str]:
        """
        Debit tokens from the active slice for an agent.
        
        Args:
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            agent_id: Agent doing the work
            
        Returns:
            Tuple of (allowed, warning_message)
        """
        budget_slice = self.get_active_slice(agent_id)
        
        if not budget_slice:
            return True, ""  # No slice = no restriction
        
        allowed, message = budget_slice.debit(input_tokens, output_tokens)
        
        if message and self.on_warning:
            self.on_warning(agent_id, message)
        
        if not allowed and self.on_slice_exhausted:
            self.on_slice_exhausted(budget_slice)
        
        return allowed, message
    
    def get_remaining(self, agent_id: str) -> int:
        """Get remaining tokens for an agent."""
        budget_slice = self.get_active_slice(agent_id)
        if budget_slice:
            return budget_slice.remaining
        return -1  # -1 indicates no slice
    
    def get_usage_summary(self) -> dict:
        """Get summary of all slice usage."""
        active = []
        for slice_id, s in self._active_slices.items():
            active.append({
                "id": slice_id,
                "colleague": s.colleague_id,
                "used": s.total_used,
                "allocated": s.allocated_tokens,
                "percent": s.usage_percent,
            })
        
        completed = []
        for s in self._completed_slices:
            completed.append({
                "colleague": s.colleague_id,
                "used": s.total_used,
                "allocated": s.allocated_tokens,
                "overage": s.overage_allowed,
            })
        
        return {
            "active": active,
            "completed": completed,
            "total_allocated": sum(s.allocated_tokens for s in self._active_slices.values()),
            "total_used": sum(s.total_used for s in self._active_slices.values()) + sum(s.total_used for s in self._completed_slices),
        }
    
    def to_dict(self) -> dict:
        return {
            "active_slices": {k: v.to_dict() for k, v in self._active_slices.items()},
            "completed_slices": [s.to_dict() for s in self._completed_slices],
            "slice_stack": self._slice_stack.copy(),
        }


@contextmanager
def budget_slice_context(
    parent_agent_id: str,
    colleague_id: str,
    allocator: BudgetAllocator,
    mode: str = "focused",
    **kwargs,
):
    """
    Context manager for budget-isolated colleague calls.
    
    Usage:
        with budget_slice_context("bohr", "hinton", allocator, mode="detailed") as slice:
            # Colleague work happens here
            # slice.debit() called automatically by LLM backend
            pass
        # Slice automatically closed on exit
    """
    budget_slice = allocator.create_slice(
        parent_agent_id=parent_agent_id,
        colleague_id=colleague_id,
        mode=mode,
        **kwargs,
    )
    
    try:
        yield budget_slice
    finally:
        allocator.close_slice(colleague_id)


# Global allocator instance
_global_allocator: Optional[BudgetAllocator] = None


def get_budget_allocator() -> BudgetAllocator:
    """Get the global budget allocator instance."""
    global _global_allocator
    if _global_allocator is None:
        _global_allocator = BudgetAllocator()
    return _global_allocator


def set_budget_allocator(allocator: BudgetAllocator) -> None:
    """Set the global budget allocator instance."""
    global _global_allocator
    _global_allocator = allocator
