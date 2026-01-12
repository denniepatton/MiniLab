"""
Budget Manager for MiniLab.

Simplified budget management that:
- Delegates to TokenAccount for authoritative usage tracking
- Provides continuous budget context to agents (not thresholds)
- Records usage to BudgetHistory for learning
- Provides cost estimates based on historical data

Philosophy: Agents see their budget and self-regulate.
TokenAccount is the single source of truth for actual usage.
BudgetHistory provides learned estimates for planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class BudgetContext:
    """
    Current budget context for injection into agent prompts.
    
    Agents receive this context and decide how to proceed based on
    their judgment, not hardcoded rules.
    """
    total_budget: int
    used_tokens: int
    remaining_tokens: int
    percent_used: float
    estimated_cost: float = 0.0
    cache_efficiency: float = 0.0
    
    def to_prompt_text(self) -> str:
        """Generate natural language budget context for agents."""
        cost_str = f" (~${self.estimated_cost:.2f})" if self.estimated_cost > 0 else ""
        cache_str = f" Cache hit rate: {self.cache_efficiency:.0f}%." if self.cache_efficiency > 0 else ""
        
        if self.percent_used < 25:
            return f"Budget: {self.remaining_tokens:,} tokens remaining ({self.percent_used:.0f}% used{cost_str}).{cache_str} Plenty of room to work thoroughly."
        elif self.percent_used < 50:
            return f"Budget: {self.remaining_tokens:,} tokens remaining ({self.percent_used:.0f}% used{cost_str}).{cache_str} Good progress, continue with current approach."
        elif self.percent_used < 75:
            return f"Budget: {self.remaining_tokens:,} tokens remaining ({self.percent_used:.0f}% used{cost_str}).{cache_str} Focus on core deliverables."
        elif self.percent_used < 90:
            return f"Budget: {self.remaining_tokens:,} tokens remaining ({self.percent_used:.0f}% used{cost_str}).{cache_str} Prioritize completing essential work."
        else:
            return f"Budget: {self.remaining_tokens:,} tokens remaining ({self.percent_used:.0f}% used{cost_str}).{cache_str} Wrap up and save progress."


class BudgetManager:
    """
    Budget tracking and estimation for MiniLab sessions.
    
    Key principles:
    - TokenAccount is the single source of truth for actual usage
    - BudgetHistory provides learned estimates for planning
    - This class provides convenience methods and agent-facing context
    - Records workflow usage to BudgetHistory for learning
    """
    
    _instance: Optional[BudgetManager] = None
    
    def __init__(self):
        self._complexity: float = 0.5
        self._workflow_usage: dict[str, int] = {}  # workflow_name -> tokens used
        self._threshold_callback = None
    
    @classmethod
    def get_instance(cls) -> BudgetManager:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset for new session."""
        cls._instance = None
    
    def initialize_session(self, total_budget: int, complexity: Any = 0.5, model: str = None) -> None:
        """
        Initialize budget for a new session.
        
        Sets up TokenAccount with budget and model for accurate pricing.
        
        Args:
            total_budget: Total tokens available for the session
            complexity: Estimated complexity 0.0-1.0 (for history recording)
            model: Model name for accurate pricing (e.g., "claude-sonnet-4-5")
        """
        from ..core import get_token_account
        
        # Accept either a float (0-1) or a label like "moderate".
        try:
            from .budget_history import _coerce_complexity
            self._complexity = _coerce_complexity(complexity)
        except Exception:
            self._complexity = 0.5
        self._workflow_usage = {}
        
        # TokenAccount is the authoritative source
        account = get_token_account()
        account.reset()
        account.set_budget(total_budget, model=model)

    def set_threshold_callback(self, callback) -> None:
        """Set a callback invoked when budget thresholds are crossed.

        Note: Threshold detection is handled elsewhere (TokenAccount warnings). This
        method exists to support orchestrator wiring and future extensions.
        """
        self._threshold_callback = callback
    
    def record_usage(
        self,
        workflow_name: str,
        tokens_used: int,
        status: str = "completed",
        stop_reason: str | None = None,
        breakdown: dict[str, Any] | None = None,
    ) -> None:
        """
        Record token usage for a workflow.
        
        Records to BudgetHistory for Bayesian learning.
        Note: TokenAccount already tracks actual usage automatically.
        This method is for workflow-level attribution and learning.
        """
        if workflow_name in self._workflow_usage:
            self._workflow_usage[workflow_name] += tokens_used
        else:
            self._workflow_usage[workflow_name] = tokens_used
        
        # Record to history for learning
        try:
            from .budget_history import get_budget_history
            # Prefer outcome-aware API
            get_budget_history().record_run(
                workflow_name=workflow_name,
                tokens_used=tokens_used,
                status=status,
                stop_reason=stop_reason,
                complexity=self._complexity,
                breakdown=breakdown,
            )
        except Exception:
            pass  # History recording is best-effort
    
    def get_context(self) -> BudgetContext:
        """Get current budget context for agent injection (from TokenAccount)."""
        try:
            from ..core import get_token_account
            account = get_token_account()
            
            total = account.budget or 0
            used = account.total_used
            remaining = max(0, total - used)
            percent = (used / total * 100) if total > 0 else 0
            
            # Get cost and cache info
            summary = account.usage_summary
            estimated_cost = summary.get("estimated_cost", 0.0)
            cache_eff = summary.get("cache_efficiency", {}).get("hit_rate", 0.0)
            
            return BudgetContext(
                total_budget=total,
                used_tokens=used,
                remaining_tokens=remaining,
                percent_used=percent,
                estimated_cost=estimated_cost,
                cache_efficiency=cache_eff,
            )
        except ImportError:
            return BudgetContext(
                total_budget=0,
                used_tokens=0,
                remaining_tokens=0,
                percent_used=0.0,
            )
    
    def get_remaining(self) -> int:
        """Get remaining tokens (from TokenAccount)."""
        try:
            from ..core import get_token_account
            return get_token_account().remaining
        except ImportError:
            return 0
    
    def get_percent_used(self) -> float:
        """Get percentage of budget used (from TokenAccount)."""
        try:
            from ..core import get_token_account
            return get_token_account().percentage_used
        except ImportError:
            return 0.0
    
    def estimate_workflow_budget(self, workflow_name: str) -> dict[str, Any]:
        """
        Estimate tokens needed for a workflow based on history.
        
        Returns estimate from BudgetHistory with confidence intervals.
        """
        try:
            from .budget_history import get_budget_history
            return get_budget_history().estimate(workflow_name, self._complexity)
        except Exception:
            # No history - return conservative default with high uncertainty
            default = 50_000
            return {
                "estimated_tokens": default,
                "confidence_low": int(default * 0.5),
                "confidence_high": int(default * 2.0),
                "data_points": 0,
                "source": "default",
            }
    
    def estimate_session_budget(self, workflows: list[str]) -> dict[str, Any]:
        """
        Estimate total tokens for planned workflows.
        
        Uses BudgetHistory for learned estimates per workflow.
        """
        try:
            from .budget_history import get_budget_history
            return get_budget_history().estimate_session(workflows, self._complexity)
        except Exception:
            # Fallback: conservative defaults
            default_per_workflow = 50_000
            total = default_per_workflow * len(workflows)
            return {
                "total_estimated": total,
                "total_low": int(total * 0.5),
                "total_high": int(total * 2.0),
                "breakdown": {wf: {"estimated_tokens": default_per_workflow, "source": "default"} for wf in workflows},
                "complexity": self._complexity,
            }
    
    def get_summary(self) -> dict[str, Any]:
        """Get budget summary combining TokenAccount data and workflow attribution."""
        try:
            from ..core import get_token_account
            account = get_token_account()
            summary = account.usage_summary
            summary["complexity"] = self._complexity
            summary["workflow_usage"] = self._workflow_usage.copy()
            return summary
        except ImportError:
            return {
                "complexity": self._complexity,
                "workflow_usage": self._workflow_usage.copy(),
            }


def get_budget_manager() -> BudgetManager:
    """Get the global BudgetManager singleton."""
    return BudgetManager.get_instance()
