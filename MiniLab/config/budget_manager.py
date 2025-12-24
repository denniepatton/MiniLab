"""
Dynamic Budget Manager for MiniLab.

Provides flexible, agent-aware budget allocation that adapts to:
- Project complexity (continuous 0.0-1.0 scale, not tiers)
- Historical data via Bayesian learning (BudgetHistory)
- Runtime conditions (budget consumption rate)
- Workflow phase transitions

Philosophy: Budgets are GUIDELINES, not handcuffs.
Agents have autonomy to optimize within overall constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import yaml


@dataclass
class WorkflowBudget:
    """Budget allocation for a single workflow."""
    workflow_name: str
    allocation_percent: float
    allocated_tokens: int
    used_tokens: int = 0
    
    @property
    def remaining(self) -> int:
        return max(0, self.allocated_tokens - self.used_tokens)
    
    @property
    def percent_used(self) -> float:
        if self.allocated_tokens == 0:
            return 0.0
        return (self.used_tokens / self.allocated_tokens) * 100


@dataclass
class BudgetGuidance:
    """Runtime guidance based on budget status."""
    level: str  # healthy, caution, critical, exhausted
    guidance: str
    allow_colleague_consultations: bool
    allow_optional_steps: bool
    
    def to_prompt_text(self) -> str:
        """Generate prompt-friendly guidance text."""
        if self.level == "healthy":
            return ""  # No special guidance needed
        elif self.level == "caution":
            return "\n⚡ BUDGET NOTE: Be efficient. Focus on core deliverables.\n"
        elif self.level == "critical":
            return "\n⚠️ BUDGET CRITICAL: Complete current task minimally. Skip non-essential work.\n"
        else:
            return "\n🛑 BUDGET EXHAUSTED: Save state and wrap up immediately.\n"


class BudgetManager:
    """
    Central budget management with dynamic allocation.
    
    Key features:
    - Loads budget configuration from YAML
    - Supports continuous complexity scaling (0.0-1.0)
    - Uses Bayesian learning from BudgetHistory
    - Provides runtime guidance to agents
    - Allows dynamic reallocation between workflows
    """
    
    _instance: Optional[BudgetManager] = None
    
    # Budget thresholds
    BUDGET_WARNING = 80
    BUDGET_CRITICAL = 95
    BUDGET_RESERVE = 5  # 5% graceful shutdown reserve
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "budgets.yaml"
        
        self.config_path = config_path
        self._config: dict[str, Any] = {}
        self._total_budget: Optional[int] = None
        self._complexity: float = 0.5  # Continuous 0.0-1.0
        self._workflow_budgets: dict[str, WorkflowBudget] = {}
        self._loaded = False
        
        # Callbacks for budget events
        self._on_threshold_crossed: Optional[Callable[[int, str], None]] = None
        self._thresholds_crossed: set[int] = set()
    
    @classmethod
    def get_instance(cls, config_path: Optional[Path] = None) -> BudgetManager:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config_path)
            cls._instance.load()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing or new sessions)."""
        cls._instance = None
    
    def load(self) -> None:
        """Load configuration from YAML."""
        if self._loaded:
            return
            
        if self.config_path.exists():
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f) or {}
        else:
            # Use sensible defaults
            self._config = {
                "workflow_allocations": {
                    "consultation": 0.05,
                    "literature_review": 0.20,
                    "planning_committee": 0.15,
                    "execute_analysis": 0.35,
                    "writeup_results": 0.15,
                    "critical_review": 0.10,
                },
                "warning_thresholds": [60, 80, 95],
                "budget_behavior": {
                    "healthy": {
                        "guidance": "Proceed normally.",
                        "allow_colleague_consultations": True,
                        "allow_optional_steps": True,
                    },
                    "caution": {
                        "guidance": "Be efficient. Focus on core deliverables.",
                        "allow_colleague_consultations": True,
                        "allow_optional_steps": False,
                    },
                    "critical": {
                        "guidance": "Wrap up. Complete current task minimally.",
                        "allow_colleague_consultations": False,
                        "allow_optional_steps": False,
                    },
                    "exhausted": {
                        "guidance": "Save state and exit gracefully.",
                        "allow_colleague_consultations": False,
                        "allow_optional_steps": False,
                    },
                },
            }
        
        self._loaded = True
    
    def initialize_session(
        self, 
        total_budget: int, 
        complexity: str | float = 0.5
    ) -> None:
        """
        Initialize budget for a new session.
        
        Args:
            total_budget: Total token budget for the session
            complexity: Project complexity as:
                - float 0.0-1.0 (continuous scale, preferred)
                - str: "simple", "moderate", "complex", "exploratory" (converted to float)
        """
        self._total_budget = total_budget
        self._complexity = self._normalize_complexity(complexity)
        self._thresholds_crossed = set()
        
        # Get allocations using Bayesian estimates from history
        allocations = self._get_bayesian_allocations()
        
        # Create workflow budgets
        self._workflow_budgets = {}
        for workflow_name, allocated_tokens in allocations.items():
            # Calculate effective allocation percent
            percent = allocated_tokens / total_budget if total_budget > 0 else 0
            self._workflow_budgets[workflow_name] = WorkflowBudget(
                workflow_name=workflow_name,
                allocation_percent=percent,
                allocated_tokens=allocated_tokens,
            )
    
    def _normalize_complexity(self, complexity: str | float) -> float:
        """Convert complexity to continuous 0.0-1.0 scale."""
        if isinstance(complexity, (int, float)):
            return max(0.0, min(1.0, float(complexity)))
        
        # Map string tiers to continuous values
        tier_map = {
            "simple": 0.2,
            "moderate": 0.5,
            "complex": 0.75,
            "exploratory": 0.9,
        }
        return tier_map.get(str(complexity).lower(), 0.5)
    
    def _get_bayesian_allocations(self) -> dict[str, int]:
        """
        Get budget allocations using Bayesian estimates from BudgetHistory.
        
        Combines historical data with priors for better estimates.
        Falls back to config-based allocations if history unavailable.
        """
        if not self._total_budget:
            return {}
        
        try:
            from .budget_history import get_budget_history
            history = get_budget_history()
            
            # Get workflow list from config
            workflows = list(self._config.get("workflow_allocations", {}).keys())
            if not workflows:
                workflows = [
                    "consultation", "literature_review", "planning_committee",
                    "execute_analysis", "writeup_results", "critical_review"
                ]
            
            # Get Bayesian estimate for entire session
            estimate = history.estimate_session(workflows, self._complexity)
            
            # Scale estimates to fit within total budget (minus 5% reserve)
            usable_budget = int(self._total_budget * (1 - self.BUDGET_RESERVE / 100))
            total_estimated = estimate["total_estimated"]
            
            if total_estimated <= 0:
                # Fall back to config-based allocation
                return self._get_allocations_for_complexity_legacy()
            
            # Scale factor to fit estimates within budget
            scale = usable_budget / total_estimated if total_estimated > usable_budget else 1.0
            
            allocations = {}
            for workflow_name in workflows:
                wf_estimate = estimate["breakdown"].get(workflow_name, {})
                estimated = wf_estimate.get("estimated_tokens", 0)
                allocations[workflow_name] = int(estimated * scale)
            
            return allocations
            
        except Exception:
            # Fall back to legacy config-based allocation
            return self._get_allocations_for_complexity_legacy()
    
    def _get_allocations_for_complexity_legacy(self) -> dict[str, int]:
        """Legacy config-based allocation (fallback)."""
        if not self._total_budget:
            return {}
        
        # Map continuous complexity to tier name for config lookup
        if self._complexity < 0.35:
            tier = "simple"
        elif self._complexity < 0.65:
            tier = "moderate"
        elif self._complexity < 0.85:
            tier = "complex"
        else:
            tier = "exploratory"
        
        allocations = self._get_allocations_for_complexity(tier)
        
        # Convert percentages to token counts
        usable_budget = int(self._total_budget * (1 - self.BUDGET_RESERVE / 100))
        return {
            name: int(usable_budget * percent)
            for name, percent in allocations.items()
        }
    
    def _get_allocations_for_complexity(self, complexity: str) -> dict[str, float]:
        """Get budget allocations adjusted for complexity."""
        base = self._config.get("workflow_allocations", {})
        
        # Check for complexity-specific overrides
        multipliers = self._config.get("complexity_multipliers", {})
        if complexity in multipliers and complexity != "moderate":
            overrides = multipliers[complexity]
            # Merge overrides with base (overrides take precedence)
            result = base.copy()
            for key, value in overrides.items():
                if key not in ["description"] and isinstance(value, (int, float)):
                    result[key] = value
            return result
        
        return base
    
    def get_workflow_budget(self, workflow_name: str) -> Optional[WorkflowBudget]:
        """Get budget info for a specific workflow."""
        return self._workflow_budgets.get(workflow_name)
    
    def record_usage(self, workflow_name: str, tokens_used: int) -> None:
        """Record token usage for a workflow."""
        if workflow_name in self._workflow_budgets:
            self._workflow_budgets[workflow_name].used_tokens += tokens_used
        
        # Check thresholds
        self._check_thresholds()
    
    def _check_thresholds(self) -> None:
        """Check and trigger threshold callbacks."""
        if self._total_budget is None:
            return
        
        total_used = sum(wb.used_tokens for wb in self._workflow_budgets.values())
        percent_used = (total_used / self._total_budget) * 100
        
        thresholds = self._config.get("warning_thresholds", [60, 80, 95])
        for threshold in thresholds:
            if percent_used >= threshold and threshold not in self._thresholds_crossed:
                self._thresholds_crossed.add(threshold)
                if self._on_threshold_crossed:
                    level = self._get_level_for_percent(percent_used)
                    self._on_threshold_crossed(threshold, level)
    
    def _get_level_for_percent(self, percent: float) -> str:
        """Get budget level based on percentage used."""
        if percent >= 95:
            return "exhausted"
        elif percent >= 80:
            return "critical"
        elif percent >= 60:
            return "caution"
        return "healthy"
    
    def get_guidance(self, percent_used: Optional[float] = None) -> BudgetGuidance:
        """
        Get current budget guidance for agents.
        
        Args:
            percent_used: Override percentage (otherwise calculated from usage)
        """
        if percent_used is None and self._total_budget:
            total_used = sum(wb.used_tokens for wb in self._workflow_budgets.values())
            percent_used = (total_used / self._total_budget) * 100
        elif percent_used is None:
            percent_used = 0.0
        
        level = self._get_level_for_percent(percent_used)
        behavior = self._config.get("budget_behavior", {}).get(level, {})
        
        return BudgetGuidance(
            level=level,
            guidance=behavior.get("guidance", ""),
            allow_colleague_consultations=behavior.get("allow_colleague_consultations", True),
            allow_optional_steps=behavior.get("allow_optional_steps", True),
        )
    
    def get_workflow_guidance(self, workflow_name: str) -> str:
        """
        Get prompt-ready guidance text for a specific workflow.
        
        Returns empty string if budget is healthy, otherwise returns
        concise guidance that can be injected into agent prompts.
        """
        wb = self._workflow_budgets.get(workflow_name)
        if not wb:
            return ""
        
        # Check workflow-specific budget
        if wb.percent_used >= 90:
            return f"\n⚠️ WORKFLOW BUDGET: {workflow_name} has used {wb.percent_used:.0f}% of its allocation. Be very concise.\n"
        elif wb.percent_used >= 70:
            return f"\n📊 Budget note: {workflow_name} at {wb.percent_used:.0f}% of allocation.\n"
        
        # Also check global budget
        if self._total_budget:
            total_used = sum(w.used_tokens for w in self._workflow_budgets.values())
            global_pct = (total_used / self._total_budget) * 100
            guidance = self.get_guidance(global_pct)
            return guidance.to_prompt_text()
        
        return ""
    
    def can_start_workflow(self, workflow_name: str) -> tuple[bool, str]:
        """
        Check if a workflow should start given current budget.
        
        Returns:
            (can_start, reason) - reason explains why if can_start is False
        """
        if not self._total_budget:
            return True, "No budget set"
        
        total_used = sum(wb.used_tokens for wb in self._workflow_budgets.values())
        percent_used = (total_used / self._total_budget) * 100
        
        if percent_used >= 95:
            return False, "Budget exhausted (>95%)"
        
        wb = self._workflow_budgets.get(workflow_name)
        if wb and wb.allocated_tokens == 0:
            return False, f"No budget allocated for {workflow_name}"
        
        return True, "OK"
    
    def reallocate_budget(
        self, 
        from_workflow: str, 
        to_workflow: str, 
        percent: float
    ) -> bool:
        """
        Dynamically reallocate budget between workflows.
        
        Agents can call this to shift budget based on actual needs.
        
        Args:
            from_workflow: Source workflow to take budget from
            to_workflow: Destination workflow to give budget to
            percent: Percentage of source workflow's remaining budget to transfer
            
        Returns:
            True if reallocation succeeded
        """
        if from_workflow not in self._workflow_budgets:
            return False
        if to_workflow not in self._workflow_budgets:
            return False
        
        source = self._workflow_budgets[from_workflow]
        dest = self._workflow_budgets[to_workflow]
        
        # Calculate transfer amount
        transfer = int(source.remaining * (percent / 100))
        
        if transfer <= 0:
            return False
        
        # Perform transfer
        source.allocated_tokens -= transfer
        dest.allocated_tokens += transfer
        
        return True
    
    def set_threshold_callback(
        self, 
        callback: Callable[[int, str], None]
    ) -> None:
        """Set callback for budget threshold crossings."""
        self._on_threshold_crossed = callback
    
    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive budget summary."""
        total_used = sum(wb.used_tokens for wb in self._workflow_budgets.values())
        
        return {
            "total_budget": self._total_budget,
            "total_used": total_used,
            "total_remaining": (self._total_budget - total_used) if self._total_budget else None,
            "percent_used": (total_used / self._total_budget * 100) if self._total_budget else 0,
            "complexity": self._complexity,
            "workflows": {
                name: {
                    "allocated": wb.allocated_tokens,
                    "used": wb.used_tokens,
                    "remaining": wb.remaining,
                    "percent_used": wb.percent_used,
                }
                for name, wb in self._workflow_budgets.items()
            },
            "guidance": self.get_guidance().level,
        }


def get_budget_manager() -> BudgetManager:
    """Get the global BudgetManager singleton."""
    return BudgetManager.get_instance()


def estimate_complexity(
    user_request: str,
    data_files: Optional[list[str]] = None,
    has_literature_phase: bool = True,
) -> float:
    """
    Estimate project complexity on continuous 0.0-1.0 scale.
    
    Factors considered:
    - Request length and vocabulary complexity
    - Number and size of data files
    - Presence of specific workflow phases
    - Keywords indicating complexity
    
    Args:
        user_request: User's initial request text
        data_files: List of data file paths
        has_literature_phase: Whether literature review is planned
    
    Returns:
        Complexity score 0.0 (simple) to 1.0 (exploratory)
    """
    score = 0.5  # Start at moderate
    
    # Text length factor (longer requests tend to be more complex)
    words = len(user_request.split())
    if words < 50:
        score -= 0.1
    elif words > 200:
        score += 0.1
    elif words > 400:
        score += 0.2
    
    # Complexity keywords
    request_lower = user_request.lower()
    
    simple_keywords = [
        "simple", "quick", "brief", "just", "only", "basic",
        "single", "one", "straightforward"
    ]
    complex_keywords = [
        "comprehensive", "thorough", "detailed", "multiple",
        "integrate", "multimodal", "survival", "longitudinal",
        "machine learning", "deep learning", "ensemble", "cross-validation",
        "exploratory", "explore", "investigate"
    ]
    
    for kw in simple_keywords:
        if kw in request_lower:
            score -= 0.05
    
    for kw in complex_keywords:
        if kw in request_lower:
            score += 0.05
    
    # Data file factor
    if data_files:
        file_count = len(data_files)
        if file_count == 1:
            score -= 0.05
        elif file_count >= 3:
            score += 0.05
        elif file_count >= 5:
            score += 0.1
    
    # Literature phase adds complexity
    if has_literature_phase:
        score += 0.05
    
    # Clamp to valid range
    return max(0.0, min(1.0, score))

