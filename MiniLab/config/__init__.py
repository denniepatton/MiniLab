"""
MiniLab Configuration Module.

Provides:
- BudgetManager: Dynamic budget allocation with Bayesian learning
- BudgetHistory: Historical token usage tracking
- Config loaders for agents.yaml, budgets.yaml, team.yaml
"""

from .budget_manager import (
    BudgetManager,
    BudgetGuidance,
    WorkflowBudget,
    get_budget_manager,
    estimate_complexity,
)
from .budget_history import (
    BudgetHistory,
    WorkflowStats,
    get_budget_history,
)
from .loader import load_agent_config

__all__ = [
    # Budget management
    "BudgetManager",
    "BudgetGuidance",
    "WorkflowBudget",
    "get_budget_manager",
    "estimate_complexity",
    # Budget history
    "BudgetHistory",
    "WorkflowStats",
    "get_budget_history",
    # Config loaders
    "load_agent_config",
]
