"""
MiniLab Configuration Module.

Configuration Files (in this directory):
- agents.yaml: Agent personas, communication styles, expertise areas
- team.yaml: Tool permissions, security capabilities, write access
- agent_flexibility.yaml: Autonomy guidance, cognitive modes, decision heuristics
- budgets.yaml: Pricing info, budget guidance, communication norms

Loaders:
- loader.py → loads agents_unified.yaml (AgentConfig, FlexibilityConfig)
- team_loader.py → loads unified agents config for security/capabilities

Budget Tracking:
- BudgetManager: Session-level budget tracking, delegates to TokenAccount
- BudgetHistory: Persistent learning from historical usage (Bayesian estimates)

Note: agents.yaml and team.yaml have some duplicated fields (display_name, guild, role).
This is intentional - allows persona updates without affecting security config.
The persona file (agents.yaml) is authoritative for identity/personality.
The team file (team.yaml) is authoritative for capabilities/permissions.
"""

from .budget_manager import (
    BudgetManager,
    BudgetContext,
    get_budget_manager,
)
from .budget_history import (
    BudgetHistory,
    WorkflowStats,
    get_budget_history,
)
from .loader import AgentConfig, FlexibilityConfig, load_agent_config, load_flexibility_config

__all__ = [
    # Budget management
    "BudgetManager",
    "BudgetContext",
    "get_budget_manager",
    # Budget history
    "BudgetHistory",
    "WorkflowStats",
    "get_budget_history",
    # Config loaders
    "AgentConfig",
    "FlexibilityConfig",
    "load_agent_config",
    "load_flexibility_config",
]
