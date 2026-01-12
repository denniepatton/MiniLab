"""
MiniLab Configuration Module.

Terminology (aligned with minilab_outline.md):
- Task: A project-DAG node representing a user-meaningful milestone
- Module: A reusable procedure that composes tools and possibly agents
- Tool: An atomic, side-effectful capability with typed I/O

Configuration Files (in this directory):
- agents_unified.yaml: Agent personas, tools, permissions, security (unified)
- agent_flexibility.yaml: Autonomy guidance, cognitive modes, decision heuristics

Loaders:
- loader.py → loads agents_unified.yaml (AgentConfig, FlexibilityConfig)
- team_loader.py → loads unified agents config for security/capabilities
- minilab_config.py → SSOT for project structure, budgets, modules, features

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
from .minilab_config import (
    MiniLabConfig,
    ProjectStructure,
    ModuleConfig,
    WorkflowConfig,  # Backward compat alias
    BudgetConfig,
    FeatureConfig,
    ErrorHandlingPolicy,
    get_config,
    reload_config,
)

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
    # SSOT config
    "MiniLabConfig",
    "ProjectStructure",
    "ModuleConfig",
    "WorkflowConfig",  # Backward compat
    "BudgetConfig",
    "FeatureConfig",
    "ErrorHandlingPolicy",
    "get_config",
    "reload_config",
]
