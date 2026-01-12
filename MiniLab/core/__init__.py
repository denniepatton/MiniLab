"""
MiniLab Core Components.

Centralized management for:
- TokenAccount: Real-time token budget tracking
- ProjectWriter: File output management
- ProjectSSOT: Single Source of Truth for project state
- TaskGraph: DAG-based task planning and execution
- Taxonomy: Universal token attribution system
- BudgetIsolation: Budget slices for colleague calls
"""

from .token_account import TokenAccount, get_token_account, BudgetExceededError, ToolUsageStats
from .token_context import (
    token_context,
    taxonomy_context,
    tool_context,
    module_context,
    get_taxonomy_context,
    get_module,
    get_op_kind,
    get_tool_family,
    get_agent_id,
)
from .taxonomy import (
    Module,
    OpKind,
    ToolFamily,
    TaxonomyContext,
    TaxonomyStats,
    estimate_tool_cost,
    estimate_module_cost,
)
from .project_writer import ProjectWriter
from .project_ssot import (
    ProjectSSOT,
    ProjectStatus,
    TaskPlan,
    TaskStatus as SSOTTaskStatus,
    AccessPolicy,
    BudgetState,
    get_ssot,
)
from .task_graph import TaskGraph, TaskNode, TaskStatus
from .budget_isolation import (
    BudgetSlice,
    BudgetAllocator,
    BudgetMode,
    BudgetEnforcementLevel,
    budget_slice_context,
    get_budget_allocator,
    set_budget_allocator,
)

__all__ = [
    # Token tracking
    "TokenAccount",
    "get_token_account",
    "BudgetExceededError",
    "ToolUsageStats",
    # Taxonomy
    "Module",
    "OpKind",
    "ToolFamily",
    "TaxonomyContext",
    "TaxonomyStats",
    "estimate_tool_cost",
    "estimate_module_cost",
    # Context managers
    "token_context",
    "taxonomy_context",
    "tool_context",
    "module_context",
    "get_taxonomy_context",
    "get_module",
    "get_op_kind",
    "get_tool_family",
    "get_agent_id",
    # Project SSOT
    "ProjectSSOT",
    "ProjectStatus",
    "TaskPlan",
    "SSOTTaskStatus",
    "AccessPolicy",
    "BudgetState",
    "get_ssot",
    # File management
    "ProjectWriter",
    # Task planning
    "TaskGraph",
    "TaskNode",
    "TaskStatus",
    # Budget isolation
    "BudgetSlice",
    "BudgetAllocator",
    "BudgetMode",
    "BudgetEnforcementLevel",
    "budget_slice_context",
    "get_budget_allocator",
    "set_budget_allocator",
]
