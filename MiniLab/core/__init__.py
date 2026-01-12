"""
MiniLab Core Components.

Centralized management for:
- TokenAccount: Real-time token budget tracking
- ProjectWriter: File output management
- ProjectSSOT: Single Source of Truth for project state
- TaskGraph: DAG-based task planning and execution
- Taxonomy: Universal token attribution system
- BudgetIsolation: Budget slices for colleague calls
- ProjectStructure: Standard project directory layout
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
from .task_graph import (
    TaskGraph,
    TaskNode,
    TaskStatus,
    TaskGraphValidationError,
    task_graph_to_dot,
    write_task_graph_dot,
    render_task_graph_png,
    export_task_graph_visuals,
    validate_task_graph,
)
from .budget_isolation import (
    BudgetSlice,
    BudgetAllocator,
    BudgetMode,
    BudgetEnforcementLevel,
    budget_slice_context,
    get_budget_allocator,
    set_budget_allocator,
)
from .project_structure import (
    PROJECT_STRUCTURE,
    create_project_structure,
    get_project_paths,
    validate_project_structure,
    ensure_project_structure,
)
from .token_learning import (
    TokenLearner,
    TokenRunRecord,
    ModuleTokenStats,
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
    "TaskGraphValidationError",
    "task_graph_to_dot",
    "write_task_graph_dot",
    "render_task_graph_png",
    "export_task_graph_visuals",
    "validate_task_graph",
    # Budget isolation
    "BudgetSlice",
    "BudgetAllocator",
    "BudgetMode",
    "BudgetEnforcementLevel",
    "budget_slice_context",
    "get_budget_allocator",
    "set_budget_allocator",
    # Project structure
    "PROJECT_STRUCTURE",
    "create_project_structure",
    "get_project_paths",
    "validate_project_structure",
    "ensure_project_structure",
    # Token learning
    "TokenLearner",
    "TokenRunRecord",
    "ModuleTokenStats",
]
