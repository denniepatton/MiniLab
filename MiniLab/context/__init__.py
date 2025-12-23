"""
MiniLab Context Module

RAG-based context management system with:
- Semantic + recency biased retrieval
- Project-specific persistent context
- Structured state objects (tasks, plans, decisions)
- Rolling task state with compression
- Budget enforcement with degradation modes
- Memory compaction and artifact storage
"""

from .context_manager import ContextManager, ProjectContext
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .state_objects import (
    TaskState,
    ProjectState,
    ConversationSummary,
    WorkingPlan,
    ExecutionPlan,
    DataManifest,
)
from .budget_enforcer import (
    BudgetEnforcer,
    BudgetConfig,
    BudgetScope,
    DegradationMode,
    ScopedBudget,
)
from .memory_manager import (
    MemoryManager,
    MemoryConfig,
    MemoryEntry,
    ContentType,
    CompressionStrategy,
    MemoryCompressor,
    SimpleCompressor,
)

__all__ = [
    # Context management
    "ContextManager",
    "ProjectContext",
    "EmbeddingManager",
    "VectorStore",
    # State objects
    "TaskState",
    "ProjectState",
    "ConversationSummary",
    "WorkingPlan",
    "ExecutionPlan",
    "DataManifest",
    # Budget enforcement
    "BudgetEnforcer",
    "BudgetConfig",
    "BudgetScope",
    "DegradationMode",
    "ScopedBudget",
    # Memory management
    "MemoryManager",
    "MemoryConfig",
    "MemoryEntry",
    "ContentType",
    "CompressionStrategy",
    "MemoryCompressor",
    "SimpleCompressor",
]
