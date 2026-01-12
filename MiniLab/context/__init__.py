"""
MiniLab Context Module

RAG-based context management system with:
- Semantic + recency biased retrieval
- Project-specific persistent context
- Structured state objects (tasks, plans, decisions)
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
]
