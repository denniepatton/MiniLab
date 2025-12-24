"""
MiniLab: Multi-agent scientific lab assistant.

A sophisticated multi-agent system for conducting scientific data analysis,
literature review, and research assistance.

Architecture:
- core/: TokenAccount and ProjectWriter for centralized management
- security/: PathGuard for file access control
- tools/: Typed tool system with Pydantic validation
- context/: RAG-based context management with FAISS
- agents/: Structured role-specific agents with ReAct loops
- workflows/: Modular workflow components
- orchestrators/: Bohr orchestrator for workflow coordination

Quick Start:
    from MiniLab import run_minilab
    
    results = await run_minilab(
        request="Analyze the genomic data in ReadData/Pluvicto",
        project_name="pluvicto_analysis"
    )
"""

__version__ = "0.4.0"

# Core components
from .core import TokenAccount, get_token_account, ProjectWriter, BudgetExceededError

# Core exports
from .orchestrators import BohrOrchestrator, MiniLabSession
from .orchestrators.bohr_orchestrator import run_minilab

# Security
from .security import PathGuard

# Context management
from .context import ContextManager, ProjectState, TaskState

# Agent system
from .agents import Agent, AgentRegistry

# LLM Backends
from .llm_backends import AnthropicBackend, LLMBackend

# Console utilities
from .utils import console

# Workflow modules
from .workflows import (
    WorkflowModule,
    WorkflowResult,
    WorkflowStatus,
    ConsultationModule,
    LiteratureReviewModule,
    PlanningCommitteeModule,
    ExecuteAnalysisModule,
    WriteupResultsModule,
    CriticalReviewModule,
)

# Agent creation utility
from .agents.registry import create_agents

__all__ = [
    # Version
    "__version__",
    # Core
    "TokenAccount",
    "get_token_account",
    "ProjectWriter",
    "BudgetExceededError",
    # Main entry point
    "run_minilab",
    "BohrOrchestrator",
    "MiniLabSession",
    "MajorWorkflow",
    # Security
    "PathGuard",
    # Context
    "ContextManager",
    "ProjectState",
    "TaskState",
    # Agents
    "Agent",
    "AgentRegistry",
    # LLM Backends
    "AnthropicBackend",
    "LLMBackend",
    # Console
    "console",
    # Workflows
    "WorkflowModule",
    "WorkflowResult",
    "WorkflowStatus",
    "ConsultationModule",
    "LiteratureReviewModule",
    "PlanningCommitteeModule",
    "ExecuteAnalysisModule",
    "WriteupResultsModule",
    "CriticalReviewModule",
    # Utilities
    "create_agents",
]
