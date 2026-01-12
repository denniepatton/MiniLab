"""
MiniLab: Multi-agent scientific lab assistant.

A sophisticated multi-agent system for conducting scientific data analysis,
literature review, and research assistance.

Terminology (aligned with minilab_outline.md):
- Task: A project-DAG node representing a user-meaningful milestone
- Module: A reusable procedure that composes tools and possibly agents
- Tool: An atomic, side-effectful capability with typed I/O

Architecture:
- core/: TokenAccount and ProjectWriter for centralized management
- security/: PathGuard for file access control
- tools/: Typed tool system with Pydantic validation
- context/: RAG-based context management with FAISS
- agents/: Structured role-specific agents with ReAct loops
- modules/: Modular module components (formerly workflows/)
- orchestrator/: TaskGraph-driven orchestrator for module coordination

Quick Start:
    from MiniLab import run_minilab
    
    results = await run_minilab(
        request="Analyze the genomic data in ReadData/Pluvicto",
        project_name="pluvicto_analysis"
    )
"""

__version__ = "0.5.0"  # Bumped for terminology refactor

# Core components
from .core import TokenAccount, get_token_account, ProjectWriter, BudgetExceededError

# Core exports
from .orchestrator import MiniLabOrchestrator, MiniLabSession, BohrOrchestrator, run_minilab

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

# Modules (new terminology) - with backward compat aliases
from .modules import (
    Module,
    ModuleResult,
    ModuleStatus,
    ConsultationModule,
    LiteratureReviewModule,
    TeamDiscussionModule,
    AnalysisExecutionModule,
    BuildReportModule,
    CriticalReviewModule,
    # New modules
    PlanningModule,
    OneOnOneModule,
    CoreInputModule,
    EvidenceGatheringModule,
    WriteArtifactModule,
    GenerateCodeModule,
    RunChecksModule,
    SanityCheckDataModule,
    InterpretStatsModule,
    InterpretPlotModule,
    CitationCheckModule,
    FormattingCheckModule,
    ConsultExternalExpertModule,
    # Backward compatibility aliases
    WorkflowModule,
    WorkflowResult,
    WorkflowStatus,
)

# Backward compatibility aliases for old module names
PlanningCommitteeModule = TeamDiscussionModule
ExecuteAnalysisModule = AnalysisExecutionModule
WriteupResultsModule = BuildReportModule

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
    "MiniLabOrchestrator",
    "BohrOrchestrator",  # Backwards-compatible alias
    "MiniLabSession",
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
    # Modules (new terminology)
    "Module",
    "ModuleResult",
    "ModuleStatus",
    "ConsultationModule",
    "LiteratureReviewModule",
    "TeamDiscussionModule",
    "AnalysisExecutionModule",
    "BuildReportModule",
    "CriticalReviewModule",
    "PlanningModule",
    "OneOnOneModule",
    "CoreInputModule",
    "EvidenceGatheringModule",
    "WriteArtifactModule",
    "GenerateCodeModule",
    "RunChecksModule",
    "SanityCheckDataModule",
    "InterpretStatsModule",
    "InterpretPlotModule",
    "CitationCheckModule",
    "FormattingCheckModule",
    "ConsultExternalExpertModule",
    # Backward compatibility aliases
    "WorkflowModule",
    "WorkflowResult",
    "WorkflowStatus",
    "PlanningCommitteeModule",
    "ExecuteAnalysisModule",
    "WriteupResultsModule",
    # Utilities
    "create_agents",
]
