"""
MiniLab Modules.

Modular components for structured analysis pipelines.

Terminology (aligned with minilab_outline.md):
- Task: A project-DAG node representing a user-meaningful milestone
- Module: A reusable procedure that composes tools and possibly agents
- Tool: An atomic, side-effectful capability with typed I/O

Module Categories:

Coordination Modules:
- ConsultationModule: Initial scope confirmation
- TeamDiscussionModule: Multi-agent feedback (was PlanningCommitteeModule)
- OneOnOneModule: Deep dive with specific expert
- PlanningModule: Full plan production
- CoreInputModule: Core subgroup answer

Evidence & Writing Modules:
- EvidenceGatheringModule: Search + evidence packets
- WriteArtifactModule: Mandatory write gateway
- BuildReportModule: Assemble narrative outputs (was WriteupResultsModule)
- LiteratureReviewModule: Literature synthesis

Execution & Verification Modules:
- GenerateCodeModule: Produce runnable scripts
- AnalysisExecutionModule: Execute analysis (was ExecuteAnalysisModule)
- RunChecksModule: Tests/lint/smoke checks
- SanityCheckDataModule: Data validation
- InterpretStatsModule: Statistical interpretation
- InterpretPlotModule: Visual plot interpretation
- CitationCheckModule: Citation integrity
- FormattingCheckModule: Rubric compliance

Review Modules:
- CriticalReviewModule: Peer review-style scrutiny

External Modules:
- ConsultExternalExpertModule: Strict-contract specialist consultation
"""

from .base import (
    Module,
    ModuleResult,
    ModuleCheckpoint,
    ModuleStatus,
    ModuleType,
    # Backward compatibility
    WorkflowModule,
    WorkflowResult,
    WorkflowCheckpoint,
    WorkflowStatus,
)

# Import modules as they are migrated
from .consultation import ConsultationModule
from .team_discussion import TeamDiscussionModule
from .literature_review import LiteratureReviewModule
from .analysis_execution import AnalysisExecutionModule
from .build_report import BuildReportModule
from .critical_review import CriticalReviewModule

# New modules
from .planning import PlanningModule
from .one_on_one import OneOnOneModule
from .core_input import CoreInputModule
from .evidence_gathering import EvidenceGatheringModule
from .write_artifact import WriteArtifactModule
from .generate_code import GenerateCodeModule
from .run_checks import RunChecksModule
from .sanity_check_data import SanityCheckDataModule
from .interpret_stats import InterpretStatsModule
from .interpret_plot import InterpretPlotModule
from .citation_check import CitationCheckModule
from .formatting_check import FormattingCheckModule
from .consult_external_expert import ConsultExternalExpertModule

# Plan dissemination utilities
from .plan_dissemination import (
    format_task_graph_as_plan,
    extract_agent_responsibilities,
    build_agent_context,
)

__all__ = [
    # Base classes
    "Module",
    "ModuleResult",
    "ModuleCheckpoint",
    "ModuleStatus",
    "ModuleType",
    # Backward compatibility
    "WorkflowModule",
    "WorkflowResult",
    "WorkflowCheckpoint",
    "WorkflowStatus",
    # Coordination modules
    "ConsultationModule",
    "TeamDiscussionModule",
    "OneOnOneModule",
    "PlanningModule",
    "CoreInputModule",
    # Evidence & writing modules
    "EvidenceGatheringModule",
    "WriteArtifactModule",
    "BuildReportModule",
    "LiteratureReviewModule",
    # Execution & verification modules
    "GenerateCodeModule",
    "AnalysisExecutionModule",
    "RunChecksModule",
    "SanityCheckDataModule",
    "InterpretStatsModule",
    "InterpretPlotModule",
    "CitationCheckModule",
    "FormattingCheckModule",
    # Review modules
    "CriticalReviewModule",
    # External modules
    "ConsultExternalExpertModule",
    # Utilities
    "format_task_graph_as_plan",
    "extract_agent_responsibilities",
    "build_agent_context",
]
