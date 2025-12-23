"""
MiniLab Workflow Modules.

Modular workflow components for structured analysis pipelines.
"""

from .base import WorkflowModule, WorkflowResult, WorkflowCheckpoint, WorkflowStatus
from .consultation import ConsultationModule
from .literature_review import LiteratureReviewModule
from .planning_committee import PlanningCommitteeModule
from .execute_analysis import ExecuteAnalysisModule
from .writeup_results import WriteupResultsModule
from .critical_review import CriticalReviewModule

# Artifact-first workflow library
from .artifacts import (
    Artifact,
    ArtifactType,
    ArtifactStatus,
    ArtifactStore,
    ArtifactManifest,
    ProvenanceRecord,
    WorkflowStep,
    DataLoadStep,
    AnalysisStep,
    ReportStep,
)
from .patterns import (
    AnalysisType,
    AnalysisConfig,
    AnalysisResult,
    AnalysisPattern,
    ValidationStrategy,
    DescriptivePattern,
    SurvivalPattern,
    ClassificationPattern,
    StandardReportTemplate,
    get_pattern,
    ANALYSIS_PATTERNS,
)

__all__ = [
    # Base workflow
    "WorkflowModule",
    "WorkflowResult",
    "WorkflowCheckpoint",
    "WorkflowStatus",
    # Workflow modules
    "ConsultationModule",
    "LiteratureReviewModule",
    "PlanningCommitteeModule",
    "ExecuteAnalysisModule",
    "WriteupResultsModule",
    "CriticalReviewModule",
    # Artifacts
    "Artifact",
    "ArtifactType",
    "ArtifactStatus",
    "ArtifactStore",
    "ArtifactManifest",
    "ProvenanceRecord",
    "WorkflowStep",
    "DataLoadStep",
    "AnalysisStep",
    "ReportStep",
    # Patterns
    "AnalysisType",
    "AnalysisConfig",
    "AnalysisResult",
    "AnalysisPattern",
    "ValidationStrategy",
    "DescriptivePattern",
    "SurvivalPattern",
    "ClassificationPattern",
    "StandardReportTemplate",
    "get_pattern",
    "ANALYSIS_PATTERNS",
]
