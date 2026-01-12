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
]
