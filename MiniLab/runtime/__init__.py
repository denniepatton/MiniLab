"""
MiniLab Runtime Package
=======================

Core orchestration runtime that governs all task delegation and tool use.

Components:
- OrchestratorRuntime: Central coordination layer
- TaskGraph: DAG of TaskNodes for work planning
- RunLog: Event-sourced audit trail
- Meeting: Team and 1:1 meeting protocols
- Verification: Output validation and retry handling
"""

from .taskgraph import TaskNode, TaskGraph, TaskStatus
from .runlog import (
    RunLog,
    Event,
    EventType,
    AnyEvent,
    RunStartedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    PlanStartedEvent,
    PlanCompletedEvent,
    TaskStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskRetryEvent,
    AgentInvokedEvent,
    AgentResponseEvent,
    ToolCalledEvent,
    ToolResultEvent,
    TeamMeetingStartedEvent,
    TeamMeetingCompletedEvent,
    VerificationPassedEvent,
    VerificationFailedEvent,
    CheckpointSavedEvent,
    BudgetWarningEvent,
)
from .orchestrator import (
    OrchestratorRuntime,
    RuntimeConfig,
    RuntimeState,
    Checkpoint,
    AgentProtocol,
)
from .meeting import (
    TeamMeeting,
    OneOnOneMeeting,
    ConsultationMeeting,
    MeetingMinutes,
    MeetingConfig,
    MeetingType,
    MeetingRole,
    Participant,
    Contribution,
    ActionItem,
)
from .verification import (
    VerificationReport,
    VerificationCheck,
    Verifier,
    SchemaVerifier,
    FileVerifier,
    CodeVerifier,
    CompositeVerifier,
    CheckResult,
    CheckSpec,
)

__all__ = [
    # Task graph
    "TaskNode",
    "TaskGraph",
    "TaskStatus",
    # Run log
    "RunLog",
    "Event",
    "EventType",
    "AnyEvent",
    "RunStartedEvent",
    "RunCompletedEvent",
    "RunFailedEvent",
    "PlanStartedEvent",
    "PlanCompletedEvent",
    "TaskStartedEvent",
    "TaskCompletedEvent",
    "TaskFailedEvent",
    "TaskRetryEvent",
    "AgentInvokedEvent",
    "AgentResponseEvent",
    "ToolCalledEvent",
    "ToolResultEvent",
    "TeamMeetingStartedEvent",
    "TeamMeetingCompletedEvent",
    "VerificationPassedEvent",
    "VerificationFailedEvent",
    "CheckpointSavedEvent",
    "BudgetWarningEvent",
    # Orchestrator
    "OrchestratorRuntime",
    "RuntimeConfig",
    "RuntimeState",
    "Checkpoint",
    "AgentProtocol",
    # Meetings
    "TeamMeeting",
    "OneOnOneMeeting",
    "ConsultationMeeting",
    "MeetingMinutes",
    "MeetingConfig",
    "MeetingType",
    "MeetingRole",
    "Participant",
    "Contribution",
    "ActionItem",
    # Verification
    "VerificationReport",
    "VerificationCheck",
    "Verifier",
    "SchemaVerifier",
    "FileVerifier",
    "CodeVerifier",
    "CompositeVerifier",
    "CheckResult",
    "CheckSpec",
]
