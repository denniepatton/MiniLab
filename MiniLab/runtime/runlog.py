"""
RunLog: Event-sourced execution log.

Provides:
- Event types for all runtime operations
- RunLog for append-only event storage
- Replay and query capabilities
"""

from __future__ import annotations

from abc import ABC
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union
import json

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of events in the run log."""

    # Planning events
    PLAN_STARTED = "plan_started"
    PLAN_COMPLETED = "plan_completed"

    # Task events
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_RETRY = "task_retry"
    TASK_SKIPPED = "task_skipped"

    # Meeting events
    TEAM_MEETING_STARTED = "team_meeting_started"
    TEAM_MEETING_COMPLETED = "team_meeting_completed"
    ONE_ON_ONE_STARTED = "one_on_one_started"
    ONE_ON_ONE_COMPLETED = "one_on_one_completed"

    # Agent events
    AGENT_INVOKED = "agent_invoked"
    AGENT_RESPONSE = "agent_response"

    # Tool events
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"

    # Verification events
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_FAILED = "verification_failed"

    # Checkpoint events
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_RESTORED = "checkpoint_restored"

    # Budget events
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"

    # Run lifecycle
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"


class Event(BaseModel, ABC):
    """Base class for all events."""

    event_type: EventType = Field(..., description="Type of event")
    timestamp: datetime = Field(default_factory=datetime.now, description="When event occurred")
    run_id: str = Field(..., description="ID of the run this event belongs to")
    sequence: int = Field(default=0, description="Sequence number within run")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {"extra": "forbid"}


class RunStartedEvent(Event):
    """Run has started."""
    event_type: EventType = EventType.RUN_STARTED
    goal: str = Field(..., description="High-level goal for this run")
    config: dict[str, Any] = Field(default_factory=dict, description="Run configuration")


class RunCompletedEvent(Event):
    """Run completed successfully."""
    event_type: EventType = EventType.RUN_COMPLETED
    summary: str = Field(..., description="Summary of what was accomplished")
    artifacts: list[str] = Field(default_factory=list, description="Output artifact paths")


class RunFailedEvent(Event):
    """Run failed."""
    event_type: EventType = EventType.RUN_FAILED
    error: str = Field(..., description="Error message")
    partial_results: Optional[dict[str, Any]] = Field(default=None, description="Partial results")


class PlanStartedEvent(Event):
    """Planning phase started."""
    event_type: EventType = EventType.PLAN_STARTED
    goal: str = Field(..., description="Goal being planned for")


class PlanCompletedEvent(Event):
    """Planning phase completed."""
    event_type: EventType = EventType.PLAN_COMPLETED
    task_count: int = Field(..., description="Number of tasks planned")
    estimated_tokens: Optional[int] = Field(default=None, description="Estimated token usage")


class TaskStartedEvent(Event):
    """Task execution started."""
    event_type: EventType = EventType.TASK_STARTED
    task_id: str = Field(..., description="ID of the task")
    task_description: str = Field(..., description="What the task does")
    owner_agent: str = Field(..., description="Agent executing the task")


class TaskCompletedEvent(Event):
    """Task completed successfully."""
    event_type: EventType = EventType.TASK_COMPLETED
    task_id: str = Field(..., description="ID of the task")
    result_summary: str = Field(..., description="Summary of result")
    tokens_used: Optional[int] = Field(default=None, description="Tokens consumed")


class TaskFailedEvent(Event):
    """Task failed."""
    event_type: EventType = EventType.TASK_FAILED
    task_id: str = Field(..., description="ID of the task")
    error: str = Field(..., description="Error message")
    retries_remaining: int = Field(default=0, description="Retries left")


class TaskRetryEvent(Event):
    """Task being retried."""
    event_type: EventType = EventType.TASK_RETRY
    task_id: str = Field(..., description="ID of the task")
    attempt: int = Field(..., description="Retry attempt number")
    reason: str = Field(..., description="Why retry was triggered")


class AgentInvokedEvent(Event):
    """Agent was invoked."""
    event_type: EventType = EventType.AGENT_INVOKED
    agent_name: str = Field(..., description="Name of the agent")
    task_id: Optional[str] = Field(default=None, description="Task being worked on")
    prompt_tokens: int = Field(default=0, description="Tokens in prompt")


class AgentResponseEvent(Event):
    """Agent produced a response."""
    event_type: EventType = EventType.AGENT_RESPONSE
    agent_name: str = Field(..., description="Name of the agent")
    response_tokens: int = Field(default=0, description="Tokens in response")
    tool_calls: int = Field(default=0, description="Number of tool calls made")


class ToolCalledEvent(Event):
    """Tool was called."""
    event_type: EventType = EventType.TOOL_CALLED
    tool_name: str = Field(..., description="Name of the tool")
    agent_name: str = Field(..., description="Agent that called the tool")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class ToolResultEvent(Event):
    """Tool returned a result."""
    event_type: EventType = EventType.TOOL_RESULT
    tool_name: str = Field(..., description="Name of the tool")
    success: bool = Field(..., description="Whether tool succeeded")
    output_chars: int = Field(default=0, description="Characters in output")


class TeamMeetingStartedEvent(Event):
    """Team meeting started."""
    event_type: EventType = EventType.TEAM_MEETING_STARTED
    topic: str = Field(..., description="Meeting topic")
    participants: list[str] = Field(default_factory=list, description="Participating agents")


class TeamMeetingCompletedEvent(Event):
    """Team meeting completed."""
    event_type: EventType = EventType.TEAM_MEETING_COMPLETED
    topic: str = Field(..., description="Meeting topic")
    decisions: list[str] = Field(default_factory=list, description="Decisions made")


class VerificationEvent(Event):
    """Verification result."""
    event_type: EventType = EventType.VERIFICATION_STARTED
    task_id: Optional[str] = Field(default=None, description="Task being verified")
    artifact_path: Optional[str] = Field(default=None, description="Artifact being verified")


class VerificationPassedEvent(Event):
    """Verification passed."""
    event_type: EventType = EventType.VERIFICATION_PASSED
    task_id: Optional[str] = Field(default=None, description="Task verified")
    checks_passed: int = Field(default=0, description="Number of checks passed")


class VerificationFailedEvent(Event):
    """Verification failed."""
    event_type: EventType = EventType.VERIFICATION_FAILED
    task_id: Optional[str] = Field(default=None, description="Task verified")
    failures: list[str] = Field(default_factory=list, description="Failed checks")


class CheckpointSavedEvent(Event):
    """Checkpoint was saved."""
    event_type: EventType = EventType.CHECKPOINT_SAVED
    checkpoint_path: str = Field(..., description="Path to checkpoint file")
    tasks_completed: int = Field(default=0, description="Tasks completed at checkpoint")


class CheckpointRestoredEvent(Event):
    """Checkpoint was restored."""
    event_type: EventType = EventType.CHECKPOINT_RESTORED
    checkpoint_path: str = Field(..., description="Path to checkpoint file")


class BudgetWarningEvent(Event):
    """Budget threshold exceeded."""
    event_type: EventType = EventType.BUDGET_WARNING
    budget_used: int = Field(..., description="Tokens used")
    budget_limit: int = Field(..., description="Token limit")
    percentage: float = Field(..., description="Percentage used")


# Union of all event types
AnyEvent = Union[
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
    VerificationEvent,
    VerificationPassedEvent,
    VerificationFailedEvent,
    CheckpointSavedEvent,
    CheckpointRestoredEvent,
    BudgetWarningEvent,
]


class RunLog:
    """
    Append-only event log for a single run.
    
    Provides event storage, streaming to disk, and query capabilities.
    """

    def __init__(self, run_id: str, output_dir: Optional[Path] = None):
        """
        Initialize a run log.
        
        Args:
            run_id: Unique identifier for this run
            output_dir: Directory to write runlog.jsonl (optional)
        """
        self.run_id = run_id
        self.events: list[AnyEvent] = []
        self._sequence = 0
        self._output_file: Optional[Path] = None

        if output_dir:
            self._output_file = output_dir / "runlog.jsonl"

    def append(self, event: AnyEvent) -> None:
        """
        Append an event to the log.
        
        Sets sequence number and run_id, then persists if output configured.
        """
        self._sequence += 1
        event.run_id = self.run_id
        event.sequence = self._sequence
        self.events.append(event)

        # Stream to disk if configured
        if self._output_file:
            self._write_event(event)

    def _write_event(self, event: AnyEvent) -> None:
        """Write a single event to disk."""
        if not self._output_file:
            return

        self._output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self._output_file, "a") as f:
            # Use model_dump with mode="json" for serialization
            event_dict = event.model_dump(mode="json")
            # Convert datetime to ISO format
            event_dict["timestamp"] = event.timestamp.isoformat()
            f.write(json.dumps(event_dict) + "\n")

    def query_by_type(self, event_type: EventType) -> list[AnyEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def query_by_task(self, task_id: str) -> list[AnyEvent]:
        """Get all events for a specific task."""
        return [
            e for e in self.events
            if hasattr(e, "task_id") and e.task_id == task_id
        ]

    def query_by_agent(self, agent_name: str) -> list[AnyEvent]:
        """Get all events for a specific agent."""
        return [
            e for e in self.events
            if hasattr(e, "agent_name") and e.agent_name == agent_name
        ]

    def latest(self, n: int = 10) -> list[AnyEvent]:
        """Get the n most recent events."""
        return self.events[-n:]

    def total_tokens(self) -> int:
        """Sum all token usage from events."""
        total = 0
        for event in self.events:
            if hasattr(event, "prompt_tokens"):
                total += event.prompt_tokens
            if hasattr(event, "response_tokens"):
                total += event.response_tokens
            if hasattr(event, "tokens_used") and event.tokens_used:
                total += event.tokens_used
        return total

    def summary(self) -> dict[str, Any]:
        """Generate a summary of the run log."""
        event_counts: dict[str, int] = {}
        for event in self.events:
            event_counts[event.event_type.value] = event_counts.get(event.event_type.value, 0) + 1

        return {
            "run_id": self.run_id,
            "event_count": len(self.events),
            "event_types": event_counts,
            "total_tokens": self.total_tokens(),
        }

    @classmethod
    def load(cls, jsonl_path: Path) -> "RunLog":
        """
        Load a run log from a JSONL file.
        
        Args:
            jsonl_path: Path to runlog.jsonl file
            
        Returns:
            Populated RunLog instance
        """
        events = []
        run_id = ""

        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if not run_id:
                        run_id = data.get("run_id", "unknown")
                    # Parse based on event_type
                    # For simplicity, store as generic Event
                    events.append(data)

        log = cls(run_id=run_id)
        log.events = events  # type: ignore
        log._sequence = len(events)
        return log
