"""
Meeting: Structured multi-agent conversations.

Provides:
- TeamMeeting: Multiple agents discussing a topic
- OneOnOneMeeting: Two agents collaborating
- MeetingMinutes: Structured output from meetings
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class MeetingType(str, Enum):
    """Types of meetings."""
    TEAM = "team"
    ONE_ON_ONE = "one_on_one"
    CONSULTATION = "consultation"
    REVIEW = "review"


class MeetingRole(str, Enum):
    """Roles in a meeting."""
    FACILITATOR = "facilitator"
    CONTRIBUTOR = "contributor"
    REVIEWER = "reviewer"
    OBSERVER = "observer"


class Participant(BaseModel):
    """A meeting participant."""

    agent_name: str = Field(..., description="Agent's name/persona")
    role: MeetingRole = Field(default=MeetingRole.CONTRIBUTOR, description="Role in meeting")
    expertise: list[str] = Field(default_factory=list, description="Areas of expertise")

    model_config = {"extra": "forbid"}


class Contribution(BaseModel):
    """A single contribution to a meeting."""

    agent_name: str = Field(..., description="Who made the contribution")
    content: str = Field(..., description="The contribution content")
    timestamp: datetime = Field(default_factory=datetime.now, description="When contributed")
    is_decision: bool = Field(default=False, description="Whether this is a decision point")
    references: list[str] = Field(default_factory=list, description="Referenced documents/artifacts")

    model_config = {"extra": "forbid"}


class ActionItem(BaseModel):
    """An action item from a meeting."""

    description: str = Field(..., description="What needs to be done")
    assignee: str = Field(..., description="Agent responsible")
    priority: str = Field(default="medium", description="Priority level")
    due_by: Optional[str] = Field(default=None, description="When it should be done")

    model_config = {"extra": "forbid"}


class MeetingMinutes(BaseModel):
    """
    Structured output from a meeting.
    
    Captures decisions, action items, and key discussion points.
    """

    meeting_id: str = Field(..., description="Unique meeting identifier")
    meeting_type: MeetingType = Field(..., description="Type of meeting")
    topic: str = Field(..., description="Main topic discussed")
    participants: list[str] = Field(default_factory=list, description="Who participated")

    started_at: datetime = Field(default_factory=datetime.now, description="Start time")
    ended_at: Optional[datetime] = Field(default=None, description="End time")

    summary: str = Field(default="", description="Brief summary of the meeting")
    key_points: list[str] = Field(default_factory=list, description="Key discussion points")
    decisions: list[str] = Field(default_factory=list, description="Decisions made")
    action_items: list[ActionItem] = Field(default_factory=list, description="Actions to take")

    contributions: list[Contribution] = Field(
        default_factory=list,
        description="All contributions made"
    )

    tokens_used: int = Field(default=0, description="Total tokens consumed")

    model_config = {"extra": "forbid"}

    def add_contribution(self, agent_name: str, content: str, is_decision: bool = False) -> None:
        """Add a contribution to the meeting."""
        self.contributions.append(
            Contribution(
                agent_name=agent_name,
                content=content,
                is_decision=is_decision,
            )
        )

    def add_decision(self, decision: str) -> None:
        """Record a decision."""
        self.decisions.append(decision)

    def add_action_item(
        self,
        description: str,
        assignee: str,
        priority: str = "medium"
    ) -> None:
        """Add an action item."""
        self.action_items.append(
            ActionItem(
                description=description,
                assignee=assignee,
                priority=priority,
            )
        )

    def close(self) -> None:
        """Close the meeting."""
        self.ended_at = datetime.now()


class MeetingConfig(BaseModel):
    """Configuration for a meeting."""

    topic: str = Field(..., description="Topic to discuss")
    participants: list[Participant] = Field(..., description="Who should participate")
    max_rounds: int = Field(default=5, description="Maximum discussion rounds")
    max_tokens: int = Field(default=10000, description="Token budget for meeting")
    require_consensus: bool = Field(default=False, description="Require all to agree")
    output_format: str = Field(default="minutes", description="Output format")

    model_config = {"extra": "forbid"}


class TeamMeeting(BaseModel):
    """
    A meeting with multiple agents.
    
    Used for planning, review, and group decisions.
    """

    config: MeetingConfig = Field(..., description="Meeting configuration")
    minutes: Optional[MeetingMinutes] = Field(default=None, description="Meeting output")

    model_config = {"extra": "forbid"}

    @classmethod
    def create(
        cls,
        topic: str,
        participants: list[str],
        max_rounds: int = 5,
        max_tokens: int = 10000,
    ) -> "TeamMeeting":
        """Create a new team meeting."""
        return cls(
            config=MeetingConfig(
                topic=topic,
                participants=[
                    Participant(agent_name=name) for name in participants
                ],
                max_rounds=max_rounds,
                max_tokens=max_tokens,
            )
        )


class OneOnOneMeeting(BaseModel):
    """
    A meeting between exactly two agents.
    
    Used for detailed collaboration, handoffs, and focused discussion.
    """

    initiator: str = Field(..., description="Agent who initiated")
    collaborator: str = Field(..., description="Agent collaborating")
    topic: str = Field(..., description="Topic of discussion")
    context: dict[str, Any] = Field(default_factory=dict, description="Shared context")

    max_exchanges: int = Field(default=10, description="Maximum back-and-forth")
    max_tokens: int = Field(default=5000, description="Token budget")

    minutes: Optional[MeetingMinutes] = Field(default=None, description="Meeting output")

    model_config = {"extra": "forbid"}

    @classmethod
    def create(
        cls,
        initiator: str,
        collaborator: str,
        topic: str,
        context: Optional[dict[str, Any]] = None,
    ) -> "OneOnOneMeeting":
        """Create a new one-on-one meeting."""
        return cls(
            initiator=initiator,
            collaborator=collaborator,
            topic=topic,
            context=context or {},
        )


class ConsultationMeeting(BaseModel):
    """
    A consultation where one agent seeks input from specialists.
    
    The consulting agent drives the agenda and collects responses.
    """

    consultant: str = Field(..., description="Agent seeking consultation")
    specialists: list[str] = Field(..., description="Specialist agents to consult")
    question: str = Field(..., description="Question being asked")
    context: dict[str, Any] = Field(default_factory=dict, description="Background context")

    max_tokens: int = Field(default=8000, description="Token budget")

    responses: dict[str, str] = Field(
        default_factory=dict,
        description="Responses from each specialist"
    )
    synthesis: Optional[str] = Field(default=None, description="Synthesized conclusion")

    model_config = {"extra": "forbid"}

    def add_response(self, specialist: str, response: str) -> None:
        """Record a specialist's response."""
        self.responses[specialist] = response
