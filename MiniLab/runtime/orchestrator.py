"""
OrchestratorRuntime: Core execution engine for MiniLab.

Provides:
- OrchestratorRuntime: Main entry point for running workflows
- Coordination between TaskGraph, agents, and verification
- Budget enforcement and checkpointing
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol
import json
import uuid

from pydantic import BaseModel, Field

from MiniLab.runtime.taskgraph import TaskGraph, TaskNode, TaskStatus
from MiniLab.runtime.runlog import (
    RunLog,
    RunStartedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    PlanStartedEvent,
    PlanCompletedEvent,
    TaskStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskRetryEvent,
    TeamMeetingStartedEvent,
    TeamMeetingCompletedEvent,
    VerificationPassedEvent,
    VerificationFailedEvent,
    CheckpointSavedEvent,
    BudgetWarningEvent,
)
from MiniLab.runtime.meeting import (
    TeamMeeting,
    OneOnOneMeeting,
    MeetingMinutes,
    MeetingType,
)
from MiniLab.runtime.verification import (
    VerificationReport,
    Verifier,
)


class AgentProtocol(Protocol):
    """Protocol for agents that can execute tasks."""

    name: str

    async def execute(
        self,
        task: TaskNode,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a task and return results."""
        ...

    async def participate_in_meeting(
        self,
        topic: str,
        context: dict[str, Any]
    ) -> str:
        """Contribute to a meeting discussion."""
        ...


class RuntimeConfig(BaseModel):
    """Configuration for the orchestrator runtime."""

    # Budget settings
    max_tokens: int = Field(default=100000, description="Maximum tokens for the run")
    budget_warning_threshold: float = Field(default=0.8, description="Warn at this fraction")

    # Execution settings
    max_retries: int = Field(default=3, description="Maximum retries per task")
    checkpoint_interval: int = Field(default=5, description="Checkpoint every N tasks")

    # Meeting settings
    max_meeting_rounds: int = Field(default=5, description="Maximum rounds per meeting")

    # Output settings
    output_dir: Optional[Path] = Field(default=None, description="Output directory")

    # Verification settings
    verify_outputs: bool = Field(default=True, description="Verify task outputs")

    model_config = {"extra": "forbid"}


class RuntimeState(BaseModel):
    """Current state of a run."""

    run_id: str = Field(..., description="Run identifier")
    status: str = Field(default="initializing", description="Current status")

    tokens_used: int = Field(default=0, description="Tokens consumed")
    tasks_completed: int = Field(default=0, description="Tasks finished")
    tasks_failed: int = Field(default=0, description="Tasks that failed")

    current_task_id: Optional[str] = Field(default=None, description="Currently executing task")
    current_agent: Optional[str] = Field(default=None, description="Currently active agent")

    started_at: datetime = Field(default_factory=datetime.now, description="Run start time")
    last_checkpoint: Optional[datetime] = Field(default=None, description="Last checkpoint time")

    model_config = {"extra": "forbid"}


class Checkpoint(BaseModel):
    """Checkpoint for run state persistence."""

    run_id: str
    state: RuntimeState
    task_graph: dict[str, Any]
    completed_results: dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"extra": "forbid"}


class OrchestratorRuntime:
    """
    Core execution engine for MiniLab workflows.
    
    Responsibilities:
    - Execute task graphs with proper ordering
    - Coordinate agent invocations
    - Run team and 1-on-1 meetings
    - Enforce token budgets
    - Verify outputs
    - Checkpoint progress
    
    Example:
        runtime = OrchestratorRuntime(config, agents)
        result = await runtime.run(goal="Analyze survival data")
    """

    def __init__(
        self,
        config: RuntimeConfig,
        agents: dict[str, AgentProtocol],
        verifier: Optional[Verifier] = None,
    ):
        """
        Initialize the runtime.
        
        Args:
            config: Runtime configuration
            agents: Available agents by name
            verifier: Optional output verifier
        """
        self.config = config
        self.agents = agents
        self.verifier = verifier

        self.run_id = ""
        self.state: Optional[RuntimeState] = None
        self.task_graph: Optional[TaskGraph] = None
        self.run_log: Optional[RunLog] = None
        self.results: dict[str, Any] = {}

        # Callbacks for extensibility
        self._on_task_complete: list[Callable] = []
        self._on_budget_warning: list[Callable] = []

    async def run(
        self,
        goal: str,
        initial_context: Optional[dict[str, Any]] = None,
        resume_from: Optional[Path] = None,
    ) -> dict[str, Any]:
        """
        Execute a complete run.
        
        Args:
            goal: High-level goal to accomplish
            initial_context: Starting context/data
            resume_from: Checkpoint path to resume from
            
        Returns:
            Dictionary with run results and artifacts
        """
        # Initialize or restore
        if resume_from:
            await self._restore_checkpoint(resume_from)
        else:
            await self._initialize_run(goal)

        # Log run start
        self.run_log.append(RunStartedEvent(
            run_id=self.run_id,
            goal=goal,
            config=self.config.model_dump(),
        ))

        try:
            # Phase 1: Planning
            await self._plan(goal, initial_context or {})

            # Phase 2: Execution
            await self._execute()

            # Phase 3: Final verification
            final_report = await self._final_verification()

            # Log completion
            self.run_log.append(RunCompletedEvent(
                run_id=self.run_id,
                summary=f"Completed {self.state.tasks_completed} tasks",
                artifacts=list(self.results.keys()),
            ))

            self.state.status = "completed"

            return {
                "run_id": self.run_id,
                "status": "completed",
                "results": self.results,
                "verification": final_report.summary() if final_report else None,
                "tokens_used": self.state.tokens_used,
                "tasks_completed": self.state.tasks_completed,
            }

        except Exception as e:
            self.run_log.append(RunFailedEvent(
                run_id=self.run_id,
                error=str(e),
                partial_results=self.results,
            ))
            self.state.status = "failed"
            raise

    async def _initialize_run(self, goal: str) -> None:
        """Initialize a new run."""
        self.run_id = f"{goal.replace(' ', '_')[:30]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        output_dir = self.config.output_dir or Path(f"outputs/{self.run_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.state = RuntimeState(run_id=self.run_id)
        self.task_graph = TaskGraph()
        self.run_log = RunLog(self.run_id, output_dir)
        self.results = {}

    async def _restore_checkpoint(self, checkpoint_path: Path) -> None:
        """Restore from a checkpoint."""
        with open(checkpoint_path) as f:
            data = json.load(f)

        checkpoint = Checkpoint.model_validate(data)

        self.run_id = checkpoint.run_id
        self.state = checkpoint.state
        self.task_graph = TaskGraph.model_validate(checkpoint.task_graph)
        self.results = checkpoint.completed_results

        output_dir = self.config.output_dir or Path(f"outputs/{self.run_id}")
        self.run_log = RunLog(self.run_id, output_dir)

    async def _plan(self, goal: str, context: dict[str, Any]) -> None:
        """
        Run planning phase to create task graph.
        
        This involves a team meeting with available agents to decompose
        the goal into tasks.
        """
        self.state.status = "planning"

        self.run_log.append(PlanStartedEvent(
            run_id=self.run_id,
            goal=goal,
        ))

        # Run planning meeting
        planning_meeting = TeamMeeting.create(
            topic=f"Planning: {goal}",
            participants=list(self.agents.keys()),
            max_rounds=3,
            max_tokens=self.config.max_tokens // 10,  # 10% for planning
        )

        minutes = await self.run_team_meeting(planning_meeting, context)

        # Extract tasks from meeting decisions/action items
        for i, action in enumerate(minutes.action_items):
            task = TaskNode(
                id=f"task_{i+1}",
                description=action.description,
                owner_agent=action.assignee,
            )
            self.task_graph.add_node(task)

        self.run_log.append(PlanCompletedEvent(
            run_id=self.run_id,
            task_count=len(self.task_graph.nodes),
        ))

    async def _execute(self) -> None:
        """Execute all tasks in the graph."""
        self.state.status = "executing"

        tasks_since_checkpoint = 0

        while not self.task_graph.is_complete():
            # Check budget
            if self.state.tokens_used >= self.config.max_tokens:
                raise RuntimeError("Token budget exceeded")

            # Warn if approaching budget
            if self.state.tokens_used >= self.config.max_tokens * self.config.budget_warning_threshold:
                self.run_log.append(BudgetWarningEvent(
                    run_id=self.run_id,
                    budget_used=self.state.tokens_used,
                    budget_limit=self.config.max_tokens,
                    percentage=self.state.tokens_used / self.config.max_tokens,
                ))
                for callback in self._on_budget_warning:
                    callback(self.state.tokens_used, self.config.max_tokens)

            # Update blocked nodes
            self.task_graph.blocked_nodes()

            # Get ready tasks
            ready = self.task_graph.ready_nodes()

            if not ready:
                # No tasks ready - either all done or all blocked
                break

            # Execute ready tasks (could parallelize in future)
            for task in ready:
                await self._execute_task(task)
                tasks_since_checkpoint += 1

                # Checkpoint periodically
                if tasks_since_checkpoint >= self.config.checkpoint_interval:
                    await self._save_checkpoint()
                    tasks_since_checkpoint = 0

    async def _execute_task(self, task: TaskNode) -> None:
        """Execute a single task."""
        self.state.current_task_id = task.id
        self.state.current_agent = task.owner_agent

        task.mark_running()

        self.run_log.append(TaskStartedEvent(
            run_id=self.run_id,
            task_id=task.id,
            task_description=task.description,
            owner_agent=task.owner_agent,
        ))

        try:
            # Get the assigned agent
            agent = self.agents.get(task.owner_agent)
            if not agent:
                raise ValueError(f"Unknown agent: {task.owner_agent}")

            # Build context from dependencies
            context = self._build_task_context(task)

            # Execute
            result = await agent.execute(task, context)

            # Verify if configured
            if self.config.verify_outputs and self.verifier:
                report = self.verifier.verify(task.id, result)
                if not report.passed:
                    raise ValueError(f"Verification failed: {report.failed_checks}")

            # Mark done
            task.mark_done(result)
            self.results[task.id] = result
            self.state.tasks_completed += 1

            self.run_log.append(TaskCompletedEvent(
                run_id=self.run_id,
                task_id=task.id,
                result_summary=str(result)[:100],
            ))

            # Callbacks
            for callback in self._on_task_complete:
                callback(task, result)

        except Exception as e:
            if task.can_retry():
                task.decrement_retries()
                task.status = TaskStatus.PENDING  # Allow retry

                self.run_log.append(TaskRetryEvent(
                    run_id=self.run_id,
                    task_id=task.id,
                    attempt=task.max_retries - task.retries,
                    reason=str(e),
                ))
            else:
                task.mark_failed(str(e))
                self.state.tasks_failed += 1

                self.run_log.append(TaskFailedEvent(
                    run_id=self.run_id,
                    task_id=task.id,
                    error=str(e),
                    retries_remaining=task.retries,
                ))

        self.state.current_task_id = None
        self.state.current_agent = None

    def _build_task_context(self, task: TaskNode) -> dict[str, Any]:
        """Build context for a task from its dependencies."""
        context = dict(task.inputs)

        for dep_id in task.depends_on:
            if dep_id in self.results:
                context[f"dep_{dep_id}"] = self.results[dep_id]

        return context

    async def run_team_meeting(
        self,
        meeting: TeamMeeting,
        context: dict[str, Any],
    ) -> MeetingMinutes:
        """
        Run a team meeting with multiple agents.
        
        Args:
            meeting: Meeting configuration
            context: Shared context for the meeting
            
        Returns:
            Meeting minutes with decisions and action items
        """
        minutes = MeetingMinutes(
            meeting_id=str(uuid.uuid4())[:8],
            meeting_type=MeetingType.TEAM,
            topic=meeting.config.topic,
            participants=[p.agent_name for p in meeting.config.participants],
        )

        self.run_log.append(TeamMeetingStartedEvent(
            run_id=self.run_id,
            topic=meeting.config.topic,
            participants=minutes.participants,
        ))

        # Run discussion rounds
        discussion_context = dict(context)
        discussion_context["topic"] = meeting.config.topic

        for round_num in range(meeting.config.max_rounds):
            for participant in meeting.config.participants:
                agent = self.agents.get(participant.agent_name)
                if not agent:
                    continue

                # Get agent's contribution
                contribution = await agent.participate_in_meeting(
                    meeting.config.topic,
                    discussion_context,
                )

                minutes.add_contribution(participant.agent_name, contribution)

                # Update context with contribution
                discussion_context[f"{participant.agent_name}_round_{round_num}"] = contribution

        minutes.close()
        meeting.minutes = minutes

        self.run_log.append(TeamMeetingCompletedEvent(
            run_id=self.run_id,
            topic=meeting.config.topic,
            decisions=minutes.decisions,
        ))

        return minutes

    async def run_1on1(
        self,
        meeting: OneOnOneMeeting,
    ) -> MeetingMinutes:
        """
        Run a one-on-one meeting between two agents.
        
        Args:
            meeting: Meeting configuration
            
        Returns:
            Meeting minutes
        """
        minutes = MeetingMinutes(
            meeting_id=str(uuid.uuid4())[:8],
            meeting_type=MeetingType.ONE_ON_ONE,
            topic=meeting.topic,
            participants=[meeting.initiator, meeting.collaborator],
        )

        initiator = self.agents.get(meeting.initiator)
        collaborator = self.agents.get(meeting.collaborator)

        if not initiator or not collaborator:
            raise ValueError("Missing agents for 1-on-1 meeting")

        context = dict(meeting.context)
        context["topic"] = meeting.topic

        # Alternating exchanges
        for exchange in range(meeting.max_exchanges):
            # Initiator speaks
            initiator_response = await initiator.participate_in_meeting(
                meeting.topic, context
            )
            minutes.add_contribution(meeting.initiator, initiator_response)
            context["last_exchange"] = initiator_response

            # Collaborator responds
            collaborator_response = await collaborator.participate_in_meeting(
                meeting.topic, context
            )
            minutes.add_contribution(meeting.collaborator, collaborator_response)
            context["last_exchange"] = collaborator_response

        minutes.close()
        meeting.minutes = minutes
        return minutes

    async def _final_verification(self) -> Optional[VerificationReport]:
        """Run final verification on all results."""
        if not self.config.verify_outputs or not self.verifier:
            return None

        from MiniLab.runtime.verification import VerificationReport

        report = VerificationReport(
            target_id=self.run_id,
            target_type="run",
            verifier_name="final",
        )

        for task_id, result in self.results.items():
            task_report = self.verifier.verify(task_id, result)

            if task_report.passed:
                self.run_log.append(VerificationPassedEvent(
                    run_id=self.run_id,
                    task_id=task_id,
                    checks_passed=len(task_report.checks),
                ))
            else:
                self.run_log.append(VerificationFailedEvent(
                    run_id=self.run_id,
                    task_id=task_id,
                    failures=[c.message for c in task_report.failed_checks],
                ))

            # Aggregate checks
            for check in task_report.checks:
                check.name = f"{task_id}_{check.name}"
                report.add_check(check)

        return report

    async def _save_checkpoint(self) -> None:
        """Save a checkpoint of current state."""
        output_dir = self.config.output_dir or Path(f"outputs/{self.run_id}")
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = Checkpoint(
            run_id=self.run_id,
            state=self.state,
            task_graph=self.task_graph.model_dump(),
            completed_results=self.results,
        )

        checkpoint_path = checkpoint_dir / f"checkpoint_{datetime.now().strftime('%H%M%S')}.json"

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint.model_dump(mode="json"), f, default=str, indent=2)

        self.state.last_checkpoint = datetime.now()

        self.run_log.append(CheckpointSavedEvent(
            run_id=self.run_id,
            checkpoint_path=str(checkpoint_path),
            tasks_completed=self.state.tasks_completed,
        ))

    def on_task_complete(self, callback: Callable[[TaskNode, dict], None]) -> None:
        """Register a callback for task completion."""
        self._on_task_complete.append(callback)

    def on_budget_warning(self, callback: Callable[[int, int], None]) -> None:
        """Register a callback for budget warnings."""
        self._on_budget_warning.append(callback)
