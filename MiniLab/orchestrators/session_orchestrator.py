"""
SessionOrchestrator: Core session lifecycle management.

Extracted from BohrOrchestrator to separate concerns:
- SessionOrchestrator: Session state, workflow execution, budget tracking
- BohrAgent: Decision-making about workflow order (as peer agent)

This allows Bohr to operate autonomously like other agents while
session management remains stable infrastructure.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from ..context import ContextManager
from ..agents import Agent
from ..agents.registry import create_agents
from ..storage import TranscriptLogger
from ..core import TokenAccount, get_token_account, ProjectWriter
from ..workflows import (
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
from ..llm_backends import AnthropicBackend


@dataclass
class SessionState:
    """
    Immutable session state container.
    
    Tracks project state, completed workflows, and accumulated context.
    Serializable for persistence and checkpointing.
    """
    session_id: str
    project_name: str
    project_path: Path
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    current_workflow: Optional[str] = None
    completed_workflows: list[str] = field(default_factory=list)  # type: ignore[assignment]
    workflow_results: dict[str, WorkflowResult] = field(default_factory=dict)  # type: ignore[assignment]
    context: dict[str, Any] = field(default_factory=dict)  # type: ignore[assignment]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "project_name": self.project_name,
            "project_path": str(self.project_path),
            "started_at": self.started_at,
            "current_workflow": self.current_workflow,
            "completed_workflows": self.completed_workflows,
            "workflow_results": {
                k: {
                    "status": v.status.value,
                    "summary": v.summary,
                    "artifacts": v.artifacts,
                }
                for k, v in self.workflow_results.items()
            },
        }
    
    def save(self) -> Path:
        """Save session state to disk."""
        session_path = self.project_path / "session.json"
        with open(session_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return session_path
    
    @classmethod
    def load(cls, session_path: Path) -> "SessionState":
        """Load session from disk."""
        with open(session_path) as f:
            data = json.load(f)
        return cls(
            session_id=data["session_id"],
            project_name=data["project_name"],
            project_path=Path(data["project_path"]),
            started_at=data.get("started_at", ""),
            current_workflow=data.get("current_workflow"),
            completed_workflows=data.get("completed_workflows", []),
        )


class SessionOrchestrator:
    """
    Core session lifecycle orchestrator.
    
    Responsibilities:
    - Session creation and persistence
    - Workflow execution (delegates to WorkflowModule classes)
    - Budget tracking and enforcement
    - Agent initialization
    - Transcript logging
    
    Does NOT decide:
    - Which workflow to run next (that's Bohr's job as agent)
    - Workflow-internal decisions (that's the lead agent's job)
    """
    
    SANDBOX_ROOT = Path(os.getenv(
        "MINILAB_SANDBOX",
        str(Path(__file__).parent.parent.parent / "Sandbox")
    ))
    
    WORKFLOW_MODULES = {
        "consultation": ConsultationModule,
        "literature_review": LiteratureReviewModule,
        "planning_committee": PlanningCommitteeModule,
        "execute_analysis": ExecuteAnalysisModule,
        "writeup_results": WriteupResultsModule,
        "critical_review": CriticalReviewModule,
    }
    
    # Budget thresholds (percentage)
    BUDGET_WARNING = 80
    BUDGET_CRITICAL = 95
    BUDGET_RESERVE = 5  # 5% graceful shutdown reserve
    
    def __init__(
        self,
        llm_backend: Optional[AnthropicBackend] = None,
        user_callback: Optional[Callable[[str], str]] = None,
        transcripts_dir: Optional[Path] = None,
        streaming_enabled: bool = False,
        on_stream_chunk: Optional[Callable[[str, str], None]] = None,
    ):
        self.llm_backend = llm_backend or AnthropicBackend(model="claude-sonnet-4-5", agent_id="session")
        self.user_callback = user_callback or self._default_user_input
        
        self.streaming_enabled = streaming_enabled
        self.on_stream_chunk = on_stream_chunk
        
        self.transcripts_dir = transcripts_dir or Path(__file__).parent.parent.parent / "Transcripts"
        self.transcript = TranscriptLogger(self.transcripts_dir)
        
        # Token accounting
        self.token_account = get_token_account()
        self.token_account.set_warning_callback(self._on_budget_warning)
        
        # Session state
        self.context_manager: Optional[ContextManager] = None
        self.tool_factory = None
        self.agents: dict[str, Agent] = {}
        self.session: Optional[SessionState] = None
        self._session_date: Optional[datetime] = None
        
        # Budget state
        self._token_budget: Optional[int] = None
        self._budget_percentage: float = 0.0
        self._budget_tight: bool = False
        self._budget_critical: bool = False
        
        # Interrupt handling
        self._interrupt_requested: bool = False
        self._exit_requested: bool = False
    
    @property
    def budget_remaining(self) -> Optional[int]:
        """Tokens remaining in budget, or None if no budget set."""
        if not self._token_budget:
            return None
        return max(0, self._token_budget - self.token_account.total_used)
    
    @property
    def budget_usable(self) -> Optional[int]:
        """Tokens usable (budget minus 5% reserve), or None if no budget."""
        if not self._token_budget:
            return None
        reserve = int(self._token_budget * (self.BUDGET_RESERVE / 100))
        return max(0, self._token_budget - self.token_account.total_used - reserve)
    
    @property
    def should_wrap_up(self) -> bool:
        """True if budget is tight and we should prioritize completion."""
        return self._budget_tight or self._budget_critical
    
    def _permission_callback(self, request: str) -> bool:
        """Permission callback for tools."""
        auto = os.getenv("MINILAB_AUTO_APPROVE", "").strip().lower()
        if auto in {"1", "true", "yes"}:
            return True
        if auto in {"0", "false", "no"}:
            return False
        
        try:
            response = self.user_callback(
                f"Permission required:\n{request}\n\nApprove? (y/N)"
            )
            return response.strip().lower() in {"y", "yes"}
        except Exception:
            return False
    
    def _on_budget_warning(self, used: int, budget: int, percentage: float) -> None:
        """Handle budget threshold events."""
        from ..utils import console
        
        self._budget_percentage = percentage
        
        if percentage >= self.BUDGET_CRITICAL:
            console.warning(f"⚠ Budget critical: {percentage:.0f}% ({used:,}/{budget:,})")
            self._budget_critical = True
            self.transcript.log_budget_warning(percentage, f"Critical: {percentage:.0f}%")
        elif percentage >= self.BUDGET_WARNING:
            console.warning(f"Budget tight: {percentage:.0f}% ({used:,}/{budget:,})")
            self._budget_tight = True
            self.transcript.log_budget_warning(percentage, f"Tight: {percentage:.0f}%")
    
    def _default_user_input(self, prompt: str) -> str:
        """Default user input via stdin."""
        from ..utils import console, Style
        print()
        console.agent_message("SESSION", prompt)
        return input(f"  {Style.BOLD}{Style.GREEN}▶ Your response:{Style.RESET} ").strip()
    
    async def create_session(
        self,
        user_request: str,
        project_name: str,
    ) -> SessionState:
        """
        Create a new session with the given project name.
        
        Args:
            user_request: User's initial request
            project_name: Project name (caller is responsible for generating)
        
        Returns:
            New SessionState
        """
        from ..utils import console
        
        self._session_date = datetime.now()
        self.token_account.reset()
        
        # Reset budget state
        self._budget_percentage = 0.0
        self._budget_tight = False
        self._budget_critical = False
        self._token_budget = None
        
        # Reset BudgetManager
        try:
            from ..config.budget_manager import BudgetManager
            BudgetManager.reset()
        except Exception:
            pass
        
        # Start transcript
        self.transcript.start_session(project_name)
        self.transcript.log_user_message(user_request)
        
        # Create project directory
        session_id = self._session_date.strftime("%Y%m%d_%H%M%S")
        project_path = self.SANDBOX_ROOT / project_name
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Create session state
        self.session = SessionState(
            session_id=session_id,
            project_name=project_name,
            project_path=project_path,
        )
        self.session.context["user_request"] = user_request
        
        # Initialize agents
        workspace_root = self.SANDBOX_ROOT.parent
        self.agents, self.context_manager, self.tool_factory = create_agents(
            workspace_root=workspace_root,
            input_callback=self._default_user_input,
            permission_callback=self._permission_callback,
            session_date=self._session_date,
        )
        
        # Configure agents
        for agent in self.agents.values():
            agent.set_transcript(self.transcript)
            if self.streaming_enabled:
                agent.enable_streaming(on_chunk=self.on_stream_chunk)
        
        self.session.save()
        return self.session
    
    async def resume_session(self, project_path: Path) -> SessionState:
        """Resume an existing session from disk."""
        session_path = project_path / "session.json"
        if not session_path.exists():
            raise ValueError(f"No session at {project_path}")
        
        self.session = SessionState.load(session_path)
        
        session_date = (
            datetime.fromisoformat(self.session.started_at)
            if self.session.started_at
            else datetime.now()
        )
        
        workspace_root = self.SANDBOX_ROOT.parent
        self.agents, self.context_manager, self.tool_factory = create_agents(
            workspace_root=workspace_root,
            input_callback=self._default_user_input,
            permission_callback=self._permission_callback,
            session_date=session_date,
        )
        
        for agent in self.agents.values():
            agent.set_transcript(self.transcript)
            if self.streaming_enabled:
                agent.enable_streaming(on_chunk=self.on_stream_chunk)
        
        return self.session
    
    def set_budget(self, token_budget: int, complexity: Optional[str] = None) -> None:
        """
        Set the session token budget.
        
        Args:
            token_budget: Total token budget
            complexity: Optional complexity for budget allocation
        """
        self._token_budget = token_budget
        self.token_account.set_budget(token_budget)
        self.transcript.set_token_budget(token_budget)
        
        # Initialize budget manager
        try:
            from ..config.budget_manager import get_budget_manager
            budget_mgr = get_budget_manager()
            budget_mgr.initialize_session(
                total_budget=token_budget,
                complexity=complexity or self._infer_complexity(token_budget)
            )
            
            # Set threshold callback
            def _on_threshold(threshold: int, level: str) -> None:
                self.transcript.log_budget_warning(threshold, f"Threshold: {threshold}% ({level})")
            budget_mgr.set_threshold_callback(_on_threshold)
        except Exception:
            pass
    
    def _infer_complexity(self, budget: int) -> str:
        """Infer complexity from budget size."""
        if budget < 200_000:
            return "simple"
        if budget < 700_000:
            return "moderate"
        if budget < 1_200_000:
            return "complex"
        return "exploratory"
    
    async def execute_workflow(
        self,
        workflow_name: str,
        objective: Optional[str] = None,
        autonomous: bool = True,
    ) -> WorkflowResult:
        """
        Execute a workflow by name.
        
        Args:
            workflow_name: Name of workflow to run
            objective: Optional objective for autonomous mode
            autonomous: Whether to use autonomous execution
        
        Returns:
            WorkflowResult
        """
        if workflow_name not in self.WORKFLOW_MODULES:
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=f"Unknown workflow: {workflow_name}",
            )
        
        # Update session state
        self.session.current_workflow = workflow_name
        self.session.save()
        
        # Create workflow instance
        workflow_class = self.WORKFLOW_MODULES[workflow_name]
        workflow = workflow_class(
            agents=self.agents,
            context_manager=self.context_manager,
            project_path=self.session.project_path,
        )
        
        # Prepare inputs
        inputs = self._prepare_inputs(workflow)
        
        # Track tokens
        from ..core.token_context import token_context
        start_tokens = self.token_account.total_used
        
        with token_context(workflow=workflow_name):
            if autonomous:
                lead_agent = workflow.primary_agents[0] if workflow.primary_agents else "darwin"
                supporting_context = self._build_supporting_context()
                exec_objective = objective or f"Complete {workflow_name}: {workflow.description}"
                
                result = await workflow.execute_autonomous(
                    inputs=inputs,
                    lead_agent=lead_agent,
                    objective=exec_objective,
                    supporting_context=supporting_context,
                )
            else:
                # Check for checkpoint
                checkpoint_path = self.session.project_path / "checkpoints" / f"{workflow_name}_checkpoint.json"
                checkpoint = None
                if checkpoint_path.exists():
                    try:
                        checkpoint = workflow.load_checkpoint(checkpoint_path)
                    except Exception:
                        pass
                
                result = await workflow.execute(inputs=inputs, checkpoint=checkpoint)
        
        # Record usage
        used_tokens = self.token_account.total_used - start_tokens
        try:
            from ..config.budget_manager import get_budget_manager
            get_budget_manager().record_usage(workflow_name, max(0, int(used_tokens)))
        except Exception:
            pass
        
        # Record usage to history for Bayesian learning
        try:
            from ..config.budget_history import get_budget_history
            complexity = self.session.context.get("complexity", 0.5)
            get_budget_history().record(workflow_name, int(used_tokens), complexity)
        except Exception:
            pass
        
        # Update session state
        self.session.workflow_results[workflow_name] = result
        if result.status == WorkflowStatus.COMPLETED:
            self.session.completed_workflows.append(workflow_name)
            if result.outputs:
                self.session.context.update(result.outputs)
        
        self.session.save()
        return result
    
    def _prepare_inputs(self, workflow: WorkflowModule) -> dict[str, Any]:
        """Prepare inputs for workflow from session context."""
        inputs = {}
        ctx = self.session.context
        
        # Standard mappings
        mappings = [
            "user_request", "project_spec", "literature_summary",
            "analysis_plan", "responsibilities", "analysis_results",
            "validation_report", "final_report", "user_preferences"
        ]
        for key in mappings:
            if key in ctx:
                inputs[key] = ctx[key]
        
        if self._token_budget:
            inputs["token_budget"] = self._token_budget
        
        # Data paths
        inputs["data_paths"] = self._scan_data_paths()
        
        if "research_topic" not in inputs and "user_request" in ctx:
            inputs["research_topic"] = ctx["user_request"]
        
        return inputs
    
    def _scan_data_paths(self) -> list[str]:
        """Scan for available data files."""
        data_root = Path("/Users/robertpatton/MiniLab/ReadData")
        paths = []
        if data_root.exists():
            for pattern in ["*.csv", "*.parquet"]:
                paths.extend(str(f) for f in data_root.rglob(pattern))
        return paths
    
    def _build_supporting_context(self) -> str:
        """Build context string from completed workflows."""
        parts = []
        
        for wf_name, wf_result in self.session.workflow_results.items():
            if wf_result.status == WorkflowStatus.COMPLETED:
                parts.append(f"## {wf_name.replace('_', ' ').title()}")
                parts.append(wf_result.summary or "Completed")
                if wf_result.artifacts:
                    parts.append(f"Artifacts: {', '.join(wf_result.artifacts)}")
                parts.append("")
        
        # Key context
        if self.session.context.get("project_spec"):
            parts.append("## Project Specification")
            parts.append(self.session.context["project_spec"][:2000])
        
        if self.session.context.get("analysis_plan"):
            parts.append("## Analysis Plan")
            parts.append(self.session.context["analysis_plan"][:2000])
        
        return "\n".join(parts)
    
    async def create_final_summary(self) -> str:
        """Create final session summary."""
        if not self.session:
            return "No session."
        
        bohr = self.agents.get("bohr")
        if not bohr:
            return self._simple_summary()
        
        results_summary = "\n".join([
            f"- {name}: {result.summary}"
            for name, result in self.session.workflow_results.items()
        ])
        
        summary_result = await bohr.execute_task(
            task=f"""Create final summary:

Project: {self.session.project_name}
Request: {self.session.context.get('user_request', 'N/A')}

Workflows:
{results_summary}

Write 2-3 paragraphs: what was asked, accomplished, key findings, next steps.""",
            project_name=self.session.project_name,
        )
        
        summary = summary_result.get("response", self._simple_summary())
        
        # Add budget breakdown
        breakdown = self._format_budget_breakdown()
        if breakdown:
            summary = summary.rstrip() + "\n\n" + breakdown
        
        # Write with ProjectWriter
        usage = self.token_account.usage_summary
        writer = ProjectWriter(
            project_path=self.session.project_path,
            project_name=self.session.project_name,
            context_manager=self.context_manager,
        )
        
        writer.write_session_summary(
            summary=summary,
            session_id=self.session.session_id,
            started_at=self.session.started_at,
            completed_workflows=self.session.completed_workflows,
            token_usage=usage,
        )
        
        return summary
    
    def _simple_summary(self) -> str:
        completed = ", ".join(self.session.completed_workflows) or "None"
        return f"Session {self.session.session_id} completed: {completed}"
    
    def _format_budget_breakdown(self) -> str:
        """Create budget breakdown section."""
        usage = self.token_account.usage_summary
        if usage.get("total_used", 0) <= 0:
            return ""
        
        lines = ["## Budget Breakdown", ""]
        lines.append(f"- Total: {usage.get('total_used', 0):,} tokens")
        
        if usage.get("budget"):
            lines.append(f"- Budget: {usage['total_used']:,} / {usage['budget']:,}")
        
        # Top consumers
        top = self.token_account.aggregate(keys=("workflow", "agent_id"))[:5]
        if top:
            lines.append("")
            lines.append("**Top Consumers**")
            for r in top:
                wf = r.get("workflow") or "(unscoped)"
                ag = r.get("agent_id") or "(unknown)"
                lines.append(f"- {wf} → {ag}: {r.get('total_tokens', 0):,}")
        
        return "\n".join(lines)
    
    def interrupt(self) -> None:
        """Request graceful interrupt."""
        self._interrupt_requested = True
        for agent in self.agents.values():
            agent.request_interrupt()
    
    def request_exit(self) -> None:
        """Request immediate exit."""
        self._exit_requested = True
        self._interrupt_requested = True
        for agent in self.agents.values():
            agent.request_interrupt()
    
    def save_transcript(self) -> Optional[Path]:
        """Save transcript and return path."""
        try:
            return self.transcript.save_transcript()
        except Exception:
            return None


def get_session_orchestrator(**kwargs) -> SessionOrchestrator:
    """Factory function for SessionOrchestrator."""
    return SessionOrchestrator(**kwargs)
