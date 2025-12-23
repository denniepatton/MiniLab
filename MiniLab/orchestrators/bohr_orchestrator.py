"""
Bohr Orchestrator - Primary user-facing orchestration layer.

Bohr acts as the conductor of the MiniLab symphony, making high-level
decisions about workflow selection and execution while delegating
specialized work to domain expert agents.

Key Responsibilities:
- User interaction and requirement gathering
- Workflow selection based on user intent
- Cross-workflow coordination
- Project state management
- Graceful error handling and recovery
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from pathlib import Path
from enum import Enum
import json
import asyncio
import os
from datetime import datetime

from ..context import ContextManager, ProjectState, TaskState
from ..agents import AgentRegistry, Agent
from ..agents.registry import create_agents
from ..agents.prompts import PromptBuilder
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


# Workflow types are now GUIDELINES, not fixed sequences.
# Bohr dynamically decides which workflows to run and in what order.
WORKFLOW_TYPES = {
    "consultation": "Understanding user needs, gathering requirements, setting scope",
    "literature_review": "Background research, citations, context gathering",
    "planning_committee": "Multi-agent deliberation on approach and methodology",
    "execute_analysis": "Data analysis, code execution, producing results",
    "writeup_results": "Documenting findings, creating reports and figures",
    "critical_review": "Quality assurance, validation, error checking",
}


@dataclass
class MiniLabSession:
    """
    State container for a MiniLab session.
    
    Tracks the current project, completed workflows, and artifacts.
    """
    session_id: str
    project_name: str
    project_path: Path
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    current_workflow: Optional[str] = None
    completed_workflows: list[str] = field(default_factory=list)
    workflow_results: dict[str, WorkflowResult] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
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
    def load(cls, session_path: Path) -> "MiniLabSession":
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


class BohrOrchestrator:
    """
    Primary orchestrator for MiniLab analysis sessions.
    
    Named after Niels Bohr, this orchestrator serves as the
    "atomic model" of the system - a central nucleus coordinating
    the electron-like specialized agents.
    
    Bohr's role:
    1. Greet users and understand their needs
    2. Select appropriate workflows
    3. Coordinate workflow execution
    4. Synthesize results across workflows
    5. Handle interrupts and errors gracefully
    """
    
    # Sandbox root for all projects (configurable via environment)
    SANDBOX_ROOT = Path(os.getenv(
        "MINILAB_SANDBOX",
        str(Path(__file__).parent.parent.parent / "Sandbox")
    ))
    
    # Workflow class mapping
    WORKFLOW_MODULES = {
        "consultation": ConsultationModule,
        "literature_review": LiteratureReviewModule,
        "planning_committee": PlanningCommitteeModule,
        "execute_analysis": ExecuteAnalysisModule,
        "writeup_results": WriteupResultsModule,
        "critical_review": CriticalReviewModule,
    }
    
    # Workflow GUIDELINES - suggestions for Bohr, not rigid sequences
    # Bohr decides dynamically based on project needs and budget
    WORKFLOW_GUIDELINES = {
        "brainstorming": {
            "typical_phases": ["consultation", "planning_committee"],
            "description": "Open-ended exploration and idea generation",
            "usually_skip": ["execute_analysis", "writeup_results"],
        },
        "literature_review": {
            "typical_phases": ["consultation", "literature_review"],
            "description": "Background research and citation gathering",
            "usually_skip": ["execute_analysis"],
        },
        "full_analysis": {
            "typical_phases": ["consultation", "literature_review", "planning_committee", "execute_analysis", "writeup_results", "critical_review"],
            "description": "Complete research pipeline from start to finish",
            "usually_skip": [],
        },
        "data_exploration": {
            "typical_phases": ["consultation", "execute_analysis", "writeup_results"],
            "description": "Exploratory data analysis focus",
            "usually_skip": ["literature_review", "planning_committee"],
        },
        "continuation": {
            "typical_phases": ["consultation", "planning_committee", "execute_analysis", "writeup_results", "critical_review"],
            "description": "Continue work on an existing project",
            "usually_skip": [],
        },
    }
    
    def __init__(
        self,
        llm_backend: Optional[AnthropicBackend] = None,
        user_callback: Optional[Callable[[str], str]] = None,
        transcripts_dir: Optional[Path] = None,
        streaming_enabled: bool = False,
        on_stream_chunk: Optional[Callable[[str, str], None]] = None,
        autonomous_mode: bool = True,  # Enable agent-driven workflow execution by default
    ):
        """
        Initialize the Bohr orchestrator.
        
        Args:
            llm_backend: LLM backend for agent communication
            user_callback: Function to get user input (for non-interactive)
            transcripts_dir: Directory for saving transcripts
            streaming_enabled: Enable streaming output from agents
            on_stream_chunk: Callback for streaming chunks (agent_id, text)
            autonomous_mode: Enable agent-driven autonomous workflow execution
        """
        self.llm_backend = llm_backend or AnthropicBackend(model="claude-sonnet-4-5", agent_id="bohr")
        self.user_callback = user_callback or self._default_user_input
        
        # Streaming configuration
        self.streaming_enabled = streaming_enabled
        self.on_stream_chunk = on_stream_chunk
        
        # Autonomous mode - let agents drive workflow execution
        # Can be overridden by MINILAB_AUTONOMOUS env var
        env_autonomous = os.getenv("MINILAB_AUTONOMOUS", "").lower()
        if env_autonomous == "false" or env_autonomous == "0":
            self.autonomous_mode = False
        elif env_autonomous == "true" or env_autonomous == "1":
            self.autonomous_mode = True
        else:
            self.autonomous_mode = autonomous_mode
        
        # Set up transcript logger
        self.transcripts_dir = transcripts_dir or Path(__file__).parent.parent.parent / "Transcripts"
        self.transcript = TranscriptLogger(self.transcripts_dir)
        
        # Initialize TokenAccount singleton
        self.token_account = get_token_account()
        
        # Set up warning callback for token budget
        self.token_account.set_warning_callback(self._on_budget_warning)
        
        self.context_manager: Optional[ContextManager] = None
        self.tool_factory = None  # Set during session start
        self.agents: dict[str, Agent] = {}
        self.session: Optional[MiniLabSession] = None
        self._session_date: Optional[datetime] = None
        
        self._interrupt_requested = False
        self._exit_requested = False
        self._token_budget: Optional[int] = None
        self._tokens_used: int = 0

    def _permission_callback(self, request: str) -> bool:
        """Permission callback for tools (terminal/env). Defaults to interactive approval."""
        # Allow non-interactive runs to auto-approve if explicitly configured.
        auto = os.getenv("MINILAB_AUTO_APPROVE", "").strip().lower()
        if auto in {"1", "true", "yes"}:
            return True
        if auto in {"0", "false", "no"}:
            return False

        try:
            response = self.user_callback(
                "Permission required for an operation:\n"
                f"{request}\n\nApprove? (y/N)"
            )
            return response.strip().lower() in {"y", "yes"}
        except Exception:
            return False

    def _infer_complexity(self, token_budget: Optional[int]) -> str:
        """Infer a reasonable complexity tier when consultation doesn't specify one."""
        if not token_budget:
            return "moderate"
        if token_budget < 200_000:
            return "simple"
        if token_budget < 700_000:
            return "moderate"
        if token_budget < 1_200_000:
            return "complex"
        return "exploratory"
    
    def _on_budget_warning(self, used: int, budget: int, percentage: float) -> None:
        """Callback when token budget warnings are triggered."""
        from ..utils import console
        
        if percentage >= 95:
            console.warning(f"⚠ Token budget critical: {percentage:.0f}% used ({used:,}/{budget:,})")
            console.agent_message("BOHR", "We're nearly at our budget limit. I'll wrap up the current task.")
        elif percentage >= 80:
            console.warning(f"Token budget at {percentage:.0f}% ({used:,}/{budget:,})")
        elif percentage >= 60:
            console.info(f"Budget update: {percentage:.0f}% used ({used:,}/{budget:,})")
        
        # Log to transcript
        self.transcript.log_budget_warning(percentage, f"Budget at {percentage:.0f}% ({used:,}/{budget:,})")
    
    def _default_user_input(self, prompt: str) -> str:
        """Default user input via stdin."""
        from ..utils import console, Style
        print()  # Clean line
        console.agent_message("BOHR", prompt)
        print()
        return input(f"  {Style.BOLD}{Style.GREEN}▶ Your response:{Style.RESET} ").strip()
    
    def _user_input_callback(self, prompt: str, options: Optional[list[str]] = None) -> str:
        """Callback for tools that need user input."""
        from ..utils import Style
        print()  # Clean line
        if options:
            print(f"\n[BOHR]: {prompt}")
            for i, opt in enumerate(options, 1):
                print(f"  {i}. {opt}")
            return input(f"  {Style.BOLD}{Style.GREEN}▶ Your choice:{Style.RESET} ").strip()
        print(f"\n[BOHR]: {prompt}")
        print()
        return input(f"  {Style.BOLD}{Style.GREEN}▶ Your response:{Style.RESET} ").strip()
    
    async def start_session(
        self,
        user_request: str,
        project_name: Optional[str] = None,
    ) -> MiniLabSession:
        """
        Start a new MiniLab session.
        
        Args:
            user_request: Initial user request/question
            project_name: Optional project name (if provided, skips Bohr naming)
            
        Returns:
            MiniLabSession tracking the analysis
        """
        from ..utils import console
        
        # Set session date (used throughout for consistent timestamps)
        self._session_date = datetime.now()
        
        # Reset TokenAccount for new session
        self.token_account.reset()

        # Start transcript as early as possible so project naming + consultation are captured.
        # We'll update the session name once the project name is finalized.
        provisional_name = f"pending_{self._session_date.strftime('%Y%m%d_%H%M%S')}"
        self.transcript.start_session(provisional_name)
        self.transcript.log_user_message(user_request)

        # Reset BudgetManager singleton for new session
        try:
            from ..config.budget_manager import BudgetManager
            BudgetManager.reset()
        except Exception:
            pass
        
        # Have Bohr generate and confirm project name
        if not project_name:
            project_name = await self._generate_project_name_interactive(user_request)
        self.transcript.update_session_name(project_name)
        
        # Create project directory
        session_id = self._session_date.strftime("%Y%m%d_%H%M%S")
        project_path = self.SANDBOX_ROOT / project_name
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize session
        self.session = MiniLabSession(
            session_id=session_id,
            project_name=project_name,
            project_path=project_path,
        )
        self.session.context["user_request"] = user_request
        
        # Transcript already started above (so naming prompts are included)
        
        # Initialize agents with context manager and tool factory
        # workspace_root should be the parent of Sandbox, not the project path
        workspace_root = self.SANDBOX_ROOT.parent
        self.agents, self.context_manager, self.tool_factory = create_agents(
            workspace_root=workspace_root,
            input_callback=self._user_input_callback,
            permission_callback=self._permission_callback,
            session_date=self._session_date,  # Pass session date to agents
        )
        
        # Set transcript logger on all agents and enable streaming if configured
        for agent in self.agents.values():
            agent.set_transcript(self.transcript)
            if self.streaming_enabled:
                agent.enable_streaming(on_chunk=self.on_stream_chunk)
        
        # Save initial session state (log internally, no console spam)
        self._log(f"Session started: {session_id}", console_print=False)
        self._log(f"Project: {project_name}", console_print=False)
        self._log(f"Request: {user_request}", console_print=False)
        self.session.save()
        
        return self.session
    
    async def resume_session(self, project_path: Path) -> MiniLabSession:
        """
        Resume an existing session.
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Restored MiniLabSession
        """
        session_path = project_path / "session.json"
        if not session_path.exists():
            raise ValueError(f"No session found at {project_path}")
        
        self.session = MiniLabSession.load(session_path)
        
        # Reinitialize agents with context manager and tool factory
        # workspace_root should be the parent of Sandbox, not the project path
        workspace_root = self.SANDBOX_ROOT.parent
        
        # Use session started_at for date injection, or current time if not available
        session_date = (
            datetime.fromisoformat(self.session.started_at)
            if getattr(self.session, "started_at", None)
            else datetime.now()
        )
        
        self.agents, self.context_manager, self.tool_factory = create_agents(
            workspace_root=workspace_root,
            input_callback=self._user_input_callback,
            permission_callback=self._permission_callback,
            session_date=session_date,
        )
        
        # Set transcript logger on resumed agents and enable streaming if configured
        for agent in self.agents.values():
            agent.set_transcript(self.transcript)
            if self.streaming_enabled:
                agent.enable_streaming(on_chunk=self.on_stream_chunk)
        
        self._log(f"Session resumed: {self.session.session_id}")
        
        return self.session
    
    async def run(self) -> dict[str, Any]:
        """
        Run the orchestration loop with AGENT-DRIVEN workflow planning.
        
        Bohr dynamically decides which workflows to run and in what order,
        based on project needs, budget, and progress. No fixed sequences.
        
        Returns:
            Final results dictionary
        """
        from ..utils import console, Spinner, Style, StatusIcon
        
        if not self.session:
            raise RuntimeError("No active session. Call start_session first.")
        
        try:
            # Always start with consultation to understand user needs
            # Consultation is the ONE mandatory phase - everything else is dynamic
            console.agent_message("BOHR", "Let's start by understanding exactly what you need. I'll then decide the best approach based on your goals and budget.")
            
            # Step 1: Execute consultation (mandatory)
            results = {}
            phase_num = 1
            
            spinner = Spinner("Running Consultation")
            spinner.start()
            self.transcript.log_stage_transition("consultation", "Starting Consultation Phase")
            self.session.current_workflow = "consultation"
            self.session.save()

            from ..core.token_context import token_context

            start_tokens = self.token_account.total_used
            with token_context(workflow="consultation"):
                consultation_result = await self._execute_workflow("consultation")
            used_tokens = self.token_account.total_used - start_tokens

            # Record truthful workflow usage into BudgetManager (if initialized)
            try:
                from ..config.budget_manager import get_budget_manager
                get_budget_manager().record_usage("consultation", max(0, int(used_tokens)))
            except Exception:
                pass
            results["consultation"] = consultation_result
            self.session.workflow_results["consultation"] = consultation_result
            
            if consultation_result.status == WorkflowStatus.COMPLETED:
                self.session.completed_workflows.append("consultation")
                spinner.stop("Consultation completed")
                
                # Extract key outputs from consultation
                if consultation_result.outputs:
                    self.session.context.update(consultation_result.outputs)
                    
                    # Set up budget
                    if consultation_result.outputs.get("token_budget"):
                        self._token_budget = consultation_result.outputs["token_budget"]
                        self.token_account.set_budget(self._token_budget)
                        self.transcript.set_token_budget(self._token_budget)

                        # Initialize dynamic workflow budgets
                        try:
                            from ..config.budget_manager import get_budget_manager

                            complexity = (
                                consultation_result.outputs.get("complexity")
                                or consultation_result.outputs.get("project_complexity")
                                or self._infer_complexity(self._token_budget)
                            )
                            budget_mgr = get_budget_manager()
                            budget_mgr.initialize_session(total_budget=self._token_budget, complexity=str(complexity))

                            # Mirror budget threshold events into transcript/logs
                            def _on_threshold(threshold: int, level: str) -> None:
                                try:
                                    from ..utils import console
                                    console.info(f"Budget threshold crossed: {threshold}% ({level})")
                                except Exception:
                                    pass
                                try:
                                    self.transcript.log_budget_warning(threshold, f"Budget threshold crossed: {threshold}% ({level})")
                                except Exception:
                                    pass

                            budget_mgr.set_threshold_callback(_on_threshold)
                        except Exception:
                            pass

                        self._show_post_consultation_summary(consultation_result)
                    
                    # Set up user preferences for contextual autonomy
                    if consultation_result.outputs.get("user_preferences"):
                        self.tool_factory.set_user_preferences(consultation_result.outputs["user_preferences"])
            else:
                spinner.stop_error(f"Consultation failed: {consultation_result.error}")
                return results  # Can't continue without consultation
            
            self.session.save()
            
            # Step 2: Agent-driven planning loop
            # Bohr decides what to do next based on project state, not fixed sequences
            completed_phases = ["consultation"]
            available_workflows = ["literature_review", "planning_committee", "execute_analysis", "writeup_results", "critical_review"]
            
            while True:
                # Check for interrupts
                if self._interrupt_requested or self._exit_requested:
                    if self._exit_requested:
                        console.info("Exiting as requested")
                        self._log("User requested exit", console_print=False)
                    else:
                        console.warning("Interrupt requested, pausing execution")
                    break
                
                # Budget check and graceful completion
                if self._token_budget:
                    current_usage = self.token_account.total_used
                    budget_pct = (current_usage / self._token_budget) * 100
                    
                    # Hard stop at budget
                    if current_usage >= self._token_budget:
                        console.error(f"Budget exceeded ({current_usage:,}/{self._token_budget:,}). Stopping.")
                        console.agent_message("BOHR", "We've hit our budget limit. Let me save our progress.")
                        self.transcript.log_budget_warning(100, "Budget exceeded - stopping")
                        break
                    
                    # Near budget - let Bohr know to wrap up
                    if budget_pct >= 85 and "writeup_results" not in completed_phases:
                        console.warning(f"Budget at {budget_pct:.0f}% - Bohr will prioritize completion")
                
                # Ask Bohr: What should we do next?
                next_action = await self._bohr_decide_next_action(
                    completed_phases=completed_phases,
                    available_workflows=[w for w in available_workflows if w not in completed_phases],
                    project_context=self.session.context,
                    budget_remaining=(self._token_budget - self.token_account.total_used) if self._token_budget else None,
                )
                
                if next_action["action"] == "done":
                    # Bohr has decided we're finished
                    console.agent_message("BOHR", next_action.get("reasoning", "I believe we've accomplished our goals."))
                    break
                
                elif next_action["action"] == "run_workflow":
                    workflow_name = next_action["workflow"]
                    phase_num += 1
                    
                    # Run the chosen workflow
                    console.info(f"Phase {phase_num}: {workflow_name.replace('_', ' ').title()}")
                    console.agent_message("BOHR", next_action.get("reasoning", f"Let's proceed with {workflow_name}."))
                    
                    spinner = Spinner(f"Running {workflow_name.replace('_', ' ').title()}")
                    spinner.start()
                    self.transcript.log_stage_transition(workflow_name, f"Starting {workflow_name.replace('_', ' ').title()}")
                    self.session.current_workflow = workflow_name
                    self.session.save()
                    
                    # Use autonomous mode if enabled, with dynamic objective
                    objective = next_action.get("objective") or next_action.get("reasoning")

                    from ..core.token_context import token_context
                    start_tokens = self.token_account.total_used
                    with token_context(workflow=workflow_name):
                        result = await self._execute_workflow(
                            workflow_name,
                            autonomous=self.autonomous_mode,
                            objective=objective,
                        )
                    used_tokens = self.token_account.total_used - start_tokens

                    # Record truthful workflow usage into BudgetManager (if initialized)
                    try:
                        from ..config.budget_manager import get_budget_manager
                        get_budget_manager().record_usage(workflow_name, max(0, int(used_tokens)))
                    except Exception:
                        pass
                    results[workflow_name] = result
                    self.session.workflow_results[workflow_name] = result
                    
                    if result.status == WorkflowStatus.COMPLETED:
                        self.session.completed_workflows.append(workflow_name)
                        completed_phases.append(workflow_name)
                        spinner.stop(f"{workflow_name} completed")
                        self._log(f"Completed: {workflow_name}", console_print=False)
                        self.transcript.log_system_event(
                            "workflow_complete",
                            f"{workflow_name} completed successfully",
                            {"summary": result.summary} if result.summary else None
                        )
                        
                        # Update context with outputs
                        if result.outputs:
                            self.session.context.update(result.outputs)
                    
                    elif result.status == WorkflowStatus.FAILED:
                        spinner.stop_error(f"{workflow_name} failed: {result.error}")
                        self._log(f"Failed: {workflow_name} - {result.error}", console_print=False)
                        self.transcript.log_system_event("workflow_failed", f"{workflow_name} failed", {"error": result.error})
                        
                        # Ask Bohr how to handle the failure
                        proceed = await self._handle_workflow_failure(workflow_name, result)
                        if not proceed:
                            break
                        # Mark as attempted so we don't loop forever
                        completed_phases.append(f"{workflow_name}_failed")
                    else:
                        spinner.stop(f"{workflow_name} finished: {result.status.value}")
                        completed_phases.append(workflow_name)
                    
                    self.session.save()
                
                else:
                    # Unknown action - shouldn't happen, but be safe
                    console.warning(f"Unknown action from Bohr: {next_action}")
                    break
            
            # Final summary
            console.info("Generating final summary...")
            from ..core.token_context import token_context
            with token_context(workflow="final_summary"):
                final_summary = await self._create_final_summary()
            results["final_summary"] = final_summary
            
            self.session.current_workflow = None
            self.session.save()
            
            # Save transcript
            try:
                transcript_path = self.transcript.save_transcript()
                self._log(f"Transcript saved to: {transcript_path}", console_print=False)
            except Exception as e:
                self._log(f"Failed to save transcript: {e}", console_print=False)
            
            return results
            
        except Exception as e:
            self._log(f"Orchestration error: {str(e)}")
            if self.session:
                self.session.save()
            try:
                self.transcript.save_transcript()
            except Exception:
                pass
            raise
    
    async def _bohr_decide_next_action(
        self,
        completed_phases: list[str],
        available_workflows: list[str],
        project_context: dict[str, Any],
        budget_remaining: Optional[int],
    ) -> dict[str, Any]:
        """
        Have Bohr intelligently decide what to do next.
        
        This is the core of agent-driven planning - Bohr assesses the project
        state and decides the optimal next step, not a fixed sequence.
        
        Returns:
            {"action": "run_workflow", "workflow": "name", "reasoning": "why"}
            or {"action": "done", "reasoning": "why we're finished"}
        """
        import json as json_module
        
        # Build context summary for Bohr
        project_spec = project_context.get("project_spec", "Not yet defined")
        user_request = project_context.get("user_request", "")
        user_preferences = project_context.get("user_preferences", "")
        
        # Summarize what's been accomplished
        accomplishments = []
        if "literature_summary" in project_context:
            accomplishments.append("- Literature review completed")
        if "analysis_plan" in project_context:
            accomplishments.append("- Analysis plan created")
        if "analysis_results" in project_context:
            accomplishments.append("- Analysis executed")
        if "final_report" in project_context:
            accomplishments.append("- Results written up")
        if "validation_report" in project_context:
            accomplishments.append("- Critical review completed")
        
        accomplishments_str = "\n".join(accomplishments) if accomplishments else "- Consultation complete, ready to proceed"
        
        # Budget awareness
        budget_info = ""
        if budget_remaining is not None and self._token_budget:
            budget_pct = ((self._token_budget - budget_remaining) / self._token_budget) * 100
            budget_info = f"\nBudget Status: {budget_pct:.0f}% used ({budget_remaining:,} tokens remaining)"
            if budget_pct >= 85:
                budget_info += "\n⚠️ BUDGET LOW: Prioritize completion tasks (writeup_results) over new work."
        
        messages = [
            {"role": "system", "content": f"""You are Bohr, deciding the next step for this research project.

Your job is to intelligently orchestrate the project, not follow a rigid sequence.
Consider: What does this project NEED based on its current state and goals?

AVAILABLE WORKFLOWS (you can run or skip any of these):
- literature_review: Background research, citations, context gathering
- planning_committee: Multi-agent deliberation on approach and methodology  
- execute_analysis: Data analysis, code execution, producing results
- writeup_results: Documenting findings, creating reports and figures
- critical_review: Quality assurance, validation, error checking

GUIDELINES for typical projects (not rules - use your judgment):
- Full analysis: Usually needs literature → planning → execution → writeup → review
- Literature-only: Just literature_review, skip execution
- Data exploration: Can skip literature and planning, go straight to analysis
- Brainstorming: Planning committee focus, may skip analysis
- Quick projects: May combine or skip steps as appropriate

RESPOND WITH ONLY A JSON OBJECT:
{{"action": "run_workflow", "workflow": "workflow_name", "reasoning": "one sentence why"}}
or
{{"action": "done", "reasoning": "one sentence why we're finished"}}

Consider the user's preferences when deciding - if they want thorough work, don't skip steps.
If they want it quick, skip less critical phases."""},
            {"role": "user", "content": f"""PROJECT STATE:

Original Request: {user_request}

User Preferences: {user_preferences if user_preferences else "No specific preferences"}

Completed Phases: {', '.join(completed_phases)}

Current Accomplishments:
{accomplishments_str}

Available Workflows (not yet run): {', '.join(available_workflows) if available_workflows else 'All complete'}
{budget_info}

What should we do next?"""}
        ]
        
        try:
            response = await self.llm_backend.acomplete(messages)
            
            # Parse JSON
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            data = json_module.loads(response)
            
            # Validate response
            if data.get("action") == "run_workflow":
                workflow = data.get("workflow", "")
                if workflow in available_workflows:
                    return data
                else:
                    # Invalid workflow - default to done if nothing available
                    if not available_workflows:
                        return {"action": "done", "reasoning": "All available workflows completed."}
                    # Pick first available
                    return {"action": "run_workflow", "workflow": available_workflows[0], "reasoning": f"Proceeding with {available_workflows[0]}"}
            
            return data  # Return done or whatever was decided
            
        except Exception as e:
            # On error, make a sensible default decision
            if not available_workflows:
                return {"action": "done", "reasoning": "All phases complete."}
            
            # If we haven't done writeup and have results, do writeup
            if "writeup_results" in available_workflows and "analysis_results" in project_context:
                return {"action": "run_workflow", "workflow": "writeup_results", "reasoning": "Writing up analysis results."}
            
            # Otherwise, do the first available
            return {"action": "run_workflow", "workflow": available_workflows[0], "reasoning": f"Proceeding with {available_workflows[0]}."}
    
    async def _determine_workflow_type(self) -> str:
        """
        Legacy method - Use Bohr to classify the overall project type.
        
        Note: This is now only used for initial classification hints.
        Actual workflow sequencing is handled by _bohr_decide_next_action.
        
        Returns:
            Workflow type string (for informational purposes)
        """
        import json as json_module
        
        user_request = self.session.context.get("user_request", "")
        
        try:
            messages = [
                {"role": "system", "content": """Classify this research request briefly.

RESPOND WITH ONLY A JSON OBJECT:
{"type": "TYPE_NAME", "reasoning": "brief explanation"}

Valid TYPE_NAME values:
- brainstorming: Exploring ideas, hypothesis generation
- literature_review: Background research, citations
- full_analysis: Complete project with data analysis
- data_exploration: EDA, understanding data
- continuation: Continuing existing work"""},
                {"role": "user", "content": f"Classify: {user_request}"}
            ]
            
            response = await self.llm_backend.acomplete(messages)
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            data = json_module.loads(response)
            return data.get("type", "full_analysis")
                    
        except Exception:
            return "full_analysis"
    
    async def _execute_workflow(
        self,
        workflow_name: str,
        autonomous: bool = False,
        objective: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Execute a specific workflow module.
        
        Args:
            workflow_name: Name of workflow to execute
            autonomous: If True, use autonomous execution mode (agent-driven)
            objective: High-level objective for autonomous mode
            
        Returns:
            WorkflowResult from execution
        """
        if workflow_name not in self.WORKFLOW_MODULES:
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=f"Unknown workflow: {workflow_name}",
            )
        
        # Instantiate workflow module
        workflow_class = self.WORKFLOW_MODULES[workflow_name]
        workflow = workflow_class(
            agents=self.agents,
            context_manager=self.context_manager,
            project_path=self.session.project_path,
        )
        
        # Prepare inputs from session context
        inputs = self._prepare_workflow_inputs(workflow)
        
        # Autonomous mode - let lead agent decide approach
        if autonomous:
            lead_agent = workflow.primary_agents[0] if workflow.primary_agents else "darwin"
            
            # Build context from prior workflow results
            supporting_context = self._build_supporting_context()
            
            # Use provided objective or generate from workflow description
            exec_objective = objective or f"Complete the {workflow_name} phase: {workflow.description}"
            
            result = await workflow.execute_autonomous(
                inputs=inputs,
                lead_agent=lead_agent,
                objective=exec_objective,
                supporting_context=supporting_context,
            )
            return result
        
        # Standard mode - structured execution with checkpoint support
        checkpoint_path = self.session.project_path / "checkpoints" / f"{workflow_name}_checkpoint.json"
        checkpoint = None
        if checkpoint_path.exists():
            try:
                checkpoint = workflow.load_checkpoint(checkpoint_path)
                self._log(f"Resuming {workflow_name} from checkpoint")
            except Exception:
                pass  # Start fresh if checkpoint is invalid
        
        # Execute workflow
        result = await workflow.execute(inputs=inputs, checkpoint=checkpoint)
        
        return result
    
    def _build_supporting_context(self) -> str:
        """Build context string from completed workflows for autonomous execution."""
        context_parts = []
        
        for wf_name, wf_result in self.session.workflow_results.items():
            if wf_result.status == WorkflowStatus.COMPLETED:
                context_parts.append(f"## {wf_name.replace('_', ' ').title()} Results")
                context_parts.append(wf_result.summary or "Completed successfully")
                if wf_result.artifacts:
                    context_parts.append(f"Artifacts: {', '.join(wf_result.artifacts)}")
                context_parts.append("")
        
        # Add key session context
        if self.session.context.get("project_spec"):
            context_parts.append("## Project Specification")
            context_parts.append(self.session.context["project_spec"][:2000])
            context_parts.append("")
        
        if self.session.context.get("analysis_plan"):
            context_parts.append("## Analysis Plan")
            context_parts.append(self.session.context["analysis_plan"][:2000])
            context_parts.append("")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _prepare_workflow_inputs(self, workflow: WorkflowModule) -> dict[str, Any]:
        """
        Prepare inputs for a workflow from session context.
        
        Args:
            workflow: Workflow module to prepare inputs for
            
        Returns:
            Dictionary of inputs
        """
        inputs = {}
        
        # Map context keys to workflow inputs
        context = self.session.context
        
        # Common mappings
        if "user_request" in context:
            inputs["user_request"] = context["user_request"]
        if "project_spec" in context:
            inputs["project_spec"] = context["project_spec"]
        if "literature_summary" in context:
            inputs["literature_summary"] = context["literature_summary"]
        if "analysis_plan" in context:
            inputs["analysis_plan"] = context["analysis_plan"]
        if "responsibilities" in context:
            inputs["responsibilities"] = context["responsibilities"]
        if "analysis_results" in context:
            inputs["analysis_results"] = context["analysis_results"]
        
        # Pass token budget to workflows that can use it for mode selection
        if self._token_budget:
            inputs["token_budget"] = self._token_budget
        if "validation_report" in context:
            inputs["validation_report"] = context["validation_report"]
        if "final_report" in context:
            inputs["final_report"] = context["final_report"]
        
        # Pass user preferences for contextual autonomy (natural language, not levels)
        # This enables agents to understand user's communication style preferences
        if "user_preferences" in context:
            inputs["user_preferences"] = context["user_preferences"]
        
        # Data paths - scan ReadData
        data_paths = self._scan_data_paths()
        inputs["data_paths"] = data_paths
        
        # Research topic from user request
        if "research_topic" not in inputs and "user_request" in context:
            inputs["research_topic"] = context["user_request"]
        
        return inputs
    
    def _show_post_consultation_summary(self, result: WorkflowResult) -> None:
        """
        Show a summary after consultation is complete.
        
        Confirms to user what was decided: token budget, scope, and next steps.
        """
        from ..utils import console
        
        outputs = result.outputs or {}
        
        # Build summary message
        lines = []
        lines.append("")
        lines.append("━" * 60)
        lines.append("  📋 CONSULTATION SUMMARY")
        lines.append("━" * 60)
        
        # Token budget
        budget = outputs.get("token_budget")
        if budget:
            budget_tier = "Quick" if budget < 200_000 else "Thorough" if budget < 700_000 else "Comprehensive"
            estimated_cost = (budget / 1_000_000) * 5.00  # Empirical ~$5/M tokens (input + output combined)
            lines.append(f"  💰 Token Budget: {budget:,} tokens ({budget_tier}, ~${estimated_cost:.2f})")
        
        # Recommended workflow
        workflow = outputs.get("recommended_workflow", "start_project")
        workflow_names = {
            "start_project": "Full Analysis Pipeline",
            "literature_review": "Literature Review Only",
            "brainstorming": "Brainstorming Session",
            "explore_dataset": "Data Exploration",
        }
        lines.append(f"  🔬 Workflow: {workflow_names.get(workflow, workflow)}")
        
        # Data detected
        data_manifest = outputs.get("data_manifest", {})
        if data_manifest.get("files"):
            file_count = len(data_manifest["files"])
            total_rows = data_manifest.get("summary", {}).get("total_rows", 0)
            lines.append(f"  📊 Data: {file_count} file(s), {total_rows:,} rows")
        
        lines.append("━" * 60)
        lines.append("")
        
        # Print the summary
        for line in lines:
            print(line)
        
        # Log to transcript
        self.transcript.log_system_event(
            "consultation_complete",
            "Consultation phase completed",
            {
                "token_budget": budget,
                "workflow": workflow,
                "data_files": len(data_manifest.get("files", [])) if data_manifest else 0,
            }
        )
    
    def _scan_data_paths(self) -> list[str]:
        """Scan ReadData directory for available data files."""
        data_root = Path("/Users/robertpatton/MiniLab/ReadData")
        paths = []
        
        if data_root.exists():
            for csv_file in data_root.rglob("*.csv"):
                paths.append(str(csv_file))
            for parquet_file in data_root.rglob("*.parquet"):
                paths.append(str(parquet_file))
        
        return paths
    
    async def _handle_workflow_failure(
        self,
        workflow_name: str,
        result: WorkflowResult,
    ) -> bool:
        """
        Handle a workflow failure.
        
        Args:
            workflow_name: Name of failed workflow
            result: Failure result
            
        Returns:
            True to continue, False to stop
        """
        response = self.user_callback(
            f"The {workflow_name} workflow failed: {result.error}\n"
            "Would you like to (r)etry, (s)kip, or (a)bort? [r/s/a]"
        )
        
        response = response.lower().strip()
        
        if response.startswith("r"):
            # Retry by not marking as complete
            return True
        elif response.startswith("s"):
            # Skip and continue
            self.session.completed_workflows.append(f"{workflow_name}_skipped")
            return True
        else:
            return False
    
    async def _create_final_summary(self) -> str:
        """
        Create a final summary of the session.
        
        Returns:
            Summary text
        """
        if not self.session:
            return "No session to summarize."
        
        bohr = self.agents.get("bohr")
        if not bohr:
            return self._simple_summary()
        
        # Gather all results
        results_summary = "\n".join([
            f"- {name}: {result.summary}"
            for name, result in self.session.workflow_results.items()
        ])
        
        summary_result = await bohr.execute_task(
            task=f"""Create a final summary of this analysis session.

Project: {self.session.project_name}
Original Request: {self.session.context.get('user_request', 'N/A')}

Completed Workflows:
{results_summary}

Write a concise summary (2-3 paragraphs) that:
1. Restates what was asked
2. Summarizes what was accomplished
3. Notes any key findings or outputs
4. Suggests next steps if applicable

This summary will be presented to the user.""",
            project_name=self.session.project_name,
        )
        
        summary = summary_result.get("response", self._simple_summary())

        # Authoritative token usage + breakdown from TokenAccount
        usage = self.token_account.usage_summary
        token_usage = {
            "total_used": usage.get("total_used", 0),
            "total_input": usage.get("total_input", 0),
            "total_output": usage.get("total_output", 0),
            "budget": usage.get("budget"),
            "percentage_used": usage.get("percentage_used", 0),
            "remaining": usage.get("remaining"),
            "cache_creation": usage.get("cache_creation", 0),
            "cache_read": usage.get("cache_read", 0),
            "estimated_cost": usage.get("estimated_cost", 0),
            "transaction_count": usage.get("transaction_count", 0),
        }

        breakdown_md = self._format_budget_breakdown()
        if breakdown_md:
            summary = summary.rstrip() + "\n\n" + breakdown_md
        
        # Use ProjectWriter for consistent output (single session_summary.md)
        from ..core import ProjectWriter
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
            token_usage=token_usage,
        )

        # Persist full accounting + aggregated breakdown for calibration/debugging
        try:
            writer.write_token_accounting(
                token_usage=token_usage,
                transactions=self.token_account.iter_transactions(),
                aggregates={
                    "by_workflow": self.token_account.aggregate(keys=("workflow",)),
                    "by_agent": self.token_account.aggregate(keys=("agent_id",)),
                    "by_workflow_agent": self.token_account.aggregate(keys=("workflow", "agent_id")),
                    "by_workflow_trigger": self.token_account.aggregate(keys=("workflow", "trigger")),
                },
            )
        except Exception:
            pass
        
        return summary

    def _format_budget_breakdown(self) -> str:
        """Create a deterministic, truthful budget breakdown section."""
        usage = self.token_account.usage_summary
        if usage.get("total_used", 0) <= 0:
            return ""

        lines: list[str] = []
        lines.append("## Budget Breakdown (Authoritative)")
        lines.append("")
        lines.append(f"- Total: {usage.get('total_used', 0):,} tokens ({usage.get('total_input', 0):,} in + {usage.get('total_output', 0):,} out)")
        if usage.get("budget"):
            lines.append(f"- Budget: {usage['total_used']:,} / {usage['budget']:,} ({usage.get('percentage_used', 0):.1f}% used)")
        if usage.get("cache_read", 0) or usage.get("cache_creation", 0):
            lines.append(f"- Cache: {usage.get('cache_read', 0):,} read, {usage.get('cache_creation', 0):,} created")
        lines.append(f"- Transaction count: {usage.get('transaction_count', 0):,}")

        # Allocation vs actual (BudgetManager)
        try:
            from ..config.budget_manager import get_budget_manager
            bm = get_budget_manager()
            bs = bm.get_summary()
            wf_rows = []
            # Actual by workflow from TokenAccount metadata
            actual_by_wf = {r.get("workflow"): r for r in self.token_account.aggregate(keys=("workflow",))}
            for wf, wb in (bs.get("workflows") or {}).items():
                actual = (actual_by_wf.get(wf) or {}).get("total_tokens", 0)
                alloc = int(wb.get("allocated", 0))
                wf_rows.append((wf, alloc, int(actual)))
            if wf_rows:
                lines.append("")
                lines.append("**Allocated vs Actual (by workflow)**")
                for wf, alloc, actual in sorted(wf_rows, key=lambda x: x[2], reverse=True):
                    delta = actual - alloc
                    lines.append(f"- {wf}: allocated {alloc:,} | actual {actual:,} | delta {delta:+,}")
        except Exception:
            pass

        # Biggest consumers: by workflow+agent
        top = self.token_account.aggregate(keys=("workflow", "agent_id"))[:10]
        if top:
            lines.append("")
            lines.append("**Top Token Consumers (workflow → agent)**")
            for r in top:
                wf = r.get("workflow") or "(unscoped)"
                ag = r.get("agent_id") or "(unknown)"
                lines.append(f"- {wf} → {ag}: {r.get('total_tokens', 0):,} tokens in {r.get('call_count', 0):,} calls")

        # What triggers most LLM spend (after tool/after colleague)
        trig = self.token_account.aggregate(keys=("workflow", "trigger"))[:10]
        if trig:
            lines.append("")
            lines.append("**Top Triggers (what caused LLM calls)**")
            for r in trig:
                wf = r.get("workflow") or "(unscoped)"
                tr = r.get("trigger") or "(none)"
                lines.append(f"- {wf} → {tr}: {r.get('total_tokens', 0):,} tokens")

        return "\n".join(lines)
    
    def _simple_summary(self) -> str:
        """Generate simple summary without LLM."""
        completed = ", ".join(self.session.completed_workflows) or "None"
        return f"Session {self.session.session_id} completed workflows: {completed}"
    
    async def _generate_project_name_interactive(self, request: str) -> str:
        """
        Have Bohr generate a project name via structured JSON response.
        
        Checks for existing projects that might be relevant.
        """
        from ..utils import console
        import json as json_module
        
        # Check for existing projects
        existing_projects = []
        if self.SANDBOX_ROOT.exists():
            existing_projects = [
                d.name for d in self.SANDBOX_ROOT.iterdir()
                if d.is_dir() and (d / "session.json").exists()
            ]
        
        # Have Bohr suggest a project name via LLM with JSON response
        existing_list = "\n".join(f"  - {p}" for p in existing_projects[:10]) if existing_projects else "  (none)"
        
        messages = [
            {"role": "system", "content": """You are Bohr, the MiniLab project orchestrator. 
Generate a concise project name based on the user's request.

RESPOND WITH ONLY A JSON OBJECT in this exact format:
{"project_name": "short_descriptive_name", "is_continuation": false, "continue_project": null}

Rules for project_name:
- Use snake_case (lowercase with underscores)
- Keep it short (2-4 words max)
- Include key subject matter and analysis type
- Add YYYYMMDD date suffix
- Example: "pluvicto_survival_20241211"

If the request mentions continuing an existing project, set:
- is_continuation: true
- continue_project: "exact_project_name_from_list"
"""},
            {"role": "user", "content": f"""User Request: {request}

Existing Projects:
{existing_list}

Today's Date: {datetime.now().strftime("%Y%m%d")}

Generate project name JSON:"""}
        ]
        
        try:
            response = await self.llm_backend.acomplete(messages)
            
            # Parse JSON response
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            data = json_module.loads(response)
            suggested_name = data.get("project_name", "project_" + datetime.now().strftime("%Y%m%d"))
            is_continuation = data.get("is_continuation", False)
            continue_project = data.get("continue_project")
            
            # If Bohr identified this as continuing an existing project
            if is_continuation and continue_project and continue_project in existing_projects:
                console.agent_message("BOHR", f"It looks like you want to continue: \033[1m{continue_project}\033[0m")
                if self.transcript:
                    self.transcript.log_agent_message("bohr", f"It looks like you want to continue: {continue_project}")
                print(f"\n  Press Enter to confirm, or type a different project name:")

                # Clear any buffered stdin (e.g. multi-line paste from initial request)
                try:
                    import sys as _sys
                    import select as _select
                    while True:
                        r, _, _ = _select.select([_sys.stdin], [], [], 0)
                        if not r:
                            break
                        _sys.stdin.readline()
                except Exception:
                    pass

                user_input = input("\n  \033[1;32m▶\033[0m ").strip()
                if self.transcript:
                    self.transcript.log_user_response("Confirm/rename project", user_input)
                
                if user_input:
                    return user_input.replace(" ", "_")
                return continue_project
            
            # New project
            console.agent_message("BOHR", f"I suggest we call this project: \033[1m{suggested_name}\033[0m")
            if self.transcript:
                self.transcript.log_agent_message("bohr", f"I suggest we call this project: {suggested_name}")
            
            # Show related existing projects if any
            related = [p for p in existing_projects if any(
                word in p.lower() for word in suggested_name.lower().replace("_", " ").split()
                if len(word) > 3
            )][:3]
            
            if related:
                print(f"\n  Related existing projects:")
                for p in related:
                    print(f"    • {p}")
            
            print(f"\n  Press Enter to accept, or type a different name:")

            # Clear any buffered stdin (e.g. multi-line paste from initial request)
            try:
                import sys as _sys
                import select as _select
                while True:
                    r, _, _ = _select.select([_sys.stdin], [], [], 0)
                    if not r:
                        break
                    _sys.stdin.readline()
            except Exception:
                pass

            user_input = input("\n  \033[1;32m▶\033[0m ").strip()
            if self.transcript:
                self.transcript.log_user_response("Accept/rename project", user_input)
            
            if user_input:
                project_name = user_input.replace(" ", "_")
            else:
                project_name = suggested_name
            
            # Confirm
            if project_name in existing_projects:
                console.agent_message("BOHR", f"Resuming project: {project_name}")
                if self.transcript:
                    self.transcript.log_agent_message("bohr", f"Resuming project: {project_name}")
            else:
                console.agent_message("BOHR", f"Creating project: {project_name}")
                if self.transcript:
                    self.transcript.log_agent_message("bohr", f"Creating project: {project_name}")
            
            return project_name
            
        except Exception as e:
            # Fallback to simple timestamp-based name
            console.warning(f"Could not generate smart name: {e}")
            fallback = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            console.agent_message("BOHR", f"Using default project name: {fallback}")
            if self.transcript:
                self.transcript.log_agent_message("bohr", f"Using default project name: {fallback} (name generation failed)")
            return fallback
    
    def interrupt(self) -> None:
        """Request graceful interruption of current workflow."""
        self._interrupt_requested = True
        # Propagate interrupt to all agents
        for agent in self.agents.values():
            agent.request_interrupt()
        self._log("Interrupt requested - all agents notified")
    
    def request_exit(self) -> None:
        """Request immediate exit, interrupting all agents."""
        self._exit_requested = True
        self._interrupt_requested = True
        # Propagate interrupt to all agents immediately
        for agent in self.agents.values():
            agent.request_interrupt()
        self._log("Exit requested - all agents interrupted", console_print=False)
    
    def _log(self, message: str, console_print: bool = True) -> None:
        """Log orchestrator message."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [BOHR_ORCHESTRATOR] {message}"
        
        if self.session:
            log_path = self.session.project_path / "logs" / "orchestrator.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(log_entry + "\n")
        
        if console_print:
            print(log_entry)


async def run_minilab(
    request: str,
    project_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Convenience function to run MiniLab.
    
    All sessions flow through consultation - no workflow shortcuts.
    Handles SIGINT (Ctrl+C) gracefully with user options.
    
    Args:
        request: User's analysis request
        project_name: Optional project name
        
    Returns:
        Results dictionary
    """
    import signal
    from ..utils import console, Spinner
    
    orchestrator = BohrOrchestrator()
    
    def handle_interrupt(signum, frame):
        """Handle Ctrl+C with user options."""
        # Pause any running spinner
        was_spinning = Spinner.pause_for_input()
        
        print("\n")
        console.info("⏸️  Interrupted! What would you like to do?")
        print("  1. Pause and provide guidance to the current workflow")
        print("  2. Skip to next workflow phase")
        print("  3. Save progress and exit")
        print("  4. Continue (cancel interrupt)")
        print()
        
        try:
            choice = input("  \033[1;32m▶ Your choice (1-4):\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "3"  # Default to save and exit on double Ctrl+C
        
        if was_spinning:
            Spinner.resume_after_input()
        
        if choice == "1":
            # Get user guidance
            Spinner.pause_for_input()
            try:
                guidance = input("  \033[1;32m▶ Your guidance:\033[0m ").strip()
                if guidance:
                    orchestrator.session.context["user_guidance"] = guidance
                    console.info("Guidance noted. Continuing...")
            except (EOFError, KeyboardInterrupt):
                pass
            Spinner.resume_after_input()
        elif choice == "2":
            console.info("Skipping to next workflow phase...")
            orchestrator.interrupt()
        if choice == "3":
            console.info("Saving progress and exiting...")
            if orchestrator.session:
                orchestrator.session.save()
                console.info(f"Progress saved to: {orchestrator.session.project_path}")
            # Request immediate exit with agent interruption
            orchestrator.request_exit()
            return  # Return from handler, main loop will check flag
        else:
            console.info("Continuing...")
    
    # Install signal handler
    original_handler = signal.signal(signal.SIGINT, handle_interrupt)
    
    try:
        # Start session
        await orchestrator.start_session(
            user_request=request,
            project_name=project_name,
        )
        
        # Run - workflow determined by consultation
        # Check for exit request after each major operation
        if orchestrator._exit_requested:
            return {"status": "interrupted", "message": "User requested exit"}
        
        results = await orchestrator.run()
        
        # Print timing report if enabled
        if os.environ.get("MINILAB_TIMING") == "1":
            from ..utils.timing import timing
            print()
            timing().print_report()
        
        # Print token usage summary from TokenAccount (authoritative source)
        usage = orchestrator.token_account.usage_summary
        if usage.get("total_used", 0) > 0:
            print()
            console.info(f"📊 Token Usage: {usage['total_input']:,} in + {usage['total_output']:,} out = {usage['total_used']:,} total")
            if usage.get("budget"):
                pct = usage.get("percentage_used", 0)
                console.info(f"   Budget: {usage['total_used']:,} / {usage['budget']:,} ({pct:.1f}% used)")
            console.info(f"   Estimated Cost: ${usage.get('estimated_cost', 0):.2f}")
            if usage.get("cache_read", 0) > 0:
                console.info(f"   Cache: {usage['cache_read']:,} read, {usage.get('cache_creation', 0):,} created")
        
        return results
    except KeyboardInterrupt:
        # Handle any uncaught keyboard interrupts gracefully
        console.warning("Session interrupted")
        if orchestrator.session:
            orchestrator.session.save()
        return {"status": "interrupted", "message": "Session interrupted by user"}
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)
