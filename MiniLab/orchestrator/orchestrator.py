"""
MiniLab Session Orchestrator.

IMPORTANT DISTINCTION:
- MiniLabOrchestrator: Infrastructure class (Python code) that manages sessions,
  executes modules, tracks budgets, and coordinates the agent team.
  This is NOT an AI agent - it's the mechanical backbone.
  
- Bohr (agent): The AI "project manager" persona in agents.yaml who makes
  scientific decisions, plans analyses, and communicates with users.
  Bohr is an LLM-powered agent like hinton, bayes, etc.

The orchestrator USES the Bohr agent for planning and summaries,
but the orchestrator itself is deterministic Python infrastructure.

Key Responsibilities:
- Session lifecycle management (start, resume, save)
- TaskGraph-based execution flow
- Budget tracking and enforcement
- Project state persistence
- User interaction for confirmations
- Module coordination (Task → Module → Tool hierarchy)

Terminology (aligned with minilab_outline.md):
- Task: A project-DAG node representing a user-meaningful milestone
- Module: A reusable procedure that composes tools and possibly agents
- Tool: An atomic, side-effectful capability with typed I/O
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
from ..config.budget_history import get_budget_history

# Import from new modules package (with backward compat aliases)
from ..modules import (
    Module,
    ModuleResult,
    ModuleStatus,
    ConsultationModule,
    LiteratureReviewModule,
    TeamDiscussionModule,  # Was PlanningCommitteeModule
    AnalysisExecutionModule,  # Was ExecuteAnalysisModule
    BuildReportModule,  # Was WriteupResultsModule
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
)
# Keep backward compat imports
from ..modules import (
    WorkflowModule,
    WorkflowResult,
    WorkflowStatus,
)

from ..core.task_graph import TaskGraph, TaskStatus as GraphTaskStatus, TaskNode
from ..llm_backends import AnthropicBackend
from ..utils import Style, console  # For continuation prompt styling + user-facing output


# Agent roster with their specializations (for TaskGraph execution)
# NOTE: These are AI agent PERSONAS, not the orchestrator
AGENT_ROSTER = {
    "bohr": "Project Manager - high-level scientific planning, user communication, synthesis",
    "gould": "Science Writer - literature review, documentation, clear explanations",
    "farber": "Clinical Expert - experimental design, medical interpretation, protocols",
    "feynman": "Theoretician - physics, mathematics, theoretical analysis, first principles",
    "shannon": "Information Theorist - statistics, signal processing, information theory",
    "greider": "Molecular Biologist - genetics, cellular mechanisms, biological interpretation",
    "dayhoff": "Bioinformatician - sequence analysis, databases, computational biology",
    "hinton": "ML Expert - machine learning, neural networks, data analysis, modeling",
    "bayes": "Statistician - Bayesian inference, probability, uncertainty quantification",
}


@dataclass
class MiniLabSession:
    """
    State container for a MiniLab session.
    
    Tracks the current project, completed modules, and artifacts.
    (Updated terminology: workflows → modules)
    """
    session_id: str
    project_name: str
    project_path: Path
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    current_module: Optional[str] = None
    completed_modules: list[str] = field(default_factory=list)
    module_results: dict[str, ModuleResult] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    
    # Backward compat aliases (property-based)
    @property
    def current_workflow(self) -> Optional[str]:
        return self.current_module
    
    @current_workflow.setter
    def current_workflow(self, val: Optional[str]) -> None:
        self.current_module = val
    
    @property
    def completed_workflows(self) -> list[str]:
        return self.completed_modules
    
    @property
    def workflow_results(self) -> dict[str, ModuleResult]:
        return self.module_results
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "project_name": self.project_name,
            "project_path": str(self.project_path),
            "started_at": self.started_at,
            "current_module": self.current_module,
            "completed_modules": self.completed_modules,
            "module_results": {
                k: {
                    "status": v.status.value,
                    "summary": v.summary,
                    "artifacts": v.artifacts,
                }
                for k, v in self.module_results.items()
            },
            # Backward compat aliases in serialized form
            "current_workflow": self.current_module,
            "completed_workflows": self.completed_modules,
        }
    
    def save(self) -> Path:
        """Save session state to disk."""
        session_path = self.project_path / "session.json"
        with open(session_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return session_path
    
    @classmethod
    def load(cls, session_path: Path) -> "MiniLabSession":
        """Load session from disk (supports old and new terminology)."""
        with open(session_path) as f:
            data = json.load(f)
        return cls(
            session_id=data["session_id"],
            project_name=data["project_name"],
            project_path=Path(data["project_path"]),
            started_at=data.get("started_at", ""),
            # Support both old and new keys
            current_module=data.get("current_module") or data.get("current_workflow"),
            completed_modules=data.get("completed_modules") or data.get("completed_workflows", []),
        )


class MiniLabOrchestrator:
    """
    Primary infrastructure orchestrator for MiniLab analysis sessions.
    
    THIS IS NOT AN AI AGENT. This is Python infrastructure code that:
    - Manages session lifecycle (start, pause, resume, save)
    - Coordinates workflow execution order
    - Tracks and enforces token budgets
    - Routes tasks to appropriate agent personas
    - Handles user interaction and confirmations
    
    The Bohr AGENT (defined in agents.yaml) is a separate LLM-powered
    persona who acts as project manager and makes scientific decisions.
    This orchestrator class USES the Bohr agent but is not itself Bohr.
    
    Think of it like:
    - MiniLabOrchestrator = the stage manager (infrastructure)
    - Bohr agent = the director (AI decision-maker)
    - Other agents = the actors (AI specialists)
    """
    
    # Sandbox root for all projects (configurable via environment)
    SANDBOX_ROOT = Path(os.getenv(
        "MINILAB_SANDBOX",
        str(Path(__file__).parent.parent.parent / "Sandbox")
    ))
    
    # Module class mapping - maps task types to module executors
    # (Updated terminology: workflows → modules)
    MODULE_CLASSES = {
        "consultation": ConsultationModule,
        "literature_review": LiteratureReviewModule,
        "team_discussion": TeamDiscussionModule,
        "planning_committee": TeamDiscussionModule,  # Alias for backward compat
        "planning": PlanningModule,
        "analysis_execution": AnalysisExecutionModule,
        "execute_analysis": AnalysisExecutionModule,  # Alias for backward compat
        "build_report": BuildReportModule,
        "writeup_results": BuildReportModule,  # Alias for backward compat
        "critical_review": CriticalReviewModule,
        # New modules
        "one_on_one": OneOnOneModule,
        "core_input": CoreInputModule,
        "evidence_gathering": EvidenceGatheringModule,
        "write_artifact": WriteArtifactModule,
        "generate_code": GenerateCodeModule,
        "run_checks": RunChecksModule,
        "sanity_check_data": SanityCheckDataModule,
        "interpret_stats": InterpretStatsModule,
        "interpret_plot": InterpretPlotModule,
        "citation_check": CitationCheckModule,
        "formatting_check": FormattingCheckModule,
        "consult_external_expert": ConsultExternalExpertModule,
    }
    
    # Backward compat alias
    WORKFLOW_MODULES = MODULE_CLASSES
    
    # Task-to-module mapping (tasks can use different modules)
    TASK_MODULE_MAP = {
        "literature_review": "literature_review",
        "lit_review": "literature_review",
        "background": "literature_review",
        "planning": "planning",
        "analysis_plan": "team_discussion",
        "deliberation": "team_discussion",
        "analysis": "analysis_execution",
        "execute": "analysis_execution",
        "data_exploration": "analysis_execution",
        "modeling": "analysis_execution",
        "statistical_analysis": "analysis_execution",
        "writeup": "build_report",
        "documentation": "build_report",
        "report": "build_report",
        "figures": "build_report",
        "review": "critical_review",
        "validation": "critical_review",
        "quality_check": "critical_review",
        # New task types
        "expert_consultation": "one_on_one",
        "core_discussion": "core_input",
        "evidence": "evidence_gathering",
        "artifact": "write_artifact",
        "code": "generate_code",
        "test": "run_checks",
        "data_check": "sanity_check_data",
        "stats_interpretation": "interpret_stats",
        "plot_interpretation": "interpret_plot",
        "citation_audit": "citation_check",
        "format_check": "formatting_check",
        "external_expert": "consult_external_expert",
    }
    
    # Backward compat alias
    TASK_WORKFLOW_MAP = TASK_MODULE_MAP
    
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
        
        # Set up transcript logger (v2: simplified to MD-only)
        # Transcript is initialized lazily in start_session once we have a project path
        # All run artifacts belong inside the project folder, not a global _runs directory
        self.transcripts_dir = transcripts_dir or Path(__file__).parent.parent.parent / "Transcripts"
        self.transcript: Optional[TranscriptWriter] = None  # Lazy init in start_session
        
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
        self._is_resumed: bool = False  # Set to True when resuming a session
        
        # Budget state flags - set by _on_budget_warning
        self._budget_percentage: float = 0.0
        self._budget_tight: bool = False  # Set at 80%
        self._budget_critical: bool = False  # Set at 95%

    def _permission_callback(self, request: str) -> bool:
        """Permission callback for tools (terminal/env). Defaults to interactive approval."""
        from ..utils import Spinner
        
        # Allow non-interactive runs to auto-approve if explicitly configured.
        auto = os.getenv("MINILAB_AUTO_APPROVE", "").strip().lower()
        if auto in {"1", "true", "yes"}:
            return True
        if auto in {"0", "false", "no"}:
            return False

        was_spinning = Spinner.pause_for_input()
        
        try:
            print()
            console.agent_message("MINILAB", "Permission required for an operation:")
            print(f"  {request}")
            print()
            print("  Would you like to approve this operation?")
            print()
            response = input("  \033[1;32m▶ Your response:\033[0m ").strip().lower()
            
            if was_spinning:
                Spinner.resume_after_input()
            
            # Interpret natural language approval
            approval_words = {"yes", "y", "approve", "ok", "okay", "sure", "go", "proceed", "allow", "do it", "fine", "1"}
            denial_words = {"no", "n", "deny", "reject", "stop", "cancel", "don't", "dont", "nope", "2"}
            
            if any(word in response for word in approval_words):
                return True
            elif any(word in response for word in denial_words):
                return False
            else:
                # Default to approval if response is ambiguous but not explicitly negative
                return len(response) > 0 and not any(word in response for word in denial_words)
        except (KeyboardInterrupt, EOFError):
            # User cancelled input - return default (no approval)
            if was_spinning:
                Spinner.resume_after_input()
            return False
        except Exception as e:
            # Unexpected error during input - log and return default
            self._log(f"Error during input confirmation: {e}", console_print=False)
            if was_spinning:
                Spinner.resume_after_input()
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
        """
        Callback when token budget warnings are triggered.
        
        This method does three things:
        1. Logs visual feedback to the console
        2. Records to transcript for post-session analysis
        3. Sets budget state flags that workflows and agents can check
        """
        from ..utils import console
        
        # Update budget state (can be checked by workflows)
        self._budget_percentage = percentage
        
        if percentage >= 95:
            console.warning(f"⚠ Token budget critical: {percentage:.0f}% used ({used:,}/{budget:,})")
            console.info("Budget nearly exhausted. Wrapping up current work with available progress.")
            self._budget_critical = True
        elif percentage >= 80:
            console.warning(f"Token budget at {percentage:.0f}% ({used:,}/{budget:,})")
            console.info("Budget constrained. Prioritizing core deliverables.")
            self._budget_tight = True
        elif percentage >= 60:
            console.info(f"Budget update: {percentage:.0f}% used ({used:,}/{budget:,})")
        
        # Log to transcript
        self.transcript.log_budget_warning(percentage, f"Budget at {percentage:.0f}% ({used:,}/{budget:,})")
    
    def _default_user_input(self, prompt: str) -> str:
        """Default user input via stdin."""
        from ..utils import console, Style
        print()  # Clean line
        console.info(prompt)
        print()
        return input(f"  {Style.BOLD}{Style.GREEN}▶ Your response:{Style.RESET} ").strip()
    
    def _user_input_callback(self, prompt: str, options: Optional[list[str]] = None) -> str:
        """Callback for tools that need user input."""
        from ..utils import Style
        print()  # Clean line
        if options:
            # Present options as natural language choices
            print(f"\n{prompt}")
            options_text = ", ".join(options[:-1]) + f", or {options[-1]}" if len(options) > 1 else options[0]
            print(f"  Available options: {options_text}")
            raw_response = input(f"  {Style.BOLD}{Style.GREEN}▶ Your response:{Style.RESET} ").strip()
            
            # Try to match user's natural language response to an option
            response_lower = raw_response.lower()
            for opt in options:
                opt_lower = opt.lower()
                # Check for exact match, partial match, or first word match
                if (opt_lower == response_lower or 
                    opt_lower in response_lower or 
                    response_lower in opt_lower or
                    opt_lower.split()[0] == response_lower.split()[0] if response_lower.split() and opt_lower.split() else False):
                    return opt
            # Return raw response if no match found
            return raw_response
        print(f"\n{prompt}")
        print()
        return input(f"  {Style.BOLD}{Style.GREEN}▶ Your response:{Style.RESET} ").strip()
    
    async def bohr_understanding_phase(
        self,
        user_request: str,
        preset_project_name: Optional[str] = None,
    ) -> tuple[str, bool]:
        """
        Phase 1: Bohr's understanding phase (BEFORE session starts).
        
        LOOP until user confirms:
        1. Bohr summarizes understanding, suggests project name, asks questions
        2. User responds in plain English
        3. Bohr interprets and either proceeds or asks follow-up
        
        Args:
            user_request: The user's initial request
            preset_project_name: Optional pre-set name (for testing/automation)
            
        Returns:
            Tuple of (confirmed_project_name, user_confirmed_bool)
        """
        from ..utils import console, Spinner
        from ..llm_backends import AnthropicBackend
        import sys
        import select
        
        # If project name is preset (e.g., for testing), skip this phase
        if preset_project_name:
            return preset_project_name, True
        
        # Create a temporary backend for this phase
        backend = AnthropicBackend(model="claude-sonnet-4-5", agent_id="bohr")
        
        # Conversation history for multi-turn
        conversation_history = []
        suggested_name = None  # Will be set from Bohr's structured response
        max_turns = 5  # Prevent infinite loops
        
        for turn in range(max_turns):
            # Build prompt based on turn
            if turn == 0:
                # First turn: Initial understanding with STRUCTURED OUTPUT
                prompt = f"""You are Bohr, the project manager for MiniLab, a multi-agent scientific research system.

The user has just submitted a new analysis request. Your task is to:
1. Demonstrate you understand their request
2. Suggest a project name
3. Ask any clarifying questions (if truly needed)

## User's Request
{user_request}

## CRITICAL: Response Format
You MUST respond with a JSON object followed by your conversational message. The JSON contains the structured data, the message is what gets displayed to the user.

```json
{{
  "project_name": "<snake_case_name_3_to_5_words>",
  "understanding_summary": "<one sentence summary of what user wants>",
  "has_questions": <true if you have clarifying questions, false if ready to proceed>
}}
```

Then, after the JSON block, write your conversational response to the user. Include:
- A brief summary of your understanding (3-5 bullet points)
- The project name you're suggesting (reference the name from your JSON)
- Any clarifying questions if needed, OR confirmation that you're ready to proceed once they approve

## Guidelines
- The JSON MUST come first, in a code block
- The project_name MUST be snake_case (lowercase, underscores, no spaces)
- Be conversational and warm in your message
- Keep the message concise (under 300 words)"""
                conversation_history.append({"role": "user", "content": prompt})
            
            # Get Bohr's response
            spinner = Spinner("Bohr is reviewing your request" if turn == 0 else "Bohr is processing")
            spinner.start()
            
            try:
                bohr_response = await backend.acomplete(
                    conversation_history,
                    max_tokens=1000,
                    temperature=0.4,
                )
            except Exception as e:
                spinner.stop_error(f"Error: {e}")
                return "", False
            
            spinner.stop()
            conversation_history.append({"role": "assistant", "content": bohr_response})
            
            # Extract the STRUCTURED DATA from Bohr's response
            from ..utils import extract_json_from_text
            structured_data = extract_json_from_text(bohr_response, fallback=None)
            
            if structured_data and structured_data.get("project_name"):
                suggested_name = structured_data["project_name"]
            elif suggested_name is None:
                # Fallback only if we truly have nothing
                suggested_name = "minilab_analysis"
            
            # Extract the conversational message (everything after the JSON block)
            # Find the end of the JSON code block and get the rest
            display_message = bohr_response
            if "```json" in bohr_response and "```" in bohr_response:
                # Find the closing ``` after the json block
                json_start = bohr_response.find("```json")
                json_end = bohr_response.find("```", json_start + 7)
                if json_end != -1:
                    # Get everything after the JSON block
                    display_message = bohr_response[json_end + 3:].strip()
                    if not display_message:
                        # If nothing after JSON, use a default message
                        display_message = f"I suggest we name this project **{suggested_name}**. Does this understanding work for you?"
            
            # Display Bohr's message (not the raw JSON)
            print()
            console.agent_message("BOHR", display_message)
            print()
            
            # Flush any buffered stdin before reading
            try:
                while True:
                    r, _, _ = select.select([sys.stdin], [], [], 0)
                    if not r:
                        break
                    sys.stdin.readline()  # Discard buffered input
            except Exception:
                pass
            
            # Get user's response
            try:
                user_response = input("  \033[1;32m▶ Your response:\033[0m ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n  Goodbye!")
                return "", False
            
            # Empty response = cancel
            if not user_response:
                console.info("Session cancelled.")
                return "", False
            
            conversation_history.append({"role": "user", "content": user_response})
            
            # Interpret whether user confirmed or wants more discussion
            # SIMPLE: Just determine if they approved. We already have the project name.
            interpretation_prompt = f"""Analyze this user response to determine their intent.

User's response: "{user_response}"

Did they:
1. CONFIRM/APPROVE (yes, looks good, proceed, let's go, go ahead, that works, sure, ok, etc.)?
2. REQUEST CHANGES or have questions?

Respond with ONLY a JSON object:
```json
{{
  "confirmed": <true if user approved/confirmed, false otherwise>,
  "follow_up": "<if not confirmed, what follow-up is needed? null if confirmed>"
}}
```"""

            spinner = Spinner("Processing")
            spinner.start()
            
            try:
                interpretation = await backend.acomplete(
                    [{"role": "user", "content": interpretation_prompt}],
                    max_tokens=200,
                    temperature=0.1,
                )
            except Exception as e:
                spinner.stop_error(f"Error: {e}")
                return suggested_name, False
            
            spinner.stop()
            
            # Parse interpretation
            result = extract_json_from_text(interpretation, fallback={
                "confirmed": False,
                "follow_up": None,
            })
            
            confirmed = result.get("confirmed", False)
            follow_up = result.get("follow_up")
            
            if confirmed:
                # User confirmed - proceed with THE ALREADY-SET project name
                # DO NOT get project_name from LLM response - use suggested_name directly
                print()
                console.agent_message("BOHR", f"Excellent! I'll proceed with project **{suggested_name}**. Let me consult the team to develop a detailed plan.")
                print()
                return suggested_name, True
            
            elif follow_up:
                # Need more discussion - add follow-up to conversation
                conversation_history.append({"role": "assistant", "content": follow_up})
                print()
                console.agent_message("BOHR", follow_up)
                print()
            
            # Continue loop for next turn
        
        # Max turns reached - default to proceeding with best guess
        console.warning("Max conversation turns reached. Proceeding with current understanding.")
        return suggested_name, True

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
        
        # Reset budget state flags
        self._budget_percentage = 0.0
        self._budget_tight = False
        self._budget_critical = False

        # Reset BudgetManager singleton for new session
        try:
            from ..config.budget_manager import BudgetManager
            BudgetManager.reset()
        except ImportError:
            # BudgetManager may not be available, continue without reset
            pass
        except Exception as e:
            # Log error but continue - budget manager reset is not critical
            self._log(f"Warning: Could not reset BudgetManager: {e}", console_print=False)
        
        # CRITICAL: Project name must be pre-approved via bohr_understanding_phase
        # Do NOT call any LLM here - just validate the approved name
        
        # CRITICAL: Validate that project_name is locked and approved
        # If it differs from what user approved, that's a critical breach
        # This should never happen if CLI is working correctly
        if not project_name or len(project_name.strip()) == 0:
            raise ValueError("Project name cannot be empty. Project name must be pre-approved by user.")
        
        # Create project directory
        session_id = self._session_date.strftime("%Y%m%d_%H%M%S")
        project_path = self.SANDBOX_ROOT / project_name
        project_path.mkdir(parents=True, exist_ok=True)

        # Initialize per-run transcript inside the project folder.
        # Location: Sandbox/<project>/runs/<session_id>/
        runs_dir = project_path / "runs" / session_id
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir = runs_dir
        self.transcript = TranscriptLogger(self.transcripts_dir)
        self.transcript.start_session(project_name)
        self.transcript.log_user_message(user_request)
        
        # Set up BudgetHistory with WORKSPACE ROOT for GLOBAL token_usage_learnings.md
        # This is the parent of Sandbox - shared across ALL projects
        workspace_root = self.SANDBOX_ROOT.parent
        budget_history = get_budget_history()
        budget_history.set_workspace_root(workspace_root)
        # Generate initial document if it doesn't exist
        budget_history.save()  # This triggers _update_living_document()
        
        # Initialize session
        self.session = MiniLabSession(
            session_id=session_id,
            project_name=project_name,
            project_path=project_path,
        )
        self.session.context["user_request"] = user_request
        
        # Transcript started above (project-scoped runs/<session_id>/)
        
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
        # Also inject token learnings context
        learnings_context = budget_history.get_learnings_context()
        for agent in self.agents.values():
            agent.set_transcript(self.transcript)
            if self.streaming_enabled:
                agent.enable_streaming(on_chunk=self.on_stream_chunk)
            # Inject token learnings context for budget-aware planning
            if hasattr(agent, 'llm') and hasattr(agent.llm, 'append_project_context'):
                agent.llm.append_project_context(
                    f"\n\n## TOKEN USAGE LEARNINGS (from resource_learning.md)\n\n"
                    f"Use this historical data to plan task budgets:\n\n{learnings_context}"
                )
        
        # Save initial session state (log internally, no console spam)
        self._log(f"Session started: {session_id}", console_print=False)
        self._log(f"Project: {project_name}", console_print=False)
        self._log(f"Request: {user_request}", console_print=False)
        self.session.save()
        
        return self.session
    
    async def resume_session(self, project_path: Path) -> MiniLabSession:
        """
        Resume an existing session with intelligent context loading.
        
        Key improvements over naive resume:
        1. Loads and validates existing TaskGraph
        2. Injects completed task context into agents
        3. Loads continuation plan if budget was exhausted
        4. Uses BudgetHistory to estimate remaining work
        5. Sets is_resumed flag to skip consultation in run()
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Restored MiniLabSession with context
        """
        from ..utils import console
        
        session_path = project_path / "session.json"
        if not session_path.exists():
            raise ValueError(f"No session found at {project_path}")
        
        # Mark this as a resumed session - consultation will be skipped
        self._is_resumed = True
        
        # Set up BudgetHistory with WORKSPACE ROOT for GLOBAL token_usage_learnings.md
        workspace_root = self.SANDBOX_ROOT.parent
        budget_history = get_budget_history()
        budget_history.set_workspace_root(workspace_root)
        
        self.session = MiniLabSession.load(session_path)

        # Treat each resume as a new run with its own transcript folder.
        self._session_date = datetime.now()
        self.session.session_id = self._session_date.strftime("%Y%m%d_%H%M%S")
        self.session.started_at = self._session_date.isoformat()

        runs_dir = project_path / "runs" / self.session.session_id
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir = runs_dir
        self.transcript = TranscriptLogger(self.transcripts_dir)
        self.transcript.start_session(self.session.project_name)
        self.transcript.log_system_event(
            "session_resumed",
            "Resumed existing project in a new run",
            {"project_path": str(project_path)},
        )
        
        # Reinitialize agents with context manager and tool factory
        
        session_date = self._session_date
        
        self.agents, self.context_manager, self.tool_factory = create_agents(
            workspace_root=workspace_root,
            input_callback=self._user_input_callback,
            permission_callback=self._permission_callback,
            session_date=session_date,
        )
        
        # Set transcript logger on resumed agents and enable streaming if configured
        # Also inject token learnings context
        learnings_context = budget_history.get_learnings_context()
        for agent in self.agents.values():
            agent.set_transcript(self.transcript)
            if self.streaming_enabled:
                agent.enable_streaming(on_chunk=self.on_stream_chunk)
            # Inject token learnings context for budget-aware planning
            if hasattr(agent, 'llm') and hasattr(agent.llm, 'append_project_context'):
                agent.llm.append_project_context(
                    f"\n\n## TOKEN USAGE LEARNINGS (from resource_learning.md)\n\n"
                    f"Use this historical data to plan task budgets:\n\n{learnings_context}"
                )
        
        # Load TaskGraph and inject prior context
        task_graph_path = project_path / "task_graph.json"
        if task_graph_path.exists():
            try:
                task_graph = TaskGraph.load(task_graph_path)
                # Enable auto-save on resumed task graph
                task_graph = self._ensure_task_graph_autosave(task_graph)
                self.session.context["task_graph"] = task_graph.to_dict()

                # Validate + export TaskGraph visuals for user visibility
                try:
                    from ..core.task_graph import validate_task_graph
                    validate_task_graph(task_graph)
                except ValueError as e:
                    # TaskGraph structure error - degraded mode (continue with warning)
                    self._log(f"TaskGraph validation warning: {e}", console_print=False)
                except Exception as e:
                    # Unexpected validation error
                    self._log(f"TaskGraph validation error: {type(e).__name__}: {e}", console_print=False)

                try:
                    from ..core import ProjectWriter
                    ProjectWriter(
                        project_path=self.session.project_path,
                        project_name=self.session.project_name,
                        context_manager=self.context_manager,
                    ).write_task_graph_visuals(task_graph=task_graph, render_png=True)
                except Exception as e:
                    # TaskGraph PNG rendering is mandatory; fail loudly with actionable detail.
                    raise RuntimeError(f"Failed to render TaskGraph visuals: {e}") from e
                
                # Build context from completed tasks for agent injection
                completed_context = self._build_resume_context(task_graph, project_path)
                if completed_context:
                    # Inject into all agent LLM backends as project context
                    # Agents use self.llm (not self.backend) for the LLM backend
                    injected_count = 0
                    for agent in self.agents.values():
                        if hasattr(agent, 'llm') and hasattr(agent.llm, 'append_project_context'):
                            agent.llm.append_project_context(completed_context)
                            injected_count += 1
                    
                    self._log(f"Injected prior progress context into {injected_count} agents", console_print=False)
                
                # Report progress to user
                progress = task_graph.get_progress()
                console.info(f"Resuming: {progress['completed']}/{progress['total_tasks']} tasks completed")
                
            except Exception as e:
                self._log(f"Could not load TaskGraph: {e}", console_print=False)
        
        # Load continuation plan if exists
        continuation_path = project_path / "continuation_plan.json"
        if continuation_path.exists():
            try:
                with open(continuation_path) as f:
                    continuation_plan = json.load(f)
                
                recommended_budget = continuation_plan.get("recommended_continuation_budget", 0)
                console.info(f"Prior run suggested {recommended_budget:,} tokens to complete")
                
                self.session.context["continuation_plan"] = continuation_plan
            except (json.JSONDecodeError, IOError) as e:
                # Continuation plan file corrupted or inaccessible - warn and continue
                self._log(f"Warning: Could not load continuation plan: {e}", console_print=False)
            except Exception as e:
                # Unexpected error loading continuation plan
                self._log(f"Error loading continuation plan: {type(e).__name__}: {e}", console_print=False)
        
        self._log(f"Session resumed: {self.session.session_id}")

        # Persist updated run metadata
        try:
            self.session.save()
        except IOError as e:
            # Could not save session - warn but continue (session state in memory)
            self._log(f"Warning: Could not persist session state: {e}", console_print=False)
        except Exception as e:
            # Unexpected error persisting session
            self._log(f"Error saving session: {type(e).__name__}: {e}", console_print=False)
        
        return self.session
    
    def _build_resume_context(self, task_graph: TaskGraph, project_path: Path) -> str:
        """
        Build context string summarizing completed work for agent injection.
        
        This is CRITICAL for avoiding redundant work on resume. Provides:
        1. Completed tasks with their outputs
        2. Existing files and their purposes (from checkpoints)
        3. Continuation priorities
        4. Clear "ALREADY DONE - DO NOT REPEAT" markers
        
        This allows agents to understand what was accomplished in prior sessions
        and BUILD ON existing work rather than recreating it.
        """
        lines = ["## Prior Session Progress"]
        lines.append("")
        lines.append("**CRITICAL: Read this carefully to avoid redundant work.**")
        lines.append("")
        
        completed_tasks = [t for t in task_graph.tasks.values() if t.status == GraphTaskStatus.COMPLETED]
        pending_tasks = [t for t in task_graph.tasks.values() if t.status != GraphTaskStatus.COMPLETED]
        
        if completed_tasks:
            lines.append(f"### ✓ COMPLETED Tasks ({len(completed_tasks)}) - DO NOT REPEAT")
            for task in completed_tasks:
                lines.append(f"- **{task.name}** (by {task.owner})")
                if task.description:
                    lines.append(f"  Objective: {task.description}")
                if task.actual_tokens:
                    lines.append(f"  Tokens used: {task.actual_tokens:,}")
                if task.outputs:
                    lines.append(f"  Outputs: {', '.join(str(v) for v in task.outputs.values())[:200]}")
            lines.append("")
        
        # Read checkpoint files to understand exactly what was done
        checkpoints_dir = project_path / "checkpoints"
        if checkpoints_dir.exists():
            checkpoint_files = list(checkpoints_dir.glob("*.md"))
            if checkpoint_files:
                lines.append("### 📋 Checkpoint Records (AUTHORITATIVE)")
                lines.append("These files describe what was completed. Read them to avoid duplication.")
                lines.append("")
                
                for cp_file in checkpoint_files[:3]:  # Limit to 3 most recent
                    try:
                        content = cp_file.read_text()
                        # Extract key sections from checkpoint
                        lines.append(f"#### From {cp_file.name}:")
                        # Take first 500 chars or until "---" separator after header
                        preview = content[:1000]
                        if "---" in preview[10:]:
                            preview = preview[:preview.index("---", 10)]
                        lines.append(preview.strip())
                        lines.append("")
                    except (IOError, UnicodeDecodeError) as e:
                        # Checkpoint file unreadable - record failure but continue
                        lines.append(f"- {cp_file.name} (could not read: {type(e).__name__})")
                lines.append("")
        
        # List existing outputs to prevent recreation
        lines.append("### 📁 Existing Project Files")
        lines.append("**DO NOT recreate these files - build on them or reference them.**")
        lines.append("")
        
        # Key directories to scan
        key_dirs = [
            ("analysis", "Analysis scripts"),
            ("figures", "Generated figures"),
            ("data", "Processed data"),
        ]
        
        for subdir, description in key_dirs:
            dir_path = project_path / subdir
            if dir_path.exists():
                files = list(dir_path.iterdir())
                if files:
                    lines.append(f"**{description}** ({subdir}/):")
                    for f in files[:10]:
                        lines.append(f"  - {f.name}")
                    if len(files) > 10:
                        lines.append(f"  - ... and {len(files) - 10} more files")
                    lines.append("")
        
        # Root-level markdown/data files
        root_files = list(project_path.glob("*.md")) + list(project_path.glob("*.json"))
        important_root_files = [f for f in root_files if f.name not in ("session.json", "task_graph.json", "continuation_plan.json")]
        if important_root_files:
            lines.append("**Project documentation:**")
            for f in important_root_files:
                lines.append(f"  - {f.name}")
            lines.append("")
        
        if pending_tasks:
            lines.append(f"### ⏳ Remaining Tasks ({len(pending_tasks)})")
            lines.append("These are the tasks that still need work:")
            for task in pending_tasks:
                lines.append(f"- **{task.name}** (assigned to {task.owner})")
                if task.description:
                    lines.append(f"  To do: {task.description}")
            lines.append("")
        
        # Check for continuation plan from prior session
        continuation_path = project_path / "continuation_plan.json"
        if continuation_path.exists():
            try:
                with open(continuation_path) as f:
                    plan = json.load(f)
                lines.append("### 🎯 Continuation Plan from Prior Session")
                if plan.get("progress_summary"):
                    lines.append(f"**Progress so far:** {plan['progress_summary']}")
                if plan.get("remaining_work"):
                    lines.append(f"**Still to do:** {plan['remaining_work']}")
                if plan.get("priority_next_steps"):
                    lines.append("**Priority next steps:**")
                    for p in plan["priority_next_steps"][:3]:
                        lines.append(f"  1. {p}")
                lines.append("")
            except Exception:
                pass
        
        lines.append("---")
        lines.append("**Remember: Your job is to CONTINUE work, not restart it. Check existing files before creating new ones.**")
        
        return "\n".join(lines) if len(lines) > 5 else ""
    
    async def run(self) -> dict[str, Any]:
        """
        Run the orchestration loop with AGENT-DRIVEN workflow planning.
        
        Flow for new sessions:
        1. Run consultation (team develops TaskGraph)
        2. Run planning committee (all agents deliberate)
        3. Present plan to user with token estimates
        4. Get user approval (may adjust)
        5. Execute the approved plan
        
        For resumed sessions: loads existing task_graph and continues.
        
        Returns:
            Final results dictionary
        """
        from ..utils import console, Spinner, Style, StatusIcon
        
        if not self.session:
            raise RuntimeError("No active session. Call start_session first.")
        
        try:
            results = {}
            task_graph = None

            # Check if this is a resumed session
            if self._is_resumed:
                # Skip consultation - load existing task_graph from session context
                console.info("Resuming from previous session...")
                
                # Restore task_graph from context (set during resume_session)
                if "task_graph" in self.session.context:
                    task_graph = TaskGraph.from_dict(self.session.context["task_graph"])
                    progress = task_graph.get_progress()
                    console.info(f"Task graph loaded: {progress['completed']}/{progress['total_tasks']} tasks completed")
                    
                    # Restore token budget if it was set
                    if self.session.context.get("token_budget"):
                        self._token_budget = self.session.context["token_budget"]
                        self.token_account.set_budget(self._token_budget)
                        self.transcript.set_token_budget(self._token_budget)
                    
                    # Initialize budget manager from prior session
                    try:
                        from ..config.budget_manager import get_budget_manager
                        complexity = self.session.context.get("complexity", "moderate")
                        budget_mgr = get_budget_manager()
                        if self._token_budget:
                            budget_mgr.initialize_session(total_budget=self._token_budget, complexity=str(complexity))
                    except ImportError:
                        pass
                    except Exception as e:
                        self._log(f"Warning: Could not initialize budget manager: {e}", console_print=False)
                else:
                    # No task_graph in context - need to load from file
                    task_graph_path = self.session.project_path / "task_graph.json"
                    if task_graph_path.exists():
                        task_graph = TaskGraph.load(task_graph_path)
                        self.session.context["task_graph"] = task_graph.to_dict()
                    else:
                        console.warning("No task graph found - running consultation")
                        self._is_resumed = False  # Fall through to consultation
            
            # Run consultation if not resuming or if resume failed to find task_graph
            if not self._is_resumed:
                # Always start with consultation to understand user needs
                # Consultation is the ONE mandatory phase - everything else is dynamic
                console.info("Starting consultation to understand your requirements...")
                
                # Step 1: Execute consultation (mandatory)
                spinner = Spinner("Running Consultation")
                spinner.start()
                self.transcript.log_stage_transition("consultation", "Starting Consultation Phase")
                self.session.current_workflow = "consultation"
                self.session.save()

                from ..core.token_context import token_context

                start_tokens = self.token_account.total_used
                with token_context(workflow="phase2.consultation"):
                    consultation_result = await self._execute_workflow("consultation")
                used_tokens = self.token_account.total_used - start_tokens

                # Record truthful workflow usage into BudgetManager (if initialized)
                try:
                    from ..config.budget_manager import get_budget_manager
                    status = "completed" if consultation_result.status == WorkflowStatus.COMPLETED else "failed"
                    stop_reason = None
                    if consultation_result.status == WorkflowStatus.PAUSED:
                        status = "user_stopped"
                        stop_reason = consultation_result.metadata.get("stop_reason") if consultation_result.metadata else None
                    get_budget_manager().record_usage(
                        "phase2.consultation",
                        max(0, int(used_tokens)),
                        status=status,
                        stop_reason=stop_reason,
                    )
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
                            self._token_budget = int(consultation_result.outputs["token_budget"])
                            self.token_account.set_budget(self._token_budget)
                            self.transcript.set_token_budget(self._token_budget)

                            # Enforce phase caps so early planning can't consume the whole session.
                            try:
                                token_budget = int(self._token_budget)
                                caps = {
                                    "phase1.scope": int(token_budget * 0.02),
                                    "phase2.consultation": int(token_budget * 0.06),
                                    "phase3.planning_committee": int(token_budget * 0.10),
                                    "phase5.final_summary": int(token_budget * 0.03),
                                }
                                # Ensure non-zero caps for small budgets
                                caps = {k: max(2_000, int(v)) for k, v in caps.items()}
                                self.token_account.set_workflow_caps(caps)
                            except Exception:
                                pass

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
                
                # Get TaskGraph from consultation
                task_graph = consultation_result.outputs.get("task_graph")
                if not task_graph:
                    # Fallback: Try to load from file
                    task_graph = TaskGraph.load(self.session.project_path / "task_graph.json")
                
                if not task_graph:
                    console.warning("No TaskGraph generated. Using fallback execution.")
                    # Create a minimal default graph
                    task_graph = self._create_fallback_task_graph()
                    task_graph.save(self.session.project_path / "task_graph.json")
                
                # Enable auto-save on the task graph
                task_graph = self._ensure_task_graph_autosave(task_graph)
                
                # Store task graph in session
                self.session.context["task_graph"] = task_graph.to_dict()
            
            # At this point task_graph is set either from resume or consultation

            # Validate graph integrity early and emit DOT/PNG artifacts.
            try:
                from ..core.task_graph import validate_task_graph
                validate_task_graph(task_graph)
            except Exception as e:
                console.warning(f"Task graph validation issue: {e}")
                try:
                    self.transcript.log_system_event("task_graph_validation", str(e))
                except Exception:
                    pass

            try:
                from ..core import ProjectWriter
                ProjectWriter(
                    project_path=self.session.project_path,
                    project_name=self.session.project_name,
                    context_manager=self.context_manager,
                ).write_task_graph_visuals(task_graph=task_graph, render_png=True)
            except Exception as e:
                raise RuntimeError(f"Failed to render TaskGraph visuals: {e}") from e

            # Mandatory early all-agents planning committee before any execution.
            if not self._is_resumed and "planning_committee" not in self.session.completed_workflows:
                print()
                console.separator("━", 60)
                console.info("🔬 TEAM CONSULTATION")
                console.separator("━", 60)
                print()
                console.info("Assembling the full team for planning...")
                print()
                
                self.transcript.log_stage_transition(
                    "planning_committee",
                    "Phase 3: All-agents deliberation to finalize analysis plan",
                )
                self.session.current_workflow = "planning_committee"
                self.session.save()

                start_tokens = self.token_account.total_used
                from ..core.token_context import token_context
                
                # NO SPINNER - we want to show the agent dialogue to the user
                with token_context(workflow="phase3.planning_committee"):
                    planning_result = await self._execute_workflow("planning_committee")
                used_tokens = self.token_account.total_used - start_tokens

                # Record truthful workflow usage into BudgetManager (if initialized)
                try:
                    from ..config.budget_manager import get_budget_manager
                    status = "completed" if planning_result.status == WorkflowStatus.COMPLETED else "failed"
                    stop_reason = None
                    if planning_result.status == WorkflowStatus.PAUSED:
                        status = "user_stopped"
                        stop_reason = planning_result.metadata.get("stop_reason") if planning_result.metadata else None
                    get_budget_manager().record_usage(
                        "phase3.planning_committee",
                        max(0, int(used_tokens)),
                        status=status,
                        stop_reason=stop_reason,
                    )
                except Exception:
                    pass

                results["planning_committee"] = planning_result
                self.session.workflow_results["planning_committee"] = planning_result
                if planning_result.status == WorkflowStatus.COMPLETED:
                    self.session.completed_workflows.append("planning_committee")
                    console.success("Team consultation completed")
                    if planning_result.outputs:
                        self.session.context.update(planning_result.outputs)
                else:
                    console.error(f"Planning committee failed: {planning_result.error}")
                    return results

                self.session.current_workflow = None
                self.session.save()
                
                # Present plan to user and get approval
                plan_approved = await self._present_plan_for_approval(task_graph)
                if not plan_approved:
                    console.info("Plan not approved. Saving session for later.")
                    self.session.save()
                    return results
            
            # Execute tasks from the graph
            task_num = 0
            while not task_graph.is_complete():
                # Check for interrupts
                if self._interrupt_requested or self._exit_requested:
                    if self._exit_requested:
                        console.info("Exiting as requested")
                        self._log("User requested exit", console_print=False)
                    else:
                        console.warning("Interrupt requested, pausing execution")
                    break
                
                # Budget check with graceful wind-down
                if self._token_budget:
                    current_usage = self.token_account.total_used
                    budget_pct = (current_usage / self._token_budget) * 100
                    
                    if current_usage >= self._token_budget:
                        # Budget exhausted - interactive continuation prompt
                        console.warning(f"Budget reached ({current_usage:,}/{self._token_budget:,} tokens)")
                        self.transcript.log_budget_warning(100, "Budget reached - prompting for continuation")
                        
                        # Stop any active spinner for user interaction
                        spinner.stop()
                        
                        # Interactive continuation prompt
                        continuation_budget = await self._interactive_continuation_prompt(task_graph, results)
                        
                        if continuation_budget:
                            # User wants to continue - add to budget and resume
                            self._token_budget += continuation_budget
                            # CRITICAL: Also update the TokenAccount budget!
                            self.token_account.set_budget(self._token_budget)
                            self.transcript.set_token_budget(self._token_budget)
                            console.success(f"Budget extended by {continuation_budget:,} tokens (new total: {self._token_budget:,})")
                            self._continuation_generated = False  # Reset flag for next threshold
                            
                            # Restart spinner for next task
                            spinner = Spinner(f"Continuing work")
                            spinner.start()
                            continue
                        else:
                            # User chose to stop
                            console.info("Saving progress and finishing up...")
                            break
                    
                    # At 95%+ budget, warn user proactively (but don't stop)
                    if budget_pct >= 95 and not getattr(self, '_continuation_generated', False):
                        console.warning(f"Budget at {budget_pct:.0f}% - approaching limit")
                        self._continuation_generated = True  # Only warn once
                    
                    # Show budget context at 60%+
                    if budget_pct >= 60:
                        progress = task_graph.get_progress()
                        remaining_tokens = self._token_budget - current_usage
                        console.info(f"Budget: {budget_pct:.0f}% used | Tasks: {progress['completed']}/{progress['total_tasks']} | Remaining: {remaining_tokens:,} tokens")
                
                # Get next ready task
                next_task = task_graph.get_next_task()
                if not next_task:
                    # No tasks ready - might be waiting on failed dependencies
                    ready = task_graph.get_ready_tasks()
                    if not ready:
                        console.info("All executable tasks completed.")
                        break
                
                task_num += 1
                task_id = next_task.id
                task_name = next_task.name
                task_owner = next_task.owner
                
                # Mark task as in progress
                next_task.mark_in_progress()
                
                # Map task to module
                module_name = self._task_to_module(task_id)
                
                console.info(f"Task {task_num}: {task_name} ({task_owner})")
                console.info(f"Task objective: {next_task.description}")
                
                spinner = Spinner(f"Running {task_name}")
                spinner.start()
                self.transcript.log_stage_transition(task_id, f"Starting task: {task_name}")
                self.session.current_module = task_id
                self.session.save()
                
                # Execute the task
                from ..core.token_context import token_context
                start_tokens = self.token_account.total_used
                start_txn_idx = 0
                try:
                    start_txn_idx = len(self.token_account.iter_transactions())
                except Exception:
                    start_txn_idx = 0
                
                task_module_key = f"phase4.task.{task_id}.{module_name}"
                with token_context(workflow=task_module_key, trigger=f"task:{task_id}"):
                    result = await self._execute_task(
                        task=next_task,
                        module_name=module_name,
                    )
                
                used_tokens = self.token_account.total_used - start_tokens

                # Build a lightweight breakdown snapshot for learning (top token buckets)
                breakdown: dict[str, Any] | None = None
                try:
                    txns = self.token_account.iter_transactions()[start_txn_idx:]
                    buckets: dict[str, int] = {}
                    for t in txns:
                        trigger = getattr(t, "trigger", None) or "(none)"
                        op = getattr(t, "operation", None) or "(none)"
                        agent = getattr(t, "agent_id", None) or "(unknown)"
                        k = f"{agent} | {trigger} | {op}"
                        buckets[k] = buckets.get(k, 0) + int(getattr(t, "total_tokens", 0))
                    top = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:12]
                    breakdown = {
                        "top_token_buckets": [{"key": k, "tokens": v} for k, v in top],
                        "txn_count": len(txns),
                    }
                except Exception:
                    breakdown = None
                
                # Record usage to BudgetHistory for learning
                try:
                    from ..config.budget_manager import get_budget_manager
                    # Outcome-aware learning: completed runs only update means.
                    status = "completed"
                    stop_reason = None
                    if result.status == WorkflowStatus.PAUSED:
                        stop_reason = (result.metadata or {}).get("stop_reason")
                        status = "budget_exhausted" if stop_reason == "budget_exceeded" else "user_stopped"
                    elif result.status == WorkflowStatus.FAILED:
                        status = "failed"
                        stop_reason = (result.metadata or {}).get("stop_reason")

                    get_budget_manager().record_usage(
                        task_workflow_key,
                        max(0, int(used_tokens)),
                        status=status,
                        stop_reason=stop_reason,
                        breakdown=breakdown,
                    )
                except Exception:
                    pass
                
                # Update task graph
                if result.status == WorkflowStatus.COMPLETED:
                    task_graph.mark_completed(task_id, actual_tokens=int(used_tokens), outputs=result.outputs)
                    spinner.stop(f"{task_name} completed")
                    self._log(f"Completed: {task_name} ({used_tokens:,} tokens)", console_print=False)
                    self.transcript.log_system_event(
                        "task_complete",
                        f"{task_name} completed successfully",
                        {"summary": result.summary, "tokens": used_tokens} if result.summary else None
                    )
                    
                    # Update session context with outputs
                    if result.outputs:
                        self.session.context.update(result.outputs)

                        # Canonical persistence for write-up outputs
                        if module_name in ("writeup_results", "build_report") and result.outputs.get("final_report"):
                            try:
                                from ..core import ProjectWriter
                                writer = ProjectWriter(
                                    project_path=self.session.project_path,
                                    project_name=self.session.project_name,
                                    context_manager=self.context_manager,
                                )
                                md_path = writer.write_summary_report(str(result.outputs["final_report"]))
                                writer.write_summary_report_pdf(markdown_path=md_path)
                            except Exception:
                                pass
                    
                    self.session.completed_modules.append(task_id)
                
                elif result.status == ModuleStatus.PAUSED:
                    # Controlled stop (budget exhausted or user stop). Do NOT mark task completed.
                    try:
                        # Preserve any partial outputs for later resume/debug.
                        if result.outputs:
                            next_task.outputs.update(result.outputs)
                    except Exception:
                        pass

                    # Return task to READY so it can be resumed.
                    next_task.status = GraphTaskStatus.READY
                    spinner.stop_error(f"{task_name} paused")
                    self._log(f"Paused: {task_name} ({used_tokens:,} tokens)", console_print=False)
                    self.transcript.log_system_event(
                        "task_paused",
                        f"{task_name} paused",
                        {"stop_reason": (result.metadata or {}).get("stop_reason"), "tokens": used_tokens},
                    )

                    # If paused due to budget exhaustion, immediately prompt for continuation.
                    stop_reason = (result.metadata or {}).get("stop_reason")
                    if stop_reason == "budget_exceeded" and self._token_budget:
                        console.warning(f"Budget reached ({self.token_account.total_used:,}/{self._token_budget:,} tokens)")
                        self.transcript.log_budget_warning(100, "Budget reached - prompting for continuation")
                        continuation_budget = await self._interactive_continuation_prompt(task_graph, results)
                        if continuation_budget:
                            self._token_budget += continuation_budget
                            self.token_account.set_budget(self._token_budget)
                            self.transcript.set_token_budget(self._token_budget)
                            console.success(
                                f"Budget extended by {continuation_budget:,} tokens (new total: {self._token_budget:,})"
                            )
                            self._continuation_generated = False
                            spinner = Spinner("Continuing work")
                            spinner.start()
                            continue
                        else:
                            console.info("Saving progress and finishing up...")
                            break

                elif result.status == WorkflowStatus.FAILED:
                    task_graph.mark_failed(task_id, result.error or "Unknown error")
                    spinner.stop_error(f"{task_name} failed: {result.error}")
                    self._log(f"Failed: {task_name} - {result.error}", console_print=False)
                    self.transcript.log_system_event("task_failed", f"{task_name} failed", {"error": result.error})
                    
                    # Ask user how to handle
                    proceed = await self._handle_task_failure(next_task, result)
                    if not proceed:
                        break
                else:
                    task_graph.mark_completed(task_id, actual_tokens=int(used_tokens))
                    spinner.stop(f"{task_name} finished: {result.status.value}")
                
                # Save progress
                results[task_id] = result
                self.session.workflow_results[task_id] = result
                task_graph.save(self.session.project_path / "task_graph.json")

                # Keep task graph visuals current as the graph evolves.
                try:
                    from ..core import ProjectWriter
                    ProjectWriter(
                        project_path=self.session.project_path,
                        project_name=self.session.project_name,
                        context_manager=self.context_manager,
                    ).write_task_graph_visuals(task_graph=task_graph, render_png=True)
                except Exception as e:
                    # TaskGraph PNG rendering is mandatory; fail loudly with actionable detail.
                    raise RuntimeError(f"Failed to render TaskGraph visuals: {e}") from e

                self.session.save()
            
            # Final summary
            console.info("Generating final summary...")
            from ..core.token_context import token_context
            with token_context(workflow="phase5.final_summary"):
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
    
    def _task_to_module(self, task_id: str) -> str:
        """Map a task ID to a module name."""
        task_lower = task_id.lower()
        
        # Check explicit mapping first
        for key, module in self.TASK_MODULE_MAP.items():
            if key in task_lower:
                return module
        
        # Default to analysis_execution for most tasks
        return "analysis_execution"
    
    # Backward compat alias
    _task_to_workflow = _task_to_module
    
    def _create_fallback_task_graph(self) -> TaskGraph:
        """Create a minimal fallback TaskGraph if consultation didn't produce one."""
        graph = TaskGraph(project_name=self.session.project_name)
        
        # Simple 3-task pipeline
        graph.add_task(
            task_id="analysis",
            name="Main Analysis",
            description="Perform the primary analysis requested by the user",
            owner="hinton",
            dependencies=[],
            estimated_tokens=100_000,
        )
        graph.add_task(
            task_id="documentation",
            name="Document Results",
            description="Create documentation and report of findings",
            owner="gould",
            dependencies=["analysis"],
            estimated_tokens=50_000,
        )
        
        return graph
    
    async def _execute_task(
        self,
        task: TaskNode,
        module_name: str,
    ) -> ModuleResult:
        """
        Execute a single task from the TaskGraph.
        
        Routes to the appropriate module based on the task type.
        """
        if module_name not in self.MODULE_CLASSES:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Unknown module for task: {module_name}",
            )
        
        # Instantiate module
        module_class = self.MODULE_CLASSES[module_name]
        module = module_class(
            agents=self.agents,
            context_manager=self.context_manager,
            project_path=self.session.project_path,
        )
        
        # Prepare inputs
        inputs = self._prepare_module_inputs(module)
        inputs["task_description"] = task.description
        inputs["task_owner"] = task.owner
        inputs["estimated_tokens"] = task.estimated_tokens
        
        # Build context from completed tasks
        supporting_context = self._build_supporting_context()
        
        # Execute with autonomous mode
        result = await module.execute_autonomous(
            inputs=inputs,
            lead_agent=task.owner,
            objective=task.description,
            supporting_context=supporting_context,
        )
        
        return result
    
    async def _handle_task_failure(
        self,
        task: TaskNode,
        result: ModuleResult,
    ) -> bool:
        """
        Handle a task failure with user-friendly prompts.
        
        Returns True to continue, False to stop.
        """
        from ..utils import Spinner
        was_spinning = Spinner.pause_for_input()
        
        print()
        console.warning(f"Task '{task.name}' encountered an issue: {result.error}")
        print()
        print("  How would you like to proceed? You can retry, skip this task, or stop and save.")
        print()
        
        try:
            response = input("  \033[1;32m▶ Your response:\033[0m ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = "stop"
        
        if was_spinning:
            Spinner.resume_after_input()
        
        # Interpret natural language
        retry_words = {"retry", "again", "redo", "try again", "1"}
        skip_words = {"skip", "continue", "next", "move on", "2"}
        stop_words = {"stop", "save", "exit", "quit", "done", "3"}
        
        if any(word in response for word in retry_words):
            # Reset task for retry
            task.status = GraphTaskStatus.READY
            return True
        elif any(word in response for word in skip_words):
            task.mark_skipped(f"User skipped after failure: {result.error}")
            return True
        else:
            return False
    
    async def _generate_continuation_plan(
        self,
        task_graph: TaskGraph,
        results: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """
        Generate a continuation plan when budget is exhausted or nearly exhausted.
        
        Uses the Bohr agent to analyze progress and suggest:
        1. What was accomplished
        2. What remains to be done
        3. Recommended token budget for continuation
        4. Key next steps
        
        Returns the plan data for interactive continuation prompting.
        Also saves the plan to continuation_plan.json for resume.
        """
        from ..utils import console
        
        try:
            bohr = self.agents.get("bohr")
            if not bohr:
                console.warning("Bohr agent unavailable for continuation planning")
                return None
            
            # Gather progress information
            progress = task_graph.get_progress()
            completed_tasks = [t for t in task_graph.tasks.values() if t.status == GraphTaskStatus.COMPLETED]
            remaining_tasks = [t for t in task_graph.tasks.values() if t.status in [GraphTaskStatus.READY, GraphTaskStatus.PENDING]]
            
            # Calculate estimated remaining tokens from history
            try:
                from ..config.budget_history import get_budget_history
                history = get_budget_history()
                remaining_estimate = sum(
                    history.estimate(self._task_to_workflow(t.id)).get("estimated_tokens", t.estimated_tokens)
                    for t in remaining_tasks
                )
            except Exception:
                remaining_estimate = sum(t.estimated_tokens for t in remaining_tasks)
            
            # Get current budget usage
            current_usage = self.token_account.total_used
            budget_used_pct = (current_usage / self._token_budget * 100) if self._token_budget else 0
            
            # Build context for Bohr
            completed_summary = "\n".join([
                f"- {t.name} ({t.owner}): {t.actual_tokens or t.estimated_tokens:,} tokens"
                for t in completed_tasks
            ]) or "No tasks completed yet"
            
            remaining_summary = "\n".join([
                f"- {t.name} ({t.owner}): ~{t.estimated_tokens:,} tokens estimated"
                for t in remaining_tasks
            ]) or "All tasks completed"
            
            # Ask Bohr for continuation recommendation
            prompt = f"""Budget has reached {budget_used_pct:.0f}%. Analyze progress and provide continuation recommendations.

## Completed Tasks
{completed_summary}

## Remaining Tasks
{remaining_summary}

## Budget Status
- Used: {current_usage:,} tokens
- Budget: {self._token_budget:,} tokens
- Estimated remaining need: {remaining_estimate:,} tokens

## Your Task
Provide a brief JSON response with:
1. "progress_summary": 1-2 sentence summary of what was accomplished
2. "remaining_work": What still needs to be done
3. "recommended_continuation_budget": Integer tokens needed to complete (be realistic)
4. "priority_next_steps": List of 2-3 most important next actions
5. "can_deliver_value_now": Boolean - is current progress useful on its own?

Return ONLY valid JSON."""

            # Use minimal tokens for this query
            response = await bohr.simple_query(
                query=prompt,
                context="Budget exhaustion - generate continuation plan"
            )
            
            # Try to extract JSON
            from ..utils import extract_json_from_text
            plan_data = extract_json_from_text(response, fallback={
                "progress_summary": f"Completed {progress['completed']}/{progress['total_tasks']} tasks",
                "remaining_work": f"{len(remaining_tasks)} tasks remaining",
                "recommended_continuation_budget": remaining_estimate,
                "priority_next_steps": [t.name for t in remaining_tasks[:3]],
                "can_deliver_value_now": progress['completed'] > 0,
            })
            
            # Add metadata
            plan_data["generated_at"] = datetime.now().isoformat()
            plan_data["tokens_used"] = current_usage
            plan_data["budget"] = self._token_budget
            plan_data["completed_tasks"] = [t.id for t in completed_tasks]
            plan_data["remaining_tasks"] = [t.id for t in remaining_tasks]
            
            # Save continuation plan
            plan_path = self.session.project_path / "continuation_plan.json"
            with open(plan_path, "w") as f:
                json.dump(plan_data, f, indent=2)
            
            self.transcript.log_system_event(
                "continuation_plan",
                "Generated continuation plan due to budget exhaustion",
                plan_data
            )
            
            return plan_data
            
        except Exception as e:
            console.warning(f"Could not generate continuation plan: {e}")
            self._log(f"Continuation plan generation failed: {e}", console_print=False)
            return None
    
    async def _interactive_continuation_prompt(
        self,
        task_graph: TaskGraph,
        results: dict[str, Any],
    ) -> Optional[int]:
        """
        Interactive prompt for continuation when budget is exhausted.
        
        Displays a plain-language summary from Bohr and lets the user decide:
        - Continue with recommended budget
        - Continue with custom budget
        - Change scope/project
        - Stop completely
        
        Returns:
            New token budget to continue with, or None to stop
        """
        from ..utils import console, Spinner
        
        # Pause any active spinner for input
        Spinner.pause_for_input()
        
        # Generate continuation plan
        plan = await self._generate_continuation_plan(task_graph, results)
        
        if not plan:
            console.warning("Unable to generate continuation summary.")
            return None
        
        # Display the continuation summary
        print()
        console.separator("━", 60)
        print(f"  {Style.BOLD}{Style.CYAN}📋 BUDGET REACHED - CONTINUATION OPTIONS{Style.RESET}")
        console.separator("━", 60)
        print()
        
        # What was accomplished
        print(f"  {Style.GREEN}✓ COMPLETED:{Style.RESET}")
        print(f"    {plan.get('progress_summary', 'Work in progress')}")
        print()
        
        # What files/outputs were created
        if plan.get('completed_tasks'):
            print(f"  {Style.GREEN}✓ DELIVERABLES:{Style.RESET}")
            for task_id in plan['completed_tasks'][:5]:
                print(f"    • {task_id}")
            if len(plan['completed_tasks']) > 5:
                print(f"    • ... and {len(plan['completed_tasks']) - 5} more")
            print()
        
        # What remains
        print(f"  {Style.YELLOW}⏳ REMAINING:{Style.RESET}")
        print(f"    {plan.get('remaining_work', 'Additional work planned')}")
        print()
        
        # Budget recommendation
        recommended = plan.get('recommended_continuation_budget', 100000)
        estimated_cost = (recommended / 1_000_000) * 3.0 + (recommended / 1_000_000) * 15.0  # rough estimate
        print(f"  {Style.CYAN}💰 RECOMMENDED BUDGET:{Style.RESET} {recommended:,} tokens (~${estimated_cost:.2f})")
        print()
        
        # Priority next steps
        if plan.get('priority_next_steps'):
            print(f"  {Style.CYAN}📌 NEXT STEPS:{Style.RESET}")
            for i, step in enumerate(plan['priority_next_steps'][:3], 1):
                print(f"    • {step}")
            print()
        
        console.separator("─", 60)
        print()
        print(f"  {Style.BOLD}What would you like to do?{Style.RESET}")
        print()
        print(f"  You can continue with the recommended {recommended:,} tokens, specify a custom amount,")
        print(f"  ask to use fewer tokens, or stop and save progress.")
        print()
        
        try:
            response = input(f"  {Style.BOLD}▶ Your response: {Style.RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        
        if not response or response == "":
            # Continue with recommended budget
            console.info(f"Continuing with {recommended:,} additional tokens...")
            return recommended
        
        if response == "stop":
            console.info("Saving progress and stopping...")
            return None
        
        if response == "less":
            # Ask Bohr for a minimal budget estimate
            try:
                bohr = self.agents.get("bohr")
                if bohr:
                    min_response = await bohr.simple_query(
                        query=f"What's the MINIMUM tokens needed to produce something useful from the remaining work? Current remaining: {plan.get('remaining_work')}. Be aggressive - cut scope if needed. Return ONLY an integer.",
                        context="Minimal budget estimation"
                    )
                    # Extract number from response
                    import re
                    numbers = re.findall(r'\d+', min_response.replace(',', ''))
                    if numbers:
                        minimal = int(numbers[0])
                        console.info(f"Bohr suggests {minimal:,} tokens as minimum for useful output")
                        print(f"  Would you like to proceed with {minimal:,} tokens?")
                        confirm = input("  \033[1;32m▶ Your response:\033[0m ").strip().lower()
                        approval_words = {"yes", "y", "ok", "proceed", "sure", "go", "fine", ""}
                        if any(word == confirm or word in confirm for word in approval_words):
                            return minimal
            except Exception:
                pass
            # Fallback: use half recommended
            half = recommended // 2
            console.info(f"Using {half:,} tokens (half of recommended)")
            return half
        
        # Try to parse as number
        try:
            custom = int(response.replace(',', '').replace('k', '000').replace('K', '000'))
            if custom < 1000:
                custom *= 1000  # Assume they meant thousands
            console.info(f"Continuing with {custom:,} tokens...")
            return custom
        except ValueError:
            # Treat as scope change or other input
            console.info(f"Unrecognized input '{response}'. Saving progress for later...")
            return None
    
    async def _execute_module(
        self,
        module_name: str,
        autonomous: bool = False,
        objective: Optional[str] = None,
    ) -> ModuleResult:
        """
        Execute a specific module.
        
        Args:
            module_name: Name of module to execute
            autonomous: If True, use autonomous execution mode (agent-driven)
            objective: High-level objective for autonomous mode
            
        Returns:
            ModuleResult from execution
        """
        if module_name not in self.MODULE_CLASSES:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Unknown module: {module_name}",
            )
        
        # Instantiate module
        module_class = self.MODULE_CLASSES[module_name]
        module = module_class(
            agents=self.agents,
            context_manager=self.context_manager,
            project_path=self.session.project_path,
        )
        
        # Prepare inputs from session context
        inputs = self._prepare_module_inputs(module)
        
        # Autonomous mode - let lead agent decide approach
        if autonomous:
            lead_agent = module.primary_agents[0] if module.primary_agents else "darwin"
            
            # Build context from prior module results
            supporting_context = self._build_supporting_context()
            
            # Use provided objective or generate from module description
            exec_objective = objective or f"Complete the {module_name} phase: {module.description}"
            
            result = await module.execute_autonomous(
                inputs=inputs,
                lead_agent=lead_agent,
                objective=exec_objective,
                supporting_context=supporting_context,
            )
            return result
        
        # Standard mode - structured execution with checkpoint support
        checkpoint_path = self.session.project_path / "checkpoints" / f"{module_name}_checkpoint.json"
        checkpoint = None
        if checkpoint_path.exists():
            try:
                checkpoint = module.load_checkpoint(checkpoint_path)
                self._log(f"Resuming {module_name} from checkpoint")
            except Exception:
                pass  # Start fresh if checkpoint is invalid
        
        # Execute module
        result = await module.execute(inputs=inputs, checkpoint=checkpoint)
        
        return result
    
    # Backward compat alias
    _execute_workflow = _execute_module
    
    def _build_supporting_context(self) -> str:
        """Build context string from completed modules for autonomous execution."""
        context_parts = []
        
        for mod_name, mod_result in self.session.module_results.items():
            if mod_result.status == ModuleStatus.COMPLETED:
                context_parts.append(f"## {mod_name.replace('_', ' ').title()} Results")
                context_parts.append(mod_result.summary or "Completed successfully")
                if mod_result.artifacts:
                    context_parts.append(f"Artifacts: {', '.join(mod_result.artifacts)}")
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
    
    def _prepare_module_inputs(self, module: Module) -> dict[str, Any]:
        """
        Prepare inputs for a module from session context.
        
        Args:
            module: Module to prepare inputs for
            
        Returns:
            Dictionary of inputs
        """
        inputs = {}
        
        # Map context keys to module inputs
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
        
        # Pass task graph for plan dissemination (so agents see full context)
        if "task_graph" in context:
            inputs["task_graph"] = context["task_graph"]
        
        # Pass token budget to modules that can use it for mode selection
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

        # Phase 1 scope confirmation artifacts (used to avoid redundant consultation prompting)
        if "scope_confirmation" in context:
            inputs["scope_confirmation"] = context["scope_confirmation"]
        if "scope_response" in context:
            inputs["scope_response"] = context["scope_response"]
        
        # Data paths - scan ReadData
        data_paths = self._scan_data_paths()
        inputs["data_paths"] = data_paths
        
        # Research topic from user request
        if "research_topic" not in inputs and "user_request" in context:
            inputs["research_topic"] = context["user_request"]
        
        return inputs
    
    # Backward compat alias
    _prepare_workflow_inputs = _prepare_module_inputs
    
    def _show_post_consultation_summary(self, result: ModuleResult) -> None:
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
    
    async def _present_plan_for_approval(self, task_graph: TaskGraph) -> bool:
        """
        Present the complete plan to user with per-step token estimates and get approval.
        
        Uses Bohr to present the plan in natural language. User can:
        - Approve as-is
        - Request adjustments (in plain English)
        - Decline/postpone
        
        Args:
            task_graph: The TaskGraph with planned tasks
            
        Returns:
            True if approved and ready to execute, False to stop
        """
        from ..utils import console
        from ..config.budget_history import BudgetHistory
        
        # Get all tasks for display
        all_tasks = list(task_graph.tasks.values())
        
        # Get Bayesian token estimates per task from BudgetHistory
        try:
            budget_history = BudgetHistory()
            task_estimates = []
            total_estimated = 0
            
            # Map complexity string to float
            complexity_str = self.session.context.get("complexity", "moderate")
            complexity_map = {"simple": 0.3, "moderate": 0.5, "complex": 0.7, "exploratory": 0.9}
            complexity = complexity_map.get(complexity_str, 0.5)
            
            for task in all_tasks:
                # Map task to module for estimate
                module_name = self._task_to_module(task.id)
                estimate_result = budget_history.estimate(module_name, complexity)
                estimate = estimate_result.get("estimated_tokens", 50_000)
                task_estimates.append({
                    "task_id": task.id,
                    "name": task.name,
                    "owner": task.owner,
                    "description": task.description,
                    "estimated_tokens": estimate,
                })
                total_estimated += estimate
        except Exception as e:
            # Fallback to rough estimates if BudgetHistory fails
            self._log(f"BudgetHistory unavailable: {e}", console_print=False)
            task_estimates = []
            total_estimated = 0
            base_estimate = 50_000  # Conservative default
            for task in all_tasks:
                task_estimates.append({
                    "task_id": task.id,
                    "name": task.name,
                    "owner": task.owner,
                    "description": task.description,
                    "estimated_tokens": base_estimate,
                })
                total_estimated += base_estimate
        
        # Format task list for Bohr
        task_list_text = "\n".join([
            f"  {i+1}. {t['name']} ({t['owner']}): ~{t['estimated_tokens']:,} tokens\n      {t['description']}"
            for i, t in enumerate(task_estimates)
        ])
        
        # Have Bohr present the plan naturally
        bohr = self.agents.get("bohr")
        if not bohr:
            # Fallback: simple text display
            console.info("Plan ready for approval:")
            print(task_list_text)
            print(f"\nTotal estimated: {total_estimated:,} tokens")
            print("\n  Does this plan look good? Feel free to approve, adjust, or decline.")
            try:
                response = input("\n  \033[1;32m▶ Your response:\033[0m ").strip().lower()
                approval_words = {"yes", "y", "approve", "ok", "okay", "sure", "go", "proceed", "looks good", "good", "fine"}
                return any(word in response for word in approval_words)
            except (EOFError, KeyboardInterrupt):
                return False
        
        # Ask Bohr to present the plan
        plan_context = f"""
The team has developed a complete analysis plan. Present it to the user naturally.

TOKEN BUDGET: {self._token_budget:,} tokens total
ESTIMATED USAGE: {total_estimated:,} tokens ({(total_estimated / self._token_budget * 100):.0f}% of budget)

PLANNED STEPS:
{task_list_text}

Instructions:
1. Present this plan conversationally (don't just list it robotically)
2. Explain what each step will accomplish
3. Highlight the token estimates per step
4. Note if the total is within budget or tight
5. Ask if the plan looks good and if they'd like to proceed
6. Mention they can request changes in plain English

Keep it concise but informative. Don't be overly formal.
"""
        
        from ..core.token_context import token_context
        with token_context(workflow="plan_presentation"):
            plan_message = await bohr.simple_query(
                query="Present the analysis plan to the user and ask for approval.",
                context=plan_context,
                max_tokens=1024,
            )
        
        console.agent_message("BOHR", plan_message)
        print()
        
        # Get user response
        try:
            user_response = input("  \033[1;32m▶ Your response:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        
        if not user_response:
            return False
        
        # Have Bohr interpret the response
        interpret_context = f"""
The user was asked to approve the analysis plan. Their response was:
"{user_response}"

Determine their intent:
- If they approve (yes, looks good, proceed, let's go, etc.) -> respond with exactly: APPROVED
- If they want adjustments -> respond with: ADJUST: <summary of requested changes>
- If they decline/postpone -> respond with exactly: DECLINED

Respond with ONLY ONE of: APPROVED, ADJUST: <changes>, or DECLINED
"""
        
        with token_context(workflow="plan_interpretation"):
            interpretation = await bohr.simple_query(
                query="Interpret the user's response to the plan approval request.",
                context=interpret_context,
                max_tokens=256,
            )
        
        interpretation = interpretation.strip().upper()
        
        if interpretation.startswith("APPROVED"):
            console.success("Plan approved! Starting execution...")
            self.transcript.log_system_event("plan_approved", "User approved execution plan")
            return True
        
        elif interpretation.startswith("ADJUST"):
            # User wants changes - for now, we'll note this and proceed with a modified approach
            adjustment = interpretation.replace("ADJUST:", "").strip()
            console.info(f"Adjustment requested: {adjustment}")
            
            # Acknowledge and ask if they want to proceed with modified understanding
            with token_context(workflow="plan_adjustment"):
                ack_message = await bohr.simple_query(
                    query="Acknowledge the user's adjustment request and ask if we should proceed with this modification in mind.",
                    context=f"User requested: {adjustment}\nOriginal plan had {len(all_tasks)} steps.",
                    max_tokens=256,
                )
            
            console.agent_message("BOHR", ack_message)
            print()
            
            try:
                confirm = input("  \033[1;32m▶ Your response:\033[0m ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
            
            approval_words = {"yes", "y", "ok", "proceed", "sure", "go", "do it", "fine"}
            if any(word in confirm for word in approval_words):
                # Store the adjustment context for execution
                self.session.context["user_adjustment"] = adjustment
                console.success("Proceeding with adjusted approach...")
                return True
            else:
                console.info("Understood. Saving session for later.")
                return False
        
        else:  # DECLINED or unknown
            console.info("Plan declined. Saving session for later.")
            self.transcript.log_system_event("plan_declined", "User declined execution plan")
            return False
    
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
    
    async def _handle_module_failure(
        self,
        module_name: str,
        result: ModuleResult,
    ) -> bool:
        """
        Handle a module failure with user-friendly prompts.
        
        Args:
            module_name: Name of failed module
            result: Failure result
            
        Returns:
            True to continue, False to stop
        """
        from ..utils import Spinner
        was_spinning = Spinner.pause_for_input()
        
        print()
        console.warning(f"The {module_name.replace('_', ' ')} module encountered an issue: {result.error}")
        print()
        print("  Would you like to retry this module, skip to the next step, or stop and save progress?")
        print()
        
        try:
            response = input("  \033[1;32m▶ Your response:\033[0m ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = "stop"
        
        if was_spinning:
            Spinner.resume_after_input()
        
        retry_words = {"retry", "again", "try", "redo", "repeat"}
        skip_words = {"skip", "next", "continue", "move on", "proceed"}
        stop_words = {"stop", "save", "quit", "exit", "halt", "end"}
        
        if any(word in response for word in retry_words):
            return True
        elif any(word in response for word in skip_words):
            self.session.completed_modules.append(f"{module_name}_skipped")
            return True
        else:
            # Default to stop for safety
            return False
    
    # Backward compat alias
    _handle_workflow_failure = _handle_module_failure
    
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
        """Create a deterministic, truthful budget breakdown section with cache stats."""
        usage = self.token_account.usage_summary
        if usage.get("total_used", 0) <= 0:
            return ""

        lines: list[str] = []
        lines.append("## Budget Breakdown (Authoritative)")
        lines.append("")
        lines.append(f"- Total: {usage.get('total_used', 0):,} tokens ({usage.get('total_input', 0):,} in + {usage.get('total_output', 0):,} out)")
        if usage.get("budget"):
            lines.append(f"- Budget: {usage['total_used']:,} / {usage['budget']:,} ({usage.get('percentage_used', 0):.1f}% used)")
        
        # Enhanced cache visibility
        cache_read = usage.get("cache_read", 0)
        cache_creation = usage.get("cache_creation", 0)
        if cache_read or cache_creation:
            lines.append(f"- Cache: {cache_read:,} read, {cache_creation:,} created")
            
            # Calculate cache efficiency
            cache_efficiency = self.token_account.cache_efficiency
            if cache_efficiency.get("hit_rate", 0) > 0:
                lines.append(f"- Cache Hit Rate: {cache_efficiency['hit_rate']:.1f}%")
                if cache_efficiency.get("cost_saved", 0) > 0:
                    lines.append(f"- Cache Savings: ~${cache_efficiency['cost_saved']:.3f}")
        
        lines.append(f"- Transaction count: {usage.get('transaction_count', 0):,}")
        
        # Show estimated cost with model pricing
        if usage.get("estimated_cost", 0) > 0:
            lines.append(f"- Estimated Cost: ${usage['estimated_cost']:.2f}")
            model = self.token_account.model
            lines.append(f"- Model: {model}")

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
        
        # Tool usage breakdown (outputs become input tokens)
        tool_summary = self.token_account.get_tool_usage_summary()
        if tool_summary:
            lines.append("")
            lines.append("**Tool Output Tokens (fed back to LLM)**")
            total_tool_tokens = sum(s["estimated_tokens"] for s in tool_summary)
            lines.append(f"Total tool output: ~{total_tool_tokens:,} tokens (est.)")
            for s in tool_summary[:8]:  # Top 8 tools
                pct = (s["estimated_tokens"] / total_tool_tokens * 100) if total_tool_tokens > 0 else 0
                lines.append(
                    f"- {s['tool']}: {s['calls']} calls → ~{s['estimated_tokens']:,} tokens ({pct:.0f}%)"
                )

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
                console.info(f"Detected continuation of existing project: {continue_project}")
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
            console.info(f"Suggested project name: {suggested_name}")
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
                console.info(f"Resuming project: {project_name}")
                if self.transcript:
                    self.transcript.log_agent_message("bohr", f"Resuming project: {project_name}")
            else:
                console.info(f"Creating project: {project_name}")
                if self.transcript:
                    self.transcript.log_agent_message("bohr", f"Creating project: {project_name}")
            
            return project_name
            
        except Exception as e:
            # Fallback to simple timestamp-based name
            console.warning(f"Could not generate smart name: {e}")
            fallback = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            console.info(f"Using default project name: {fallback}")
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
    
    def _ensure_task_graph_autosave(self, task_graph: TaskGraph) -> TaskGraph:
        """
        Ensure a TaskGraph has auto-save enabled for its project.
        
        This enables the TaskGraph to persist immediately after any mutation
        (mark_completed, mark_failed, add_node, etc.) without requiring
        explicit save calls in the orchestrator.
        
        Args:
            task_graph: The TaskGraph to configure
            
        Returns:
            The same TaskGraph with save path set
        """
        if self.session:
            save_path = self.session.project_path / "task_graph.json"
            task_graph.set_save_path(save_path)
        return task_graph
    
    def _log(self, message: str, console_print: bool = True) -> None:
        """Log orchestrator message."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [ORCHESTRATOR] {message}"
        
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
    Main entry point for MiniLab.
    
    Implements the full startup flow:
    1. User provides request
    2. Bohr provides understanding + project name + clarifying questions (ONE call)
    3. User confirms/adjusts in plain English
    4. Full team consultation to develop specific plan
    5. Bohr presents plan with per-step token estimates
    6. User approves or adjusts
    7. Execution begins
    
    Args:
        request: User's analysis request
        project_name: Optional pre-set project name (for testing/automation)
        
    Returns:
        Results dictionary
    """
    import signal
    from ..utils import console, Spinner
    
    orchestrator = MiniLabOrchestrator()
    
    def handle_interrupt(signum, frame):
        """Handle Ctrl+C with user options."""
        # Pause any running spinner
        was_spinning = Spinner.pause_for_input()
        
        print("\n")
        console.info("⏸️  Interrupted! What would you like to do?")
        print("  You can pause to provide guidance, skip to the next phase, save and exit, or just continue.")
        print()
        
        try:
            response = input("  \033[1;32m▶ Your response:\033[0m ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = "exit"  # Default to save and exit on double Ctrl+C
        
        if was_spinning:
            Spinner.resume_after_input()
        
        guidance_words = {"pause", "guide", "guidance", "help", "input", "tell", "instruct"}
        skip_words = {"skip", "next", "phase", "move", "advance"}
        exit_words = {"save", "exit", "quit", "stop", "done", "end"}
        continue_words = {"continue", "resume", "go", "proceed", "cancel", "nevermind", "nothing"}
        
        if any(word in response for word in guidance_words):
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
        elif any(word in response for word in skip_words):
            console.info("Skipping to next workflow phase...")
            orchestrator.interrupt()
        elif any(word in response for word in exit_words):
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
        # Phase 1: Bohr's understanding + project name + questions (conversational loop)
        # This replaces the hard-coded "scope confirmation" phase
        project_name, user_confirmed = await orchestrator.bohr_understanding_phase(
            user_request=request,
            preset_project_name=project_name,
        )
        
        if not user_confirmed:
            return {"status": "cancelled", "message": "User cancelled during understanding phase"}
        
        # Start the session with the confirmed project name
        await orchestrator.start_session(
            user_request=request,
            project_name=project_name,
        )
        
        # CRITICAL: Mark that understanding phase completed
        # This tells consultation to skip redundant user interaction
        orchestrator.session.context["scope_confirmed"] = True
        orchestrator.session.context["scope_confirmation"] = f"User confirmed understanding of request for project '{project_name}'"
        orchestrator.session.context["scope_response"] = "User confirmed via understanding phase"
        
        # Check for exit request
        if orchestrator._exit_requested:
            return {"status": "interrupted", "message": "User requested exit"}
        
        # Run the main orchestration (consultation, team planning, approval, execution)
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
