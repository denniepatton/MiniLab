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


class MajorWorkflow(Enum):
    """Major workflow types that users can request."""
    BRAINSTORMING = "brainstorming"
    LITERATURE_REVIEW = "literature_review"
    START_PROJECT = "start_project"
    WORK_ON_EXISTING = "work_on_existing"
    EXPLORE_DATASET = "explore_dataset"


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
    
    # Major workflow -> mini-workflow sequences
    WORKFLOW_SEQUENCES = {
        MajorWorkflow.BRAINSTORMING: [
            "consultation",
            "planning_committee",
        ],
        MajorWorkflow.LITERATURE_REVIEW: [
            "consultation",
            "literature_review",
        ],
        MajorWorkflow.START_PROJECT: [
            "consultation",
            "literature_review",
            "planning_committee",
            "execute_analysis",
            "writeup_results",
            "critical_review",
        ],
        MajorWorkflow.WORK_ON_EXISTING: [
            "consultation",  # Understand what to continue
            "planning_committee",  # Plan next steps
            "execute_analysis",
            "writeup_results",
            "critical_review",
        ],
        MajorWorkflow.EXPLORE_DATASET: [
            "consultation",
            "execute_analysis",  # EDA focus
            "writeup_results",
        ],
    }
    
    def __init__(
        self,
        llm_backend: Optional[AnthropicBackend] = None,
        user_callback: Optional[Callable[[str], str]] = None,
        transcripts_dir: Optional[Path] = None,
    ):
        """
        Initialize the Bohr orchestrator.
        
        Args:
            llm_backend: LLM backend for agent communication
            user_callback: Function to get user input (for non-interactive)
            transcripts_dir: Directory for saving transcripts
        """
        self.llm_backend = llm_backend or AnthropicBackend(model="claude-sonnet-4-5")
        self.user_callback = user_callback or self._default_user_input
        
        # Set up transcript logger
        self.transcripts_dir = transcripts_dir or Path(__file__).parent.parent.parent / "Transcripts"
        self.transcript = TranscriptLogger(self.transcripts_dir)
        
        self.context_manager: Optional[ContextManager] = None
        self.tool_factory = None  # Set during session start
        self.agents: dict[str, Agent] = {}
        self.session: Optional[MiniLabSession] = None
        
        self._interrupt_requested = False
        self._exit_requested = False
        self._token_budget: Optional[int] = None
        self._tokens_used: int = 0
    
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
        
        # Have Bohr generate and confirm project name
        if not project_name:
            project_name = await self._generate_project_name_interactive(user_request)
        
        # Create project directory
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_path = self.SANDBOX_ROOT / project_name
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize session
        self.session = MiniLabSession(
            session_id=session_id,
            project_name=project_name,
            project_path=project_path,
        )
        self.session.context["user_request"] = user_request
        
        # Start transcript logging
        self.transcript.start_session(project_name)
        self.transcript.log_user_message(user_request)
        
        # Initialize agents with context manager and tool factory
        # workspace_root should be the parent of Sandbox, not the project path
        workspace_root = self.SANDBOX_ROOT.parent
        self.agents, self.context_manager, self.tool_factory = create_agents(
            workspace_root=workspace_root,
            input_callback=self._user_input_callback,
            permission_callback=None,
        )
        
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
        self.agents, self.context_manager, self.tool_factory = create_agents(
            workspace_root=workspace_root,
            input_callback=self._user_input_callback,
            permission_callback=None,
        )
        
        self._log(f"Session resumed: {self.session.session_id}")
        
        return self.session
    
    async def run(
        self,
        major_workflow: Optional[MajorWorkflow] = None,
    ) -> dict[str, Any]:
        """
        Run the orchestration loop.
        
        If major_workflow is not specified, Bohr will determine
        the appropriate workflow based on user interaction.
        
        Args:
            major_workflow: Optional explicit workflow to run
            
        Returns:
            Final results dictionary
        """
        from ..utils import console, Spinner, Style, StatusIcon
        
        if not self.session:
            raise RuntimeError("No active session. Call start_session first.")
        
        try:
            # Determine workflow if not specified
            if not major_workflow:
                spinner = Spinner("Analyzing request...")
                spinner.start()
                major_workflow = await self._determine_workflow()
                spinner.stop(f"Selected workflow: {major_workflow.value}")
            
            self._log(f"Selected workflow: {major_workflow.value}", console_print=False)
            
            # Get workflow sequence
            workflow_sequence = self.WORKFLOW_SEQUENCES[major_workflow]
            
            # Narrative-style introduction instead of system message
            workflow_intros = {
                MajorWorkflow.START_PROJECT: "Sounds like we're starting a new project! I'll guide us through the full research pipeline - from understanding your goals through analysis and writeup.",
                MajorWorkflow.LITERATURE_REVIEW: "I'll help you explore the literature on this topic. Let's see what's out there and build a solid foundation.",
                MajorWorkflow.BRAINSTORMING: "Let's brainstorm together! I'll help you think through the possibilities and form a concrete plan.",
                MajorWorkflow.EXPLORE_DATASET: "Let's dive into your data and see what stories it has to tell. I love a good exploration!",
                MajorWorkflow.WORK_ON_EXISTING: "Picking up where we left off - let me review what we've done and figure out the best next steps.",
            }
            intro = workflow_intros.get(major_workflow, f"Let's work through this together.")
            console.agent_message("BOHR", intro)
            
            # Execute workflow sequence
            results = {}
            for i, workflow_name in enumerate(workflow_sequence, 1):
                if self._interrupt_requested or self._exit_requested:
                    if self._exit_requested:
                        console.info("Exiting as requested")
                        self._log("User requested exit", console_print=False)
                    else:
                        console.warning("Interrupt requested, pausing execution")
                        self._log("Interrupt requested, pausing execution", console_print=False)
                    break
                
                # Check token budget before starting new workflow
                if self._token_budget:
                    current_usage = self.llm_backend.token_usage.get("total_tokens", 0)
                    budget_used_pct = (current_usage / self._token_budget) * 100
                    remaining_workflows = len(workflow_sequence) - i + 1
                    remaining_budget = self._token_budget - current_usage
                    budget_per_workflow = remaining_budget / max(1, remaining_workflows)
                    
                    if current_usage >= self._token_budget:
                        console.warning(f"We've used our token budget ({current_usage:,}/{self._token_budget:,}).")
                        console.agent_message("BOHR", "I'll wrap up with what we have. Here's a summary of our work so far...")
                        break
                    elif budget_used_pct >= 85:
                        # Dynamically decide: skip to writeup if we're running low
                        console.warning(f"Budget at {budget_used_pct:.0f}% - I'll prioritize the most important remaining work.")
                        remaining = workflow_sequence[i-1:]
                        if "writeup_results" in remaining and workflow_name not in ["writeup_results", "critical_review"]:
                            console.agent_message("BOHR", "Given our budget, let me skip ahead to summarizing our findings.")
                            workflow_name = "writeup_results"
                    elif budget_used_pct >= 60:
                        # Inform but continue - allocate remaining budget dynamically
                        console.info(f"Budget update: {budget_used_pct:.0f}% used ({current_usage:,}/{self._token_budget:,}). ~{budget_per_workflow:,.0f} tokens per remaining phase.")
                
                # Show progress with spinner
                phase_msg = f"Phase {i}/{len(workflow_sequence)}: {workflow_name.replace('_', ' ').title()}"
                console.info(phase_msg)
                spinner = Spinner(f"Running {workflow_name.replace('_', ' ').title()}")
                spinner.start()
                
                # Log stage transition to transcript
                self.transcript.log_stage_transition(
                    workflow_name,
                    f"Starting {workflow_name.replace('_', ' ').title()} (Phase {i}/{len(workflow_sequence)})"
                )
                
                self._log(f"Starting workflow: {workflow_name}", console_print=False)
                self.session.current_workflow = workflow_name
                self.session.save()
                
                result = await self._execute_workflow(workflow_name)
                results[workflow_name] = result
                self.session.workflow_results[workflow_name] = result
                
                if result.status == WorkflowStatus.COMPLETED:
                    self.session.completed_workflows.append(workflow_name)
                    spinner.stop(f"{workflow_name} completed")
                    self._log(f"Completed: {workflow_name}", console_print=False)
                    # Log completion to transcript
                    self.transcript.log_system_event(
                        "workflow_complete",
                        f"{workflow_name} completed successfully",
                        {"summary": result.summary} if result.summary else None
                    )
                elif result.status == WorkflowStatus.FAILED:
                    spinner.stop_error(f"{workflow_name} failed: {result.error}")
                    self._log(f"Failed: {workflow_name} - {result.error}", console_print=False)
                    # Log failure to transcript
                    self.transcript.log_system_event(
                        "workflow_failed",
                        f"{workflow_name} failed",
                        {"error": result.error}
                    )
                    # Ask user how to proceed
                    proceed = await self._handle_workflow_failure(workflow_name, result)
                    if not proceed:
                        break
                else:
                    spinner.stop(f"{workflow_name} finished with status: {result.status.value}")
                
                # Pass outputs to next workflow's context
                if result.outputs:
                    self.session.context.update(result.outputs)
                    # Extract token budget from consultation if present
                    if workflow_name == "consultation" and result.outputs.get("token_budget"):
                        self._token_budget = result.outputs["token_budget"]
                        # Post-consultation summary
                        self._show_post_consultation_summary(result)
                
                self.session.save()
            
            # Final summary
            console.info("Generating final summary...")
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
            # Try to save transcript even on error
            try:
                self.transcript.save_transcript()
            except Exception:
                pass
            raise
    
    async def _determine_workflow(self) -> MajorWorkflow:
        """
        Use Bohr to determine the appropriate workflow via JSON response.
        
        Returns:
            Selected MajorWorkflow
        """
        import json as json_module
        
        user_request = self.session.context.get("user_request", "")
        
        # Use LLM to classify with structured JSON response
        try:
            messages = [
                {"role": "system", "content": """You are Bohr, classifying a research request into the appropriate workflow.

RESPOND WITH ONLY A JSON OBJECT:
{"workflow": "WORKFLOW_NAME", "confidence": 0.0-1.0, "reasoning": "brief explanation"}

Valid WORKFLOW_NAME values:
- BRAINSTORMING: For exploring ideas, hypothesis generation, open-ended discussion
- LITERATURE_REVIEW: For researching background, finding citations, understanding field
- START_PROJECT: For new complete analysis projects with data
- WORK_ON_EXISTING: For continuing or iterating on an existing project
- EXPLORE_DATASET: For data exploration, EDA, understanding what's in data"""},
                {"role": "user", "content": f"Classify this request:\n\n{user_request}"}
            ]
            
            response = await self.llm_backend.acomplete(messages)
            
            # Parse JSON response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            data = json_module.loads(response)
            workflow_str = data.get("workflow", "START_PROJECT").upper()
            
            # Map to enum
            workflow_map = {
                "BRAINSTORMING": MajorWorkflow.BRAINSTORMING,
                "LITERATURE_REVIEW": MajorWorkflow.LITERATURE_REVIEW,
                "START_PROJECT": MajorWorkflow.START_PROJECT,
                "WORK_ON_EXISTING": MajorWorkflow.WORK_ON_EXISTING,
                "EXPLORE_DATASET": MajorWorkflow.EXPLORE_DATASET,
            }
            
            return workflow_map.get(workflow_str, MajorWorkflow.START_PROJECT)
                    
        except Exception:
            # Fall back to START_PROJECT on any error
            return MajorWorkflow.START_PROJECT
    
    async def _execute_workflow(self, workflow_name: str) -> WorkflowResult:
        """
        Execute a specific workflow module.
        
        Args:
            workflow_name: Name of workflow to execute
            
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
        
        # Check for existing checkpoint
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
        
        # Save summary to file
        summary_path = self.session.project_path / "session_summary.md"
        with open(summary_path, "w") as f:
            f.write(f"# Session Summary: {self.session.project_name}\n\n")
            f.write(f"Session ID: {self.session.session_id}\n")
            f.write(f"Started: {self.session.started_at}\n\n")
            f.write(summary)
        
        return summary
    
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
                print(f"\n  Press Enter to confirm, or type a different project name:")
                user_input = input("\n  \033[1;32m▶\033[0m ").strip()
                
                if user_input:
                    return user_input.replace(" ", "_")
                return continue_project
            
            # New project
            console.agent_message("BOHR", f"I suggest we call this project: \033[1m{suggested_name}\033[0m")
            
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
            user_input = input("\n  \033[1;32m▶\033[0m ").strip()
            
            if user_input:
                project_name = user_input.replace(" ", "_")
            else:
                project_name = suggested_name
            
            # Confirm
            if project_name in existing_projects:
                console.agent_message("BOHR", f"Resuming project: {project_name}")
            else:
                console.agent_message("BOHR", f"Creating project: {project_name}")
            
            return project_name
            
        except Exception as e:
            # Fallback to simple timestamp-based name
            console.warning(f"Could not generate smart name: {e}")
            fallback = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            console.agent_message("BOHR", f"Using default project name: {fallback}")
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
        
        # Print token usage summary
        if hasattr(orchestrator.llm_backend, 'token_usage'):
            usage = orchestrator.llm_backend.token_usage
            if usage.get("total_tokens", 0) > 0:
                print()
                console.info(f"📊 Token Usage: {usage['input_tokens']:,} in + {usage['output_tokens']:,} out = {usage['total_tokens']:,} total")
                if usage.get("cache_read_tokens", 0) > 0:
                    console.info(f"   Cache: {usage['cache_read_tokens']:,} read, {usage.get('cache_creation_tokens', 0):,} created")
        
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
