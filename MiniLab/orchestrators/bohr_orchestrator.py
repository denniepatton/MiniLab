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
        self.llm_backend = llm_backend or AnthropicBackend(model="claude-sonnet-4-20250514")
        self.user_callback = user_callback or self._default_user_input
        
        # Set up transcript logger
        self.transcripts_dir = transcripts_dir or Path(__file__).parent.parent.parent / "Transcripts"
        self.transcript = TranscriptLogger(self.transcripts_dir)
        
        self.context_manager: Optional[ContextManager] = None
        self.tool_factory = None  # Set during session start
        self.agents: dict[str, Agent] = {}
        self.session: Optional[MiniLabSession] = None
        
        self._interrupt_requested = False
    
    def _default_user_input(self, prompt: str) -> str:
        """Default user input via stdin."""
        print(f"\n[BOHR]: {prompt}")
        return input("[YOU]: ").strip()
    
    def _user_input_callback(self, prompt: str, options: Optional[list[str]] = None) -> str:
        """Callback for tools that need user input."""
        if options:
            print(f"\n[BOHR]: {prompt}")
            for i, opt in enumerate(options, 1):
                print(f"  {i}. {opt}")
            return input("[YOU]: ").strip()
        return self.user_callback(prompt)
    
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
            console.agent_message("BOHR", f"Starting {major_workflow.value} workflow ({len(workflow_sequence)} phases)")
            
            # Execute workflow sequence
            results = {}
            for i, workflow_name in enumerate(workflow_sequence, 1):
                if self._interrupt_requested:
                    console.warning("Interrupt requested, pausing execution")
                    self._log("Interrupt requested, pausing execution", console_print=False)
                    break
                
                # Show progress with spinner
                phase_msg = f"Phase {i}/{len(workflow_sequence)}: {workflow_name.replace('_', ' ').title()}"
                console.info(phase_msg)
                spinner = Spinner(f"Running {workflow_name}...")
                spinner.start()
                
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
                elif result.status == WorkflowStatus.FAILED:
                    spinner.stop_error(f"{workflow_name} failed: {result.error}")
                    self._log(f"Failed: {workflow_name} - {result.error}", console_print=False)
                    # Ask user how to proceed
                    proceed = await self._handle_workflow_failure(workflow_name, result)
                    if not proceed:
                        break
                else:
                    spinner.stop(f"{workflow_name} finished with status: {result.status.value}")
                
                # Pass outputs to next workflow's context
                if result.outputs:
                    self.session.context.update(result.outputs)
                
                self.session.save()
            
            # Final summary
            console.info("Generating final summary...")
            final_summary = await self._create_final_summary()
            results["final_summary"] = final_summary
            
            self.session.current_workflow = None
            self.session.save()
            
            return results
            
        except Exception as e:
            self._log(f"Orchestration error: {str(e)}")
            if self.session:
                self.session.save()
            raise
    
    async def _determine_workflow(self) -> MajorWorkflow:
        """
        Use Bohr to determine the appropriate workflow.
        
        Returns:
            Selected MajorWorkflow
        """
        user_request = self.session.context.get("user_request", "")
        
        # Ask Bohr to classify the request
        bohr = self.agents.get("bohr")
        if not bohr:
            # Fallback classification without agent
            return self._simple_workflow_classification(user_request)
        
        classification_result = await bohr.execute_task(
            task=f"""Classify this user request into one of these workflow types:

User Request: {user_request}

Workflow Options:
1. BRAINSTORMING - User wants to explore ideas, discuss approaches, no concrete analysis yet
2. LITERATURE_REVIEW - User wants background research on a topic
3. START_PROJECT - User wants to conduct a complete new analysis
4. WORK_ON_EXISTING - User wants to continue or modify an existing project
5. EXPLORE_DATASET - User wants to understand a dataset through EDA

Respond with ONLY the workflow name (e.g., "START_PROJECT").
If unclear, ask a clarifying question using the user_input tool.""",
        )
        
        response = classification_result.get("response", "").upper().strip()
        
        # Map response to enum
        workflow_map = {
            "BRAINSTORMING": MajorWorkflow.BRAINSTORMING,
            "LITERATURE_REVIEW": MajorWorkflow.LITERATURE_REVIEW,
            "START_PROJECT": MajorWorkflow.START_PROJECT,
            "WORK_ON_EXISTING": MajorWorkflow.WORK_ON_EXISTING,
            "EXPLORE_DATASET": MajorWorkflow.EXPLORE_DATASET,
        }
        
        for key, value in workflow_map.items():
            if key in response:
                return value
        
        # Default to start_project
        return MajorWorkflow.START_PROJECT
    
    def _simple_workflow_classification(self, request: str) -> MajorWorkflow:
        """Simple keyword-based workflow classification."""
        request_lower = request.lower()
        
        if any(w in request_lower for w in ["brainstorm", "idea", "discuss", "think about"]):
            return MajorWorkflow.BRAINSTORMING
        elif any(w in request_lower for w in ["literature", "papers", "research", "review", "background"]):
            return MajorWorkflow.LITERATURE_REVIEW
        elif any(w in request_lower for w in ["continue", "existing", "resume", "previous"]):
            return MajorWorkflow.WORK_ON_EXISTING
        elif any(w in request_lower for w in ["explore", "eda", "understand data", "look at"]):
            return MajorWorkflow.EXPLORE_DATASET
        else:
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
        Have Bohr generate a project name and confirm with user.
        
        Checks for existing projects that might be relevant.
        """
        from ..utils import console
        
        # Generate intelligent project name
        suggested = self._generate_smart_project_name(request)
        
        # Check for existing projects with similar themes
        existing_projects = []
        related_projects = []
        if self.SANDBOX_ROOT.exists():
            existing_projects = [
                d.name for d in self.SANDBOX_ROOT.iterdir()
                if d.is_dir() and (d / "session.json").exists()
            ]
            
            # Only show truly related projects (same dataset/topic, not random word matches)
            key_identifiers = self._extract_key_identifiers(request)
            if key_identifiers:
                related_projects = [
                    p for p in existing_projects 
                    if any(ident.lower() in p.lower() for ident in key_identifiers)
                ]
        
        # Build prompt for user
        console.agent_message("BOHR", f"I suggest we call this project: \033[1m{suggested}\033[0m")
        
        if related_projects:
            print(f"\n  Related existing projects you may want to continue:")
            for p in related_projects[:3]:
                print(f"    • {p}")
        
        print(f"\n  Press Enter to accept, or type a different name:")
        
        user_input = input("\n  \033[1;32m▶\033[0m ").strip()
        
        if user_input:
            project_name = user_input.replace(" ", "_")
        else:
            project_name = suggested
        
        # Confirm
        if project_name in existing_projects:
            console.agent_message("BOHR", f"Resuming project: {project_name}")
        else:
            console.agent_message("BOHR", f"Creating project: {project_name}")
        
        return project_name
    
    def _extract_key_identifiers(self, request: str) -> list[str]:
        """
        Extract key identifiers from request (dataset names, specific terms).
        
        These are proper nouns, paths, or domain-specific terms.
        """
        identifiers = []
        
        # Look for path references (ReadData/Something, Sandbox/Something)
        import re
        path_matches = re.findall(r'(?:ReadData|Sandbox)/(\w+)', request)
        identifiers.extend(path_matches)
        
        # Look for capitalized terms (likely proper nouns/dataset names)
        words = request.split()
        for word in words:
            # Clean punctuation
            clean = re.sub(r'[^\w]', '', word)
            # Proper nouns or acronyms
            if clean and (clean[0].isupper() or clean.isupper()) and len(clean) > 2:
                # Skip common sentence starters
                if clean.lower() not in {'the', 'this', 'that', 'please', 'note', 'use', 'let', 'help'}:
                    identifiers.append(clean)
        
        return list(set(identifiers))  # Dedupe
    
    def _generate_smart_project_name(self, request: str) -> str:
        """
        Generate an intelligent project name based on request content.
        
        Focuses on: dataset names, analysis type, key subject matter.
        """
        import re
        
        # Extract dataset/folder name from paths
        path_matches = re.findall(r'(?:ReadData|Sandbox)/(\w+)', request)
        dataset_name = path_matches[0] if path_matches else None
        
        # Detect analysis type
        request_lower = request.lower()
        analysis_type = None
        if any(term in request_lower for term in ['explor', 'eda', 'understand', 'look at', 'examine']):
            analysis_type = "exploration"
        elif any(term in request_lower for term in ['predict', 'model', 'classifier', 'regression']):
            analysis_type = "prediction"
        elif any(term in request_lower for term in ['review', 'literature', 'background']):
            analysis_type = "literature_review"
        elif any(term in request_lower for term in ['correlat', 'associat', 'relationship']):
            analysis_type = "association"
        elif any(term in request_lower for term in ['signature', 'biomarker', 'marker']):
            analysis_type = "biomarker"
        else:
            analysis_type = "analysis"
        
        # Build name
        if dataset_name:
            name = f"{dataset_name}_{analysis_type}"
        else:
            # Fall back to extracting key terms
            identifiers = self._extract_key_identifiers(request)
            if identifiers:
                name = f"{identifiers[0]}_{analysis_type}"
            else:
                name = analysis_type
        
        # Add date
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"{name}_{timestamp}"
    
    def interrupt(self) -> None:
        """Request graceful interruption of current workflow."""
        self._interrupt_requested = True
        self._log("Interrupt requested")
    
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
    workflow: Optional[str] = None,
) -> dict[str, Any]:
    """
    Convenience function to run MiniLab.
    
    Args:
        request: User's analysis request
        project_name: Optional project name
        workflow: Optional explicit workflow name
        
    Returns:
        Results dictionary
    """
    orchestrator = BohrOrchestrator()
    
    # Start session
    await orchestrator.start_session(
        user_request=request,
        project_name=project_name,
    )
    
    # Parse workflow if specified
    major_workflow = None
    if workflow:
        try:
            major_workflow = MajorWorkflow(workflow.lower())
        except ValueError:
            print(f"Unknown workflow: {workflow}, will auto-detect")
    
    # Run
    results = await orchestrator.run(major_workflow=major_workflow)
    
    return results
