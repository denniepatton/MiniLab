"""
CONSULTATION Module.

Initial phase for understanding user goals and generating a TaskGraph.
Bohr creates a DAG of tasks that defines the execution plan.

FAST Design: Minimal LLM calls, direct data exploration.
"""

from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
import csv
import json

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus
from ..core.task_graph import TaskGraph
from ..utils import console


@dataclass
class ConsultationOutput:
    """Structured output from consultation."""
    project_name: str
    task_graph: TaskGraph
    token_budget: int
    complexity: float


class ConsultationModule(Module):
    """
    CONSULTATION: The entry point for MiniLab.
    
    Goal: Converge quickly on project name, scope, outputs, acceptance 
    criteria, and budget.
    
    Produces a TaskGraph that defines the entire execution plan.
    Bohr analyzes the request and generates a DAG of tasks.
    
    Outputs:
        - artifacts/plan.md: Project plan
        - planning/task_dag.json: Task graph
        - project_specification.md: Initial spec
    """
    
    name = "consultation"
    description = "Initial user consultation to clarify goals and create execution plan"
    
    required_inputs = ["user_request"]
    optional_inputs = ["existing_project", "data_paths", "scope_confirmation", "scope_response"]
    expected_outputs = ["task_graph", "project_spec", "token_budget"]
    
    primary_agents = ["bohr"]
    supporting_agents = []
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """
        Execute consultation module.
        
        Steps:
        1. Quick direct data scan (NO agent loop)
        2. Bohr analyzes and proposes plan + TaskGraph
        3. User confirms/adjusts
        4. Return TaskGraph for orchestrator execution
        """
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        user_request = inputs["user_request"]
        
        # Initialize state
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = ModuleStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "user_request": user_request,
                "data_manifest": {},
            }
        
        self._log_step(f"Starting consultation: {user_request[:80]}...")
        
        try:
            # Step 1: Quick data scan (no LLM call)
            if self._current_step <= 0:
                data_paths = self._extract_data_paths(user_request)
                self._state["data_manifest"] = self._quick_data_scan(data_paths)
                self._current_step = 1
                self.save_checkpoint()

            # Check if Phase 1 understanding already happened (scope_confirmation set)
            scope_already_confirmed = inputs.get("scope_confirmation") is not None
            
            # Step 2: Bohr analyzes request and proposes plan
            if self._current_step <= 1:
                bohr = self.agents.get("bohr")
                if not bohr:
                    return ModuleResult(status=ModuleStatus.FAILED, error="Bohr agent unavailable")
                
                manifest_text = self._format_manifest(self._state["data_manifest"])
                
                analysis = await bohr.simple_query(
                    query=self._build_consultation_prompt(user_request, manifest_text),
                    context=f"Project: {self.project_path.name}"
                )
                
                self._state["analysis"] = analysis
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: User interaction (SKIP ONLY THE PROMPT if already confirmed)
            if self._current_step <= 2:
                if scope_already_confirmed:
                    self._state["user_response"] = inputs.get("scope_response", "User confirmed understanding")
                else:
                    # Need user interaction
                    from ..utils import Spinner
                    was_spinning = Spinner.pause_for_input()
                    
                    console.agent_message("BOHR", self._state["analysis"])
                    print()
                    try:
                        user_response = input("  \033[1;32m▶ Your response:\033[0m ").strip()
                    except (KeyboardInterrupt, EOFError):
                        user_response = ""
                    
                    if was_spinning:
                        Spinner.resume_after_input()
                    
                    self._state["user_response"] = user_response
                
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Bohr interprets user response and generates structured plan
            if self._current_step <= 3:
                bohr = self.agents.get("bohr")
                
                structured_plan = await bohr.simple_query(
                    query=self._build_interpretation_prompt(
                        user_request=self._state["user_request"],
                        bohr_analysis=self._state.get("analysis", ""),
                        user_response=self._state.get("user_response", ""),
                        data_manifest=self._state.get("data_manifest", {}),
                    ),
                    context="Interpret user response and generate execution plan"
                )
                
                plan_data = self._extract_json_from_response(structured_plan)
                
                # Validate project name matches session
                proposed_project = plan_data.get("project_name", "").strip()
                actual_project = self.project_path.name
                
                if proposed_project and proposed_project != actual_project:
                    raise ValueError(
                        f"Consultation proposed different project name '{proposed_project}' "
                        f"but approved project is '{actual_project}'."
                    )
                
                # Create TaskGraph from Bohr's interpretation
                task_graph = TaskGraph.from_bohr_plan(plan_data, self.project_path.name)
                self._state["task_graph"] = task_graph.to_dict()
                self._state["token_budget"] = plan_data.get("token_budget", 300_000)
                
                # Save TaskGraph to planning directory
                planning_dir = self.project_path / "planning"
                planning_dir.mkdir(parents=True, exist_ok=True)
                task_graph.save(planning_dir / "task_dag.json")
                
                self._current_step = 4
            
            # Write outputs
            self._write_outputs()
            
            self._status = ModuleStatus.COMPLETED
            
            # Reconstruct TaskGraph from state
            task_graph = TaskGraph.from_dict(self._state["task_graph"])
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "task_graph": task_graph,
                    "project_spec": self._state.get("analysis", ""),
                    "data_manifest": self._state["data_manifest"],
                    "token_budget": self._state.get("token_budget"),
                    "user_preferences": self._state.get("user_response", ""),
                    "complexity": self._estimate_complexity(user_request),
                },
                artifacts=[
                    str(self.project_path / "artifacts" / "plan.md"),
                    str(self.project_path / "planning" / "task_dag.json"),
                ],
                summary=f"Consultation complete. Created task graph with {task_graph.get_progress()['total_tasks']} tasks.",
            )
            
        except Exception as e:
            self._status = ModuleStatus.FAILED
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
    
    def _build_consultation_prompt(self, user_request: str, manifest_text: str) -> str:
        """Build the consultation prompt for Bohr."""
        return f"""Analyze this request and propose an optimal analysis plan.

## User Request
{user_request}

## Data Found
{manifest_text if manifest_text else "No data files detected."}

## Your Task
As the project orchestrator, propose what needs to be done. Be decisive and advisory.

### Format your response as:

### My Understanding
Summarize what the user wants in 2-3 sentences.

### Recommended Approach
Your expert recommendation for how to tackle this. Be specific about:
- What tasks need to be done and in what order
- Which agent should handle each task
- What the deliverables will be

### Recommended Budget
Suggest ONE specific token budget (as an integer) with brief reasoning. Example:
"I recommend 300000 tokens for this project. This allows for thorough analysis and complete documentation."

### Questions (only if truly necessary)
Ask only if there's genuine ambiguity you cannot resolve yourself. If the user said "use your best judgment" or similar, make all decisions yourself."""

    def _build_interpretation_prompt(
        self, 
        user_request: str, 
        bohr_analysis: str, 
        user_response: str,
        data_manifest: dict[str, Any],
    ) -> str:
        """Build prompt for Bohr to interpret user response and generate structured plan."""
        manifest_summary = self._format_manifest(data_manifest) if data_manifest else "No data files"
        
        return f"""You previously proposed an analysis plan and the user has responded. 
Your job is to interpret their response and create a final execution plan.

## Original User Request
{user_request}

## Your Previous Analysis
{bohr_analysis}

## User's Response
{user_response if user_response else "The user accepted your plan without modifications."}

## Available Data
{manifest_summary}

## Your Task
Interpret the user's response to understand:
1. What token budget they want (interpret naturally - "1 million", "1,000,000", "1M", "500k" all mean the same thing)
2. Any modifications to your proposed plan
3. Any preferences they expressed

Then create a complete task execution plan.

## CRITICAL: Return ONLY a JSON object with this exact structure:

```json
{{
  "token_budget": <integer - the token budget as a plain number>,
  "user_preferences": "<string summarizing what the user wants>",
  "tasks": [
    {{
      "id": "<snake_case_id>",
      "name": "<Human Readable Name>",
      "description": "<What this task accomplishes>",
      "owner": "<agent_id: bohr|gould|farber|feynman|shannon|greider|dayhoff|hinton|bayes>",
      "dependencies": ["<id of tasks that must complete first>"],
      "estimated_tokens": <integer estimate>
    }}
  ]
}}
```

## Agent Reference
- bohr: Project coordination, high-level decisions
- gould: Writing, documentation, literature review
- farber: Medical/clinical interpretation
- feynman: Physics, mathematical modeling
- shannon: Information theory, data analysis
- greider: Molecular biology, genomics
- dayhoff: Bioinformatics, sequence analysis
- hinton: Machine learning, statistical modeling
- bayes: Statistical analysis, Bayesian methods

## Guidelines
- Token budget should be the integer the user specified
- If user didn't specify budget, use your recommended budget
- Task estimated_tokens should sum to approximately the total budget
- Start with tasks that have no dependencies
- Include a final documentation/writeup task"""

    def _extract_json_from_response(self, response: str) -> dict[str, Any]:
        """Extract JSON from Bohr's response."""
        from ..utils import extract_json_from_text
        
        default = {
            "token_budget": 300_000,
            "user_preferences": "",
            "tasks": [
                {"id": "data_exploration", "name": "Data Exploration", "description": "Explore the data", "owner": "hinton", "dependencies": [], "estimated_tokens": 50000},
                {"id": "analysis", "name": "Core Analysis", "description": "Perform analysis", "owner": "hinton", "dependencies": ["data_exploration"], "estimated_tokens": 150000},
                {"id": "documentation", "name": "Documentation", "description": "Write up results", "owner": "gould", "dependencies": ["analysis"], "estimated_tokens": 100000},
            ]
        }
        
        return extract_json_from_text(response, fallback=default)

    def _estimate_complexity(self, user_request: str) -> float:
        """Simple complexity estimate."""
        length = len(user_request)
        if length < 100:
            return 0.3
        elif length < 300:
            return 0.5
        elif length < 600:
            return 0.7
        else:
            return 0.8

    def _extract_data_paths(self, request: str) -> list[Path]:
        """Extract data paths mentioned in request."""
        paths: list[Path] = []
        workspace_root = self.project_path.parent.parent
        
        for data_dir in ["ReadData", "Data", "data"]:
            data_path = workspace_root / data_dir
            if data_path.exists():
                if data_dir.lower() in request.lower() or "data" in request.lower():
                    paths.append(data_path)
        
        for part in request.split():
            clean_part = part.strip(".,;:'\"()")
            if "/" in clean_part:
                potential_path = workspace_root / clean_part
                if potential_path.exists():
                    paths.append(potential_path)
        
        return paths
    
    def _quick_data_scan(self, paths: list[Path]) -> dict[str, Any]:
        """Quickly scan data files directly."""
        manifest: dict[str, Any] = {"files": [], "summary": {}}
        
        for path in paths:
            items = [path] if path.is_file() else list(path.iterdir())
            for item in items:
                if item.is_file() and item.suffix in ['.csv', '.tsv', '.txt']:
                    manifest["files"].append(self._scan_file(item))
        
        manifest["summary"] = {
            "total_files": len(manifest["files"]),
            "total_rows": sum(f.get("rows", 0) for f in manifest["files"]),
        }
        return manifest
    
    def _scan_file(self, path: Path) -> dict[str, Any]:
        """Scan a single data file."""
        info: dict[str, Any] = {"name": path.name, "path": str(path)}
        try:
            with open(path, 'r') as f:
                delimiter = '\t' if path.suffix == '.tsv' else ','
                reader = csv.reader(f, delimiter=delimiter)
                header = next(reader, None)
                info["columns"] = len(header) if header else 0
                info["column_names"] = header[:15] if header else []
                info["rows"] = sum(1 for _ in reader)
        except Exception as e:
            info["error"] = str(e)
        return info
    
    def _format_manifest(self, manifest: dict[str, Any]) -> str:
        """Format manifest for display."""
        if not manifest.get("files"):
            return ""
        lines = [f"Found {len(manifest['files'])} file(s):"]
        for f in manifest["files"]:
            lines.append(f"• {f['name']}: {f.get('rows', '?')} rows, {f.get('columns', '?')} columns")
            if f.get("column_names"):
                cols = ", ".join(f["column_names"][:8])
                if len(f["column_names"]) > 8:
                    cols += f" (+{len(f['column_names'])-8} more)"
                lines.append(f"  Columns: {cols}")
        return "\n".join(lines)
    
    def _write_outputs(self) -> None:
        """Write consultation outputs to artifacts directory."""
        # Create artifacts directory structure
        artifacts_dir = self.project_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Write plan.md
        with open(artifacts_dir / "plan.md", "w") as f:
            f.write("# Project Plan\n\n")
            f.write(f"## Overview\n\n{self._state.get('analysis', '')}\n\n")
            f.write(f"## User Preferences\n\n{self._state.get('user_response', 'None specified')}\n\n")
            f.write(f"## Token Budget\n\n{self._state.get('token_budget', 300000):,} tokens\n")
        
        # Write project_specification.md for backward compatibility
        with open(self.project_path / "project_specification.md", "w") as f:
            f.write("# Project Specification\n\n")
            f.write(self._state.get("analysis", ""))
        
        # Write data manifest if there are data files
        data_manifest = self._state.get("data_manifest", {})
        if data_manifest.get("files"):
            with open(artifacts_dir / "data_manifest.md", "w") as f:
                f.write("# Data Manifest\n\n")
                f.write(self._format_manifest(data_manifest))
