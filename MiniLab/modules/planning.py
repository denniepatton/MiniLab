"""
PLANNING Module.

Subgraph module for full plan production after team discussion.
Produces artifacts/plan.md, DAG, directory structure, and acceptance checks.
"""

from typing import Any, Optional
from pathlib import Path
import json

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..core.task_graph import TaskGraph
from ..utils import console


class PlanningModule(Module):
    """
    PLANNING: Full plan production subgraph.
    
    Goal: Produce artifacts/plan.md + DAG + directory structure + acceptance checks.
    
    Subgraph pattern:
    1. Outline phases and dependencies
    2. Define deliverables and acceptance checks
    3. Estimate token budget (with re-allocation policy)
    4. Write plan + decisions artifacts
    5. Initialize DAG and skeleton
    
    Primary Agent: Bohr
    
    Outputs:
        - artifacts/plan.md: Complete project plan
        - artifacts/acceptance_checks.md: What "done" means
        - planning/task_dag.json: Task graph
        - planning/task_dag.dot: DOT visualization
        - Directory skeleton
    """
    
    name = "planning"
    description = "Full plan production with DAG and acceptance checks"
    module_type = ModuleType.SUBGRAPH
    
    required_inputs = ["project_spec", "analysis_plan"]
    optional_inputs = ["risks_unknowns", "token_budget", "responsibilities"]
    expected_outputs = ["plan", "task_graph", "acceptance_checks", "directory_skeleton"]
    
    primary_agents = ["bohr"]
    supporting_agents = []
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """Execute planning module."""
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = ModuleStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "project_spec": inputs["project_spec"],
                "analysis_plan": inputs["analysis_plan"],
                "risks_unknowns": inputs.get("risks_unknowns", []),
                "token_budget": inputs.get("token_budget", 300000),
                "responsibilities": inputs.get("responsibilities", ""),
            }
        
        self._log_step("Starting planning module")
        
        try:
            # Step 1: Outline phases and dependencies
            if self._current_step <= 0:
                self._log_step("Step 1: Outlining phases and dependencies")
                
                outline_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Create a detailed phase outline with dependencies.

Project Specification:
{inputs['project_spec'][:2000]}

Analysis Plan:
{inputs['analysis_plan'][:2000]}

Known Risks:
{json.dumps(inputs.get('risks_unknowns', []), indent=2)}

Create numbered phases mapped to DAG task nodes:
1. Phase name, description, owner agent
2. Dependencies (which phases must complete first)
3. Estimated effort (token estimate)

Be explicit about the order and dependencies.""",
                )
                
                self._state["phases_outline"] = outline_result.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Define acceptance checks
            if self._current_step <= 1:
                self._log_step("Step 2: Defining acceptance checks")
                
                checks_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Define acceptance checks for each deliverable.

Phases Outline:
{self._state['phases_outline']}

Project Specification:
{inputs['project_spec'][:1000]}

For each expected output, define:
1. What "done" means for this deliverable
2. Verification criteria (how to check correctness)
3. Quality requirements
4. File location and format

Be specific and measurable.""",
                )
                
                self._state["acceptance_checks"] = checks_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Create structured plan
            if self._current_step <= 2:
                self._log_step("Step 3: Creating structured plan")
                
                plan_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Create the complete project plan document.

Phases:
{self._state['phases_outline']}

Acceptance Checks:
{self._state['acceptance_checks']}

Responsibilities:
{inputs.get('responsibilities', 'To be assigned based on phases')}

Token Budget: {self._state['token_budget']:,}

Create a comprehensive plan including:
1. Executive summary
2. Numbered phases with task details
3. Delegation (who does what, and why)
4. Token budget per phase with re-allocation policy
5. Exact final outputs (filenames, formats)
6. Input data summaries
7. Acceptance checks per deliverable""",
                )
                
                self._state["complete_plan"] = plan_result.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Generate TaskGraph
            if self._current_step <= 3:
                self._log_step("Step 4: Generating TaskGraph")
                
                graph_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Generate a TaskGraph JSON from the plan.

Complete Plan:
{self._state['complete_plan'][:3000]}

Create JSON:
```json
{{
  "tasks": [
    {{
      "id": "phase_id",
      "name": "Phase Name",
      "description": "What this accomplishes",
      "owner": "agent_id",
      "dependencies": ["previous_phase_id"],
      "estimated_tokens": 50000
    }}
  ]
}}
```

Ensure dependencies form a valid DAG (no cycles).""",
                )
                
                # Parse TaskGraph
                response = graph_result.get("response", "")
                try:
                    import re
                    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                    if json_match:
                        graph_data = json.loads(json_match.group(1))
                    else:
                        json_match = re.search(r'\{[^{}]*"tasks".*\}', response, re.DOTALL)
                        if json_match:
                            graph_data = json.loads(json_match.group(0))
                        else:
                            graph_data = {"tasks": []}
                    
                    task_graph = TaskGraph.from_bohr_plan(graph_data, self.project_path.name)
                    self._state["task_graph"] = task_graph.to_dict()
                except (json.JSONDecodeError, Exception) as e:
                    self._log_step(f"TaskGraph parsing warning: {e}")
                    task_graph = TaskGraph(self.project_path.name)
                    self._state["task_graph"] = task_graph.to_dict()
                
                self._current_step = 4
                self.save_checkpoint()
            
            # Step 5: Initialize directory skeleton
            if self._current_step <= 4:
                self._log_step("Step 5: Initializing directory skeleton")
                
                self._create_directory_skeleton()
                self._current_step = 5
            
            # Write outputs
            self._write_outputs()
            
            self._status = ModuleStatus.COMPLETED
            
            task_graph = TaskGraph.from_dict(self._state["task_graph"])
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "plan": self._state.get("complete_plan", ""),
                    "task_graph": task_graph,
                    "acceptance_checks": self._state.get("acceptance_checks", ""),
                    "directory_skeleton": True,
                },
                artifacts=[
                    str(self.project_path / "artifacts" / "plan.md"),
                    str(self.project_path / "artifacts" / "acceptance_checks.md"),
                    str(self.project_path / "planning" / "task_dag.json"),
                ],
                summary="Planning complete. Project structure initialized.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
    
    def _create_directory_skeleton(self) -> None:
        """Create the standard project directory structure."""
        directories = [
            "artifacts",
            "planning",
            "transcripts",
            "logs",
            "data/raw",
            "data/interim",
            "data/processed",
            "scripts",
            "results/figures",
            "results/tables",
            "reports",
            "env",
            "eval",
            "memory/notes",
            "memory/sources",
            "memory/index",
            "cache",
        ]
        
        for d in directories:
            (self.project_path / d).mkdir(parents=True, exist_ok=True)
    
    def _write_outputs(self) -> None:
        """Write planning outputs."""
        artifacts_dir = self.project_path / "artifacts"
        planning_dir = self.project_path / "planning"
        
        for d in [artifacts_dir, planning_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Write plan.md
        with open(artifacts_dir / "plan.md", "w") as f:
            f.write("# Project Plan\n\n")
            f.write(self._state.get("complete_plan", ""))
        
        # Write acceptance_checks.md
        with open(artifacts_dir / "acceptance_checks.md", "w") as f:
            f.write("# Acceptance Checks\n\n")
            f.write(self._state.get("acceptance_checks", ""))
        
        # Write TaskGraph
        task_graph = TaskGraph.from_dict(self._state.get("task_graph", {}))
        task_graph.save(planning_dir / "task_dag.json")
        
        # Write DOT visualization
        try:
            dot_content = task_graph.to_dot()
            with open(planning_dir / "task_dag.dot", "w") as f:
                f.write(dot_content)
            
            # Try to render PNG
            task_graph.render_png(
                planning_dir / "task_dag.dot",
                planning_dir / "task_dag.png"
            )
        except Exception as e:
            self._log_step(f"DOT rendering warning: {e}")
