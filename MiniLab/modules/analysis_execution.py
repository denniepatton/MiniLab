"""
ANALYSIS_EXECUTION Module (formerly ExecuteAnalysisModule).

Implementation phase where analysis is actually performed.
Primary loop: Dayhoff (data) -> Hinton (model) -> Bayes (validation)
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import json

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from .plan_dissemination import (
    format_task_graph_as_plan,
    extract_agent_responsibilities,
    build_agent_context,
)
from ..utils import console


@dataclass
class AnalysisStep:
    """A single step in the analysis execution."""
    step_number: int
    agent: str
    task: str
    status: str
    outputs: dict = field(default_factory=dict)
    code_files: list[str] = field(default_factory=list)
    data_files: list[str] = field(default_factory=list)
    error: Optional[str] = None


class AnalysisExecutionModule(Module):
    """
    ANALYSIS_EXECUTION: Implementation and computation phase.
    
    Purpose:
        - Execute the analysis plan step by step
        - Data processing and preparation (Dayhoff)
        - Model building and training (Hinton)
        - Statistical validation (Bayes)
        - Feature engineering (Shannon)
        - Biological interpretation hooks (Greider)
    
    Primary Loop: Dayhoff -> Hinton -> Bayes
    
    Outputs:
        - scripts/: Generated code
        - results/: Figures and tables
        - data/processed/: Processed data
        - artifacts/interpretation.md: Results interpretation
    """
    
    name = "analysis_execution"
    description = "Execute the analysis plan with iterative refinement"
    module_type = ModuleType.SUBGRAPH
    
    required_inputs = ["analysis_plan", "responsibilities", "data_paths"]
    optional_inputs = ["max_iterations", "quality_threshold", "code_style", "task_graph", "project_spec"]
    expected_outputs = ["analysis_results", "code_artifacts", "data_artifacts", "validation_report"]
    
    primary_agents = ["dayhoff", "hinton", "bayes"]
    supporting_agents = ["shannon", "greider", "feynman"]
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """Execute analysis module."""
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
            
            task_graph = inputs.get("task_graph")
            task_graph_plan = format_task_graph_as_plan(task_graph) if task_graph else ""
            
            responsibilities = inputs.get("responsibilities", {})
            if isinstance(responsibilities, str):
                responsibilities = extract_agent_responsibilities(responsibilities, task_graph)
            
            self._state = {
                "analysis_plan": inputs["analysis_plan"],
                "data_paths": inputs["data_paths"],
                "task_graph_plan": task_graph_plan,
                "responsibilities": responsibilities,
                "project_spec": inputs.get("project_spec", ""),
                "steps": [],
                "current_iteration": 0,
                "validation_scores": [],
                "code_files": [],
                "data_files": [],
                "results": {},
            }
        
        max_iterations = inputs.get("max_iterations", 3)
        quality_threshold = inputs.get("quality_threshold", 0.8)
        
        self._log_step("Starting analysis execution")
        
        try:
            # Step 1: Parse analysis plan
            if self._current_step <= 0:
                self._log_step("Step 1: Parsing analysis plan")
                
                parse_result = await self._run_agent_task(
                    agent_name="dayhoff",
                    task=f"""Parse this analysis plan into executable steps.

Analysis Plan:
{inputs['analysis_plan']}

Available Data:
{json.dumps(inputs['data_paths'], indent=2)}

For each step specify: step number, responsible agent, task, inputs, outputs.""",
                )
                
                self._state["parsed_plan"] = parse_result.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Data preparation (Dayhoff)
            if self._current_step <= 1:
                self._log_step("Step 2: Data preparation")
                
                data_task = f"""Prepare the data for analysis.

Data Paths:
{json.dumps(inputs['data_paths'], indent=2)}

Analysis Requirements:
{self._state['parsed_plan']}

Tasks:
1. Load and inspect the data files
2. Clean and preprocess as needed
3. Handle missing values
4. Create train/test splits if needed
5. Save processed data to data/processed/

OUTPUT LOCATIONS:
- Code: scripts/01_preprocess.py
- Data: data/processed/

Use tools to write code and execute it."""
                
                data_task_with_context = self._inject_plan_context("dayhoff", data_task)
                
                data_result = await self._run_agent_task(
                    agent_name="dayhoff",
                    task=data_task_with_context,
                )
                
                self._state["data_preparation"] = data_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Feature engineering (Shannon)
            if self._current_step <= 2:
                self._log_step("Step 3: Feature engineering")
                
                feature_result = await self._run_agent_task(
                    agent_name="shannon",
                    task=f"""Perform feature engineering and selection.

Prepared Data:
{self._state['data_preparation']}

Analysis Goals:
{inputs['analysis_plan'][:1000]}

Tasks:
1. Analyze information content
2. Identify redundant features
3. Create derived features if beneficial
4. Recommend feature selection strategy

OUTPUT: scripts/02_features.py""",
                )
                
                self._state["feature_engineering"] = feature_result.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Model development loop (Hinton + Bayes)
            if self._current_step <= 3:
                self._log_step("Step 4: Model development loop")
                
                iteration = self._state.get("current_iteration", 0)
                best_score = 0.0
                
                while iteration < max_iterations:
                    self._log_step(f"Model iteration {iteration + 1}/{max_iterations}")
                    
                    # Hinton: Build/improve model
                    model_result = await self._run_agent_task(
                        agent_name="hinton",
                        task=f"""{'Build initial model' if iteration == 0 else 'Improve model based on feedback'}.

Prepared Data:
{self._state['data_preparation']}

Features:
{self._state['feature_engineering']}

{'Previous Validation Feedback:' + self._state.get('last_validation', '') if iteration > 0 else ''}

Analysis Requirements:
{inputs['analysis_plan'][:1000]}

OUTPUT: scripts/03_model.py

Use code_editor to write and execute model code.""",
                    )
                    
                    self._state["model_development"] = model_result.get("response", "")
                    
                    # Bayes: Validate model
                    validation_result = await self._run_agent_task(
                        agent_name="bayes",
                        task=f"""Validate the model and provide feedback.

Model Development:
{self._state['model_development']}

Data Context:
{self._state['data_preparation'][:500]}

Tasks:
1. Assess statistical validity
2. Check for overfitting
3. Evaluate uncertainty quantification
4. Verify assumptions

OUTPUT: scripts/04_validate.py

Provide a validation score (0-1) and specific feedback.
Start your response with "VALIDATION_SCORE: X.XX".""",
                    )
                    
                    validation_response = validation_result.get("response", "")
                    self._state["last_validation"] = validation_response
                    
                    try:
                        score_line = validation_response.split("\n")[0]
                        if "VALIDATION_SCORE:" in score_line:
                            score = float(score_line.split(":")[1].strip())
                        else:
                            score = 0.5
                    except (ValueError, IndexError):
                        score = 0.5
                    
                    self._state["validation_scores"].append({
                        "iteration": iteration,
                        "score": score,
                        "feedback": validation_response,
                    })
                    
                    best_score = max(best_score, score)
                    
                    if score >= quality_threshold:
                        self._log_step(f"Quality threshold met: {score:.2f}")
                        break
                    
                    iteration += 1
                    self._state["current_iteration"] = iteration
                    self.save_checkpoint()
                
                self._current_step = 4
                self.save_checkpoint()
            
            # Step 5: Biological interpretation (Greider)
            if self._current_step <= 4:
                self._log_step("Step 5: Biological interpretation")
                
                bio_result = await self._run_agent_task(
                    agent_name="greider",
                    task=f"""Provide biological interpretation of the analysis results.

Model Results:
{self._state['model_development']}

Validation:
{self._state.get('last_validation', '')}

Project Context:
{inputs.get('project_spec', '')[:1000]}

Interpret the biological significance and implications.""",
                )
                
                self._state["biological_interpretation"] = bio_result.get("response", "")
                self._current_step = 5
                self.save_checkpoint()
            
            # Step 6: Final summary
            if self._current_step <= 5:
                self._log_step("Step 6: Final summary")
                
                summary_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Summarize the analysis execution results.

Data Preparation:
{self._state['data_preparation'][:500]}

Feature Engineering:
{self._state['feature_engineering'][:500]}

Model Development:
{self._state['model_development'][:500]}

Validation:
{self._state.get('last_validation', '')[:500]}

Biological Interpretation:
{self._state.get('biological_interpretation', '')[:500]}

Create a comprehensive summary of what was accomplished.""",
                )
                
                self._state["execution_summary"] = summary_result.get("response", "")
                self._current_step = 6
            
            # Write outputs
            self._write_outputs()
            
            self._status = ModuleStatus.COMPLETED
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "analysis_results": self._state.get("execution_summary", ""),
                    "code_artifacts": self._state.get("code_files", []),
                    "data_artifacts": self._state.get("data_files", []),
                    "validation_report": self._state.get("last_validation", ""),
                },
                artifacts=[
                    str(self.project_path / "artifacts" / "interpretation.md"),
                ],
                summary="Analysis execution complete.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
    
    def _inject_plan_context(self, agent_name: str, task: str) -> str:
        """Inject plan context into task."""
        task_graph_plan = self._state.get("task_graph_plan", "")
        responsibilities = self._state.get("responsibilities", {})
        project_spec = self._state.get("project_spec", "")
        
        if task_graph_plan or responsibilities:
            context = build_agent_context(
                agent_name,
                task_graph_plan,
                responsibilities,
                project_spec,
            )
            return f"{context}\n\n{task}"
        return task
    
    def _write_outputs(self) -> None:
        """Write outputs to appropriate directories."""
        artifacts_dir = self.project_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        with open(artifacts_dir / "interpretation.md", "w") as f:
            f.write("# Analysis Results Interpretation\n\n")
            f.write("## Execution Summary\n\n")
            f.write(self._state.get("execution_summary", "") + "\n\n")
            f.write("## Biological Interpretation\n\n")
            f.write(self._state.get("biological_interpretation", "") + "\n\n")
            f.write("## Validation Summary\n\n")
            f.write(self._state.get("last_validation", "") + "\n")


# Backward compatibility alias
ExecuteAnalysisModule = AnalysisExecutionModule
