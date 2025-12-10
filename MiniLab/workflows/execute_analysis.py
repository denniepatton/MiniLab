"""
EXECUTE ANALYSIS Workflow Module.

Implementation phase where analysis is actually performed.
Primary loop: Dayhoff (data) -> Hinton (model) -> Bayes (validation)
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import json

from .base import WorkflowModule, WorkflowResult, WorkflowCheckpoint, WorkflowStatus
from ..utils import console


@dataclass
class AnalysisStep:
    """A single step in the analysis execution."""
    step_number: int
    agent: str
    task: str
    status: str  # pending, running, completed, failed
    outputs: dict = field(default_factory=dict)
    code_files: list[str] = field(default_factory=list)
    data_files: list[str] = field(default_factory=list)
    error: Optional[str] = None


class ExecuteAnalysisModule(WorkflowModule):
    """
    EXECUTE ANALYSIS: Implementation and computation phase.
    
    Purpose:
        - Execute the analysis plan step by step
        - Data processing and preparation (Dayhoff)
        - Model building and training (Hinton)
        - Statistical validation (Bayes)
        - Feature engineering (Shannon)
        - Biological interpretation hooks (Greider)
    
    Primary Loop: Dayhoff -> Hinton -> Bayes
        - Dayhoff prepares data for each stage
        - Hinton builds/trains models
        - Bayes validates and provides feedback
        - Loop until quality criteria met
    
    Supporting:
        - Shannon: Feature selection/engineering
        - Greider: Biological context for interpretation
        - Feynman: Technical debugging/optimization
    
    Outputs:
        - analysis_results: Key findings and metrics
        - code_artifacts: Python scripts generated
        - data_artifacts: Processed data files
        - validation_report: Statistical validation
    """
    
    name = "execute_analysis"
    description = "Execute the analysis plan with iterative refinement"
    
    required_inputs = ["analysis_plan", "responsibilities", "data_paths"]
    optional_inputs = ["max_iterations", "quality_threshold", "code_style"]
    expected_outputs = ["analysis_results", "code_artifacts", "data_artifacts", "validation_report"]
    
    primary_agents = ["dayhoff", "hinton", "bayes"]
    supporting_agents = ["shannon", "greider", "feynman"]
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[WorkflowCheckpoint] = None,
    ) -> WorkflowResult:
        """
        Execute analysis workflow.
        
        Steps:
        1. Parse analysis plan into executable steps
        2. Data preparation phase (Dayhoff)
        3. Feature engineering if needed (Shannon)
        4. Model development loop (Hinton + Bayes)
        5. Biological interpretation (Greider)
        6. Final validation (Bayes)
        """
        # Validate inputs
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        # Restore or initialize state
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = WorkflowStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "analysis_plan": inputs["analysis_plan"],
                "data_paths": inputs["data_paths"],
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
            # Step 1: Parse analysis plan into steps
            if self._current_step <= 0:
                self._log_step("Step 1: Parsing analysis plan")
                
                parse_result = await self._run_agent_task(
                    agent_name="dayhoff",
                    task=f"""Parse this analysis plan into concrete executable steps.

Analysis Plan:
{inputs['analysis_plan']}

Available Data:
{json.dumps(inputs['data_paths'], indent=2)}

For each step, specify:
1. Step number
2. Responsible agent (dayhoff/hinton/bayes/shannon/greider)
3. Concrete task description
4. Input requirements
5. Expected outputs

Format as a numbered list of steps.""",
                )
                
                self._state["parsed_plan"] = parse_result.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Data preparation (Dayhoff)
            if self._current_step <= 1:
                self._log_step("Step 2: Data preparation")
                
                data_result = await self._run_agent_task(
                    agent_name="dayhoff",
                    task=f"""Prepare the data for analysis.

Data Paths:
{json.dumps(inputs['data_paths'], indent=2)}

Analysis Requirements:
{self._state['parsed_plan']}

Tasks:
1. Load and inspect the data files
2. Clean and preprocess as needed
3. Handle missing values appropriately
4. Create train/test splits if needed
5. Save processed data to Sandbox/

Use the filesystem and code_editor tools to:
- Read the data files
- Write Python preprocessing scripts
- Execute and save processed data

Report what data is ready and its structure.""",
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
1. Analyze information content of features
2. Identify redundant or low-value features
3. Create derived features if beneficial
4. Recommend feature selection strategy
5. Implement feature transformations

Write code to implement feature engineering.
Report the final feature set and rationale.""",
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

Tasks:
1. {'Design model architecture' if iteration == 0 else 'Modify model based on feedback'}
2. Implement training code
3. Train and evaluate on validation set
4. Report performance metrics

Use code_editor to write model code.
Focus on {'establishing baseline' if iteration == 0 else 'addressing validation concerns'}.""",
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
1. Assess statistical validity of approach
2. Check for overfitting/underfitting
3. Evaluate uncertainty quantification
4. Verify assumptions are met
5. Suggest improvements if needed

Provide a validation score (0-1) and specific feedback.
Start your response with "VALIDATION_SCORE: X.XX" on the first line.""",
                    )
                    
                    validation_response = validation_result.get("response", "")
                    self._state["last_validation"] = validation_response
                    
                    # Parse validation score
                    try:
                        score_line = validation_response.split("\n")[0]
                        if "VALIDATION_SCORE:" in score_line:
                            score = float(score_line.split(":")[1].strip())
                        else:
                            score = 0.5  # Default if not found
                    except (ValueError, IndexError):
                        score = 0.5
                    
                    self._state["validation_scores"].append({
                        "iteration": iteration,
                        "score": score,
                        "feedback": validation_response,
                    })
                    
                    best_score = max(best_score, score)
                    
                    if score >= quality_threshold:
                        self._log_step(f"Quality threshold met: {score:.2f} >= {quality_threshold}")
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

Analysis Context:
{inputs['analysis_plan'][:1000]}

Tasks:
1. Interpret key findings in biological context
2. Identify relevant pathways/mechanisms
3. Compare with known biology
4. Suggest biological validation experiments
5. Note any surprising or contradictory findings

Provide interpretations that would be meaningful to domain experts.""",
                )
                
                self._state["biological_interpretation"] = bio_result.get("response", "")
                self._current_step = 5
                self.save_checkpoint()
            
            # Step 6: Final validation report (Bayes)
            if self._current_step <= 5:
                self._log_step("Step 6: Final validation report")
                
                final_validation = await self._run_agent_task(
                    agent_name="bayes",
                    task=f"""Create final validation report for the analysis.

All Validation Results:
{json.dumps(self._state['validation_scores'], indent=2)}

Model Development:
{self._state['model_development']}

Biological Context:
{self._state['biological_interpretation']}

Create a comprehensive validation report including:
1. Summary of validation iterations
2. Final performance metrics with confidence intervals
3. Statistical tests performed
4. Assumptions and their validity
5. Limitations and caveats
6. Recommendations for interpretation

This report will accompany the analysis results.""",
                )
                
                self._state["validation_report"] = final_validation.get("response", "")
                self._current_step = 6
            
            # Compile results
            results = {
                "model_summary": self._state.get("model_development", ""),
                "validation_summary": self._state.get("validation_report", ""),
                "biological_interpretation": self._state.get("biological_interpretation", ""),
                "iterations": len(self._state.get("validation_scores", [])),
                "final_score": self._state["validation_scores"][-1]["score"] if self._state.get("validation_scores") else 0,
            }
            
            # Write outputs to files
            output_dir = self.project_path / "results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Analysis results
            results_path = output_dir / "analysis_results.md"
            with open(results_path, "w") as f:
                f.write("# Analysis Results\n\n")
                f.write("## Model Summary\n\n")
                f.write(self._state.get("model_development", ""))
                f.write("\n\n## Biological Interpretation\n\n")
                f.write(self._state.get("biological_interpretation", ""))
            
            # Validation report
            validation_path = output_dir / "validation_report.md"
            with open(validation_path, "w") as f:
                f.write("# Validation Report\n\n")
                f.write(self._state.get("validation_report", ""))
            
            # Iteration history
            history_path = output_dir / "iteration_history.json"
            with open(history_path, "w") as f:
                json.dump(self._state.get("validation_scores", []), f, indent=2)
            
            self._status = WorkflowStatus.COMPLETED
            self._log_step("Analysis execution completed")
            
            return WorkflowResult(
                status=WorkflowStatus.COMPLETED,
                outputs={
                    "analysis_results": results,
                    "code_artifacts": self._state.get("code_files", []),
                    "data_artifacts": self._state.get("data_files", []),
                    "validation_report": self._state.get("validation_report", ""),
                },
                artifacts=[str(results_path), str(validation_path), str(history_path)],
                summary=f"Analysis complete after {results['iterations']} iterations. Final score: {results['final_score']:.2f}",
            )
            
        except Exception as e:
            self._status = WorkflowStatus.FAILED
            self._log_step(f"Error: {str(e)}")
            self.save_checkpoint()
            
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=str(e),
                outputs=self._state,
            )
