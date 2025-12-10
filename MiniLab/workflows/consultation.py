"""
CONSULTATION Workflow Module.

Initial phase for understanding user goals, clarifying requirements,
and establishing project scope. Led by Bohr (orchestrator).
"""

from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path

from .base import WorkflowModule, WorkflowResult, WorkflowCheckpoint, WorkflowStatus
from ..utils import console


@dataclass
class ConsultationOutput:
    """Structured output from consultation."""
    project_name: str
    goals: list[str]
    constraints: list[str]
    data_sources: list[str]
    success_criteria: list[str]
    clarifications: dict[str, str]
    recommended_workflow: str


class ConsultationModule(WorkflowModule):
    """
    CONSULTATION: Initial user engagement and goal clarification.
    
    Purpose:
        - Understand what the user wants to accomplish
        - Clarify ambiguous requirements
        - Identify available data and resources
        - Establish success criteria
        - Recommend appropriate workflow path
    
    Primary Agent: Bohr (orchestrator, user-facing)
    Supporting: Feynman (technical feasibility)
    
    Outputs:
        - project_spec: Structured project specification
        - recommended_workflow: Suggested next workflow
        - clarifications: Q&A from consultation
    """
    
    name = "consultation"
    description = "Initial user consultation to clarify goals and scope"
    
    required_inputs = ["user_request"]
    optional_inputs = ["existing_project", "data_paths", "constraints"]
    expected_outputs = ["project_spec", "recommended_workflow", "clarifications"]
    
    primary_agents = ["bohr"]
    supporting_agents = ["feynman"]
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[WorkflowCheckpoint] = None,
    ) -> WorkflowResult:
        """
        Execute consultation workflow.
        
        Steps:
        1. Parse initial user request
        2. Ask clarifying questions (Bohr)
        3. Assess technical feasibility (Feynman)
        4. Synthesize into project specification
        5. Recommend workflow path
        """
        # Validate inputs
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        # Restore from checkpoint if provided
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = WorkflowStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "user_request": inputs["user_request"],
                "clarifications": {},
                "goals": [],
                "constraints": [],
            }
        
        self._log_step(f"Starting consultation for: {inputs['user_request'][:100]}...")
        
        try:
            # Step 1: Initial understanding (Bohr)
            if self._current_step <= 0:
                self._log_step("Step 1: Initial request analysis")
                
                initial_analysis = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Analyze this user request and identify:
1. What is the user trying to accomplish?
2. What data or resources do they have?
3. What are the implicit assumptions?
4. What clarifications are needed?

User Request: {inputs['user_request']}

Provide a structured analysis.""",
                    context={
                        "existing_project": inputs.get("existing_project"),
                        "data_paths": inputs.get("data_paths", []),
                    },
                )
                
                self._state["initial_analysis"] = initial_analysis.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Clarifying questions
            if self._current_step <= 1:
                self._log_step("Step 2: Generating clarifying questions")
                
                questions_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Based on this analysis, what questions should we ask the user
to clarify their requirements?

Analysis: {self._state['initial_analysis']}

Generate 3-5 important clarifying questions. Focus on:
- Specific outcomes they want
- Constraints on methods or tools
- Timeline and scope expectations
- Success metrics""",
                )
                
                self._state["questions"] = questions_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Get user responses (via user_input tool)
            if self._current_step <= 2:
                self._log_step("Step 3: Requesting user clarification")
                
                # Use Bohr to interact with user
                clarification_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Present these clarifying questions to the user and gather their responses.
Use the user_input tool to ask each question.

Questions to ask:
{self._state['questions']}

After getting responses, summarize the clarifications.""",
                )
                
                self._state["clarifications"] = clarification_result.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Technical feasibility check (Feynman)
            if self._current_step <= 3:
                self._log_step("Step 4: Assessing technical feasibility")
                
                feasibility_result = await self._run_agent_task(
                    agent_name="feynman",
                    task=f"""Assess the technical feasibility of this project:

Original Request: {inputs['user_request']}
Clarifications: {self._state['clarifications']}

Consider:
1. Is this technically achievable with available tools?
2. What are the main technical challenges?
3. What computational resources might be needed?
4. Are there any showstoppers?

Provide a feasibility assessment.""",
                )
                
                self._state["feasibility"] = feasibility_result.get("response", "")
                self._current_step = 4
                self.save_checkpoint()
            
            # Step 5: Synthesize project specification (Bohr)
            if self._current_step <= 4:
                self._log_step("Step 5: Creating project specification")
                
                spec_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Create a project specification document based on:

Original Request: {inputs['user_request']}
Clarifications: {self._state['clarifications']}
Feasibility Assessment: {self._state['feasibility']}

Structure the specification with:
1. Project Name (short, descriptive)
2. Goals (numbered list)
3. Data Sources (what data will be used)
4. Constraints (limitations, requirements)
5. Success Criteria (how we know it's done)
6. Recommended Workflow (one of: brainstorming, literature_review, 
   start_project, work_on_existing, explore_dataset)

Write this as a structured document.""",
                )
                
                self._state["project_spec"] = spec_result.get("response", "")
                self._current_step = 5
            
            # Determine recommended workflow
            workflow_map = {
                "brainstorming": "Brainstorming/ideation phase",
                "literature_review": "Background research needed",
                "start_project": "Ready to begin new analysis",
                "work_on_existing": "Continue existing project",
                "explore_dataset": "Data exploration focus",
            }
            
            # Extract recommendation from spec
            recommended = "start_project"  # Default
            spec_lower = self._state["project_spec"].lower()
            for wf_key in workflow_map:
                if wf_key in spec_lower:
                    recommended = wf_key
                    break
            
            # Write project spec to file
            spec_path = self.project_path / "project_specification.md"
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            with open(spec_path, "w") as f:
                f.write(f"# Project Specification\n\n")
                f.write(f"Generated by MiniLab Consultation Module\n\n")
                f.write(self._state["project_spec"])
            
            self._status = WorkflowStatus.COMPLETED
            self._log_step("Consultation completed successfully")
            
            return WorkflowResult(
                status=WorkflowStatus.COMPLETED,
                outputs={
                    "project_spec": self._state["project_spec"],
                    "recommended_workflow": recommended,
                    "clarifications": self._state["clarifications"],
                    "feasibility": self._state["feasibility"],
                },
                artifacts=[str(spec_path)],
                summary=f"Consultation complete. Recommended workflow: {recommended}",
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
