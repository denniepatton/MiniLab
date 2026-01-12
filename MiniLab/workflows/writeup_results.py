"""
WRITE-UP RESULTS Workflow Module.

Documentation phase for creating comprehensive reports.
Led by Gould (librarian/documentation expert).
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import json

from .base import WorkflowModule, WorkflowResult, WorkflowCheckpoint, WorkflowStatus
from ..utils import console


class WriteupResultsModule(WorkflowModule):
    """
    WRITE-UP RESULTS: Documentation and reporting phase.
    
    Purpose:
        - Create comprehensive analysis report
        - Document methodology clearly
        - Generate figures and visualizations
        - Compile bibliography
        - Prepare for peer review
    
    Primary Agent: Gould (documentation, citations)
    Supporting: 
        - Hinton (technical descriptions)
        - Bayes (statistical reporting)
        - Greider (biological narrative)
    
    Outputs:
        - final_report: Complete analysis report
        - figures: Generated visualizations
        - supplementary: Additional materials
        - bibliography: References used
    """
    
    name = "writeup_results"
    description = "Create comprehensive documentation and reports"
    
    required_inputs = ["analysis_results", "validation_report", "project_spec"]
    optional_inputs = ["figure_style", "report_format", "target_audience"]
    expected_outputs = ["final_report", "figures", "supplementary", "bibliography"]
    
    primary_agents = ["gould"]
    supporting_agents = ["hinton", "bayes", "greider"]
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[WorkflowCheckpoint] = None,
    ) -> WorkflowResult:
        """
        Execute write-up workflow.
        
        Steps:
        1. Outline report structure
        2. Write methods section (Hinton)
        3. Write results section (Bayes)
        4. Write biological discussion (Greider)
        5. Generate figures and tables
        6. Compile full report (Gould)
        7. Add citations and references
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
                "analysis_results": inputs["analysis_results"],
                "validation_report": inputs["validation_report"],
                "project_spec": inputs["project_spec"],
                "sections": {},
                "figures": [],
            }
        
        report_format = inputs.get("report_format", "markdown")
        target_audience = inputs.get("target_audience", "technical")
        report_profile = inputs.get("report_profile", "full_paper")

        # Allow callers to control structure explicitly.
        requested_sections = inputs.get("report_sections")
        if isinstance(requested_sections, str):
            requested_sections = [s.strip() for s in requested_sections.split(",") if s.strip()]

        default_profiles: dict[str, list[str]] = {
            "methods_only": ["Title", "Methods"],
            "lit_review": ["Title", "Abstract", "Introduction", "Literature Review", "Conclusion", "References"],
            "code_summary": ["Title", "Summary", "Implementation Details", "How to Run", "Limitations"],
            "full_paper": [
                "Title",
                "Abstract",
                "Introduction",
                "Methods",
                "Results",
                "Discussion",
                "Conclusion",
                "Figure Legends",
                "References",
            ],
        }

        section_plan = requested_sections if requested_sections else default_profiles.get(str(report_profile), default_profiles["full_paper"])
        
        self._log_step("Starting results write-up")
        
        try:
            # Step 1: Outline report structure (Gould)
            if self._current_step <= 0:
                self._log_step("Step 1: Creating report outline")
                
                outline_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Create an outline for the analysis report.

Project Specification:
{inputs['project_spec']}

Analysis Results Summary:
{json.dumps(inputs['analysis_results'], indent=2) if isinstance(inputs['analysis_results'], dict) else inputs['analysis_results'][:1000]}

Target Audience: {target_audience}
Format: {report_format}

Report profile: {report_profile}
Required sections (adapt to context; omit only if clearly irrelevant):
- """ + "\n- ".join(section_plan) + f"""

Create a detailed outline aligned to the required sections.

For each section, note key points to cover.""",
                )
                
                self._state["outline"] = outline_result.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Write methods section (Hinton)
            if self._current_step <= 1:
                self._log_step("Step 2: Writing methods section")
                
                methods_result = await self._run_agent_task(
                    agent_name="hinton",
                    task=f"""Write the Methods section of the report.

Report Outline:
{self._state['outline']}

Analysis Details:
{json.dumps(inputs['analysis_results'], indent=2) if isinstance(inputs['analysis_results'], dict) else inputs['analysis_results']}

Write a clear Methods section covering:
1. Data sources and preprocessing
2. Feature engineering approach
3. Model architecture and design choices
4. Training procedure and hyperparameters
5. Evaluation metrics used

Be precise and reproducible. Include enough detail that another
researcher could replicate the analysis.""",
                )
                
                self._state["sections"]["methods"] = methods_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Write results section (Bayes)
            if self._current_step <= 2:
                self._log_step("Step 3: Writing results section")
                
                results_result = await self._run_agent_task(
                    agent_name="bayes",
                    task=f"""Write the Results section of the report.

Validation Report:
{inputs['validation_report']}

Analysis Results:
{json.dumps(inputs['analysis_results'], indent=2) if isinstance(inputs['analysis_results'], dict) else inputs['analysis_results']}

Write a clear Results section covering:
1. Key quantitative findings with confidence intervals
2. Statistical test results
3. Model performance metrics
4. Comparison to baselines if applicable
5. Any unexpected findings

Present results objectively without interpretation.
Note statistical significance and effect sizes.""",
                )
                
                self._state["sections"]["results"] = results_result.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Write discussion (Greider)
            if self._current_step <= 3:
                self._log_step("Step 4: Writing discussion section")
                
                discussion_result = await self._run_agent_task(
                    agent_name="greider",
                    task=f"""Write the Discussion section of the report.

Results:
{self._state['sections']['results']}

Project Context:
{inputs['project_spec']}

Write a Discussion section covering:
1. Interpretation of key findings
2. Biological/practical significance
3. Comparison with prior work
4. Limitations of the analysis
5. Future directions and implications

Connect the technical results to real-world meaning.
Be appropriately cautious about over-interpretation.""",
                )
                
                self._state["sections"]["discussion"] = discussion_result.get("response", "")
                self._current_step = 4
                self.save_checkpoint()
            
            # Step 5: Generate figure descriptions (Hinton)
            if self._current_step <= 4:
                self._log_step("Step 5: Planning figures and visualizations")
                
                figures_result = await self._run_agent_task(
                    agent_name="hinton",
                    task=f"""Plan figures and visualizations for the report.

Results:
{self._state['sections']['results']}

Methods:
{self._state['sections']['methods']}

For each figure:
1. Figure number and title
2. What it shows
3. Data to include
4. Visualization type (line plot, heatmap, etc.)
5. Key message
6. **Figure legend** (publication-style: what is shown, cohort/data, axes, statistical annotations, sample sizes)

Also write Python code to generate key figures using matplotlib/seaborn.
Save the code using code_editor tool.

Plan 3-5 main figures and any supplementary figures needed.""",
                )
                
                self._state["figure_descriptions"] = figures_result.get("response", "")
                self._current_step = 5
                self.save_checkpoint()
            
            # Step 6: Compile full report (Gould)
            if self._current_step <= 5:
                self._log_step("Step 6: Compiling full report")
                
                compile_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Compile the full analysis report from all sections.

Outline:
{self._state['outline']}

Methods:
{self._state['sections']['methods']}

Results:
{self._state['sections']['results']}

Discussion:
{self._state['sections']['discussion']}

Figure Descriptions:
{self._state['figure_descriptions']}

Project Specification:
{inputs['project_spec']}

Report profile: {report_profile}
Required sections (adapt to context; omit only if clearly irrelevant):
- """ + "\n- ".join(section_plan) + f"""

Compile a complete report aligned to the required sections.

Figure Legends requirements (if figures are present):
- Include a dedicated "Figure Legends" section.
- Use the format: "Figure 1 | Title. Legend..." (one paragraph each).

Format in {report_format}. Ensure smooth transitions between sections.
Write this as a single cohesive document.""",
                )
                
                self._state["full_report"] = compile_result.get("response", "")
                self._current_step = 6
                self.save_checkpoint()
            
            # Step 7: Add citations (Gould)
            if self._current_step <= 6:
                self._log_step("Step 7: Adding citations and references")
                
                citations_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Add citations and compile references for the report.

Full Report:
{self._state['full_report']}

Tasks:
1. Identify claims that need citations
2. Search for appropriate references using pubmed/arxiv tools
3. Add in-text citations
4. Compile reference list

Use citation tools to look up DOIs/PMIDs.
Format references consistently.
Return the report with citations added.""",
                )
                
                self._state["report_with_citations"] = citations_result.get("response", "")
                self._current_step = 7
            
            # Write outputs to files
            output_dir = self.project_path / "report"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Main report
            report_ext = ".md" if report_format == "markdown" else ".txt"
            report_path = output_dir / f"analysis_report{report_ext}"
            with open(report_path, "w") as f:
                f.write(self._state.get("report_with_citations", self._state.get("full_report", "")))
            
            # Individual sections
            sections_dir = output_dir / "sections"
            sections_dir.mkdir(exist_ok=True)
            for section_name, content in self._state.get("sections", {}).items():
                section_path = sections_dir / f"{section_name}.md"
                with open(section_path, "w") as f:
                    f.write(f"# {section_name.title()}\n\n")
                    f.write(content)
            
            # Figure descriptions
            figures_path = output_dir / "figure_descriptions.md"
            with open(figures_path, "w") as f:
                f.write("# Figures\n\n")
                f.write(self._state.get("figure_descriptions", ""))
            
            # Supplementary outline
            supp_path = output_dir / "supplementary_outline.md"
            with open(supp_path, "w") as f:
                f.write("# Supplementary Materials\n\n")
                f.write("## Additional Methods Details\n\n")
                f.write("## Extended Results\n\n")
                f.write("## Supplementary Figures\n\n")
            
            self._status = WorkflowStatus.COMPLETED
            self._log_step("Write-up completed successfully")
            
            artifacts = [str(report_path), str(figures_path), str(supp_path)]
            artifacts.extend([str(sections_dir / f"{s}.md") for s in self._state.get("sections", {})])
            
            return WorkflowResult(
                status=WorkflowStatus.COMPLETED,
                outputs={
                    "final_report": self._state.get("report_with_citations", ""),
                    "figures": self._state.get("figure_descriptions", ""),
                    "supplementary": "See supplementary_outline.md",
                    "bibliography": "Citations included in report",
                },
                artifacts=artifacts,
                summary="Report compilation complete with all sections and citations.",
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
