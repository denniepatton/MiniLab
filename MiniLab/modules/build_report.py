"""
BUILD_REPORT Module (formerly WriteupResultsModule).

Documentation phase for creating comprehensive reports.
Led by Gould (librarian/documentation expert).

This module assembles narrative outputs grounded in artifacts and results.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import json

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


class BuildReportModule(Module):
    """
    BUILD_REPORT: Documentation and reporting phase.
    
    Subgraph: outline → draft → cite → format → audit → finalize
    
    Purpose:
        - Create comprehensive analysis report
        - Document methodology clearly
        - Generate figures and visualizations
        - Compile bibliography
        - Prepare for peer review
    
    Primary Agent: Gould (documentation, citations)
    Supporting: Hinton, Bayes, Greider
    
    Outputs:
        - reports/: Final documents (docx/pdf/md)
        - reports/methods.docx: Methods narrative
        - results/figures/: Generated visualizations
    """
    
    name = "build_report"
    description = "Create comprehensive documentation and reports"
    module_type = ModuleType.SUBGRAPH
    
    required_inputs = ["analysis_results", "validation_report", "project_spec"]
    optional_inputs = ["figure_style", "report_format", "target_audience", "report_profile", "report_sections"]
    expected_outputs = ["final_report", "figures", "supplementary", "bibliography"]
    
    primary_agents = ["gould"]
    supporting_agents = ["hinton", "bayes", "greider"]
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """
        Execute build report module.
        
        Steps:
        1. Outline report structure (Gould)
        2. Write methods section (Hinton)
        3. Write results section (Bayes)
        4. Write biological discussion (Greider)
        5. Generate figures and tables
        6. Compile full report (Gould)
        7. Add citations and references
        """
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
                "analysis_results": inputs["analysis_results"],
                "validation_report": inputs["validation_report"],
                "project_spec": inputs["project_spec"],
                "sections": {},
                "figures": [],
            }
        
        report_format = inputs.get("report_format", "markdown")
        target_audience = inputs.get("target_audience", "technical")
        report_profile = inputs.get("report_profile", "full_paper")

        requested_sections = inputs.get("report_sections")
        if isinstance(requested_sections, str):
            requested_sections = [s.strip() for s in requested_sections.split(",") if s.strip()]

        default_profiles = {
            "methods_only": ["Title", "Methods"],
            "lit_review": ["Title", "Abstract", "Introduction", "Literature Review", "Conclusion", "References"],
            "code_summary": ["Title", "Summary", "Implementation Details", "How to Run", "Limitations"],
            "full_paper": [
                "Title", "Abstract", "Introduction", "Methods", "Results",
                "Discussion", "Conclusion", "Figure Legends", "References",
            ],
        }

        section_plan = requested_sections if requested_sections else default_profiles.get(str(report_profile), default_profiles["full_paper"])
        
        self._log_step("Starting results write-up")
        
        try:
            # Step 1: Outline report structure
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
Required sections: {', '.join(section_plan)}

Create a detailed outline with key points for each section.""",
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

Be precise and reproducible.""",
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

Present results objectively without interpretation.""",
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

Write a Discussion covering:
1. Interpretation of key findings
2. Biological/practical significance
3. Comparison with prior work
4. Limitations
5. Future directions""",
                )
                
                self._state["sections"]["discussion"] = discussion_result.get("response", "")
                self._current_step = 4
                self.save_checkpoint()
            
            # Step 5: Plan figures (Hinton)
            if self._current_step <= 4:
                self._log_step("Step 5: Planning figures")
                
                figures_result = await self._run_agent_task(
                    agent_name="hinton",
                    task=f"""Plan figures for the report.

Results:
{self._state['sections']['results']}

Methods:
{self._state['sections']['methods']}

For each figure provide:
1. Figure number and title
2. What it shows
3. Data to include
4. Visualization type
5. Figure legend

Also write Python code to generate key figures.
Save code to: scripts/05_figures.py
Save figures to: results/figures/

Plan 3-5 main figures.""",
                )
                
                self._state["figure_descriptions"] = figures_result.get("response", "")
                self._current_step = 5
                self.save_checkpoint()
            
            # Step 6: Compile full report (Gould)
            if self._current_step <= 5:
                self._log_step("Step 6: Compiling full report")
                
                compile_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Compile the full analysis report.

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

Required sections: {', '.join(section_plan)}

Compile a complete report with smooth transitions.
Format: {report_format}""",
                )
                
                self._state["full_report"] = compile_result.get("response", "")
                self._current_step = 6
                self.save_checkpoint()
            
            # Step 7: Add citations (Gould)
            if self._current_step <= 6:
                self._log_step("Step 7: Adding citations")
                
                citations_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Add citations and compile references.

Full Report:
{self._state['full_report']}

Tasks:
1. Identify claims needing citations
2. Search for appropriate references
3. Add in-text citations
4. Compile reference list

Return the report with citations added.""",
                )
                
                self._state["report_with_citations"] = citations_result.get("response", "")
                self._current_step = 7
            
            # Write outputs
            self._write_outputs(report_format)
            
            self._status = ModuleStatus.COMPLETED
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "final_report": self._state.get("report_with_citations", self._state.get("full_report", "")),
                    "figures": self._state.get("figure_descriptions", ""),
                    "supplementary": "",
                    "bibliography": "",
                },
                artifacts=[
                    str(self.project_path / "reports" / f"analysis_report.{report_format[:2]}"),
                ],
                summary="Report generation complete.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
    
    def _write_outputs(self, report_format: str) -> None:
        """Write outputs to appropriate directories."""
        reports_dir = self.project_path / "reports"
        sections_dir = reports_dir / "sections"
        
        for d in [reports_dir, sections_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Main report
        report_ext = ".md" if report_format == "markdown" else ".txt"
        report_path = reports_dir / f"analysis_report{report_ext}"
        with open(report_path, "w") as f:
            f.write(self._state.get("report_with_citations", self._state.get("full_report", "")))
        
        # Individual sections
        for section_name, content in self._state.get("sections", {}).items():
            section_path = sections_dir / f"{section_name}.md"
            with open(section_path, "w") as f:
                f.write(content)


# Backward compatibility alias
WriteupResultsModule = BuildReportModule
