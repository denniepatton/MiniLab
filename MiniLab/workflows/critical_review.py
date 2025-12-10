"""
CRITICAL REVIEW Workflow Module.

Quality assessment phase for evaluating analysis quality.
Led by Farber (critic/quality assessor).
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import json

from .base import WorkflowModule, WorkflowResult, WorkflowCheckpoint, WorkflowStatus
from ..utils import console


@dataclass
class ReviewIssue:
    """A single issue identified in review."""
    severity: str  # critical, major, minor, suggestion
    category: str  # methodology, statistics, interpretation, presentation
    description: str
    location: str  # Where in the analysis/report
    recommendation: str


class CriticalReviewModule(WorkflowModule):
    """
    CRITICAL REVIEW: Quality assessment and iteration.
    
    Purpose:
        - Critically evaluate the analysis
        - Identify methodological issues
        - Check statistical validity
        - Assess interpretation soundness
        - Recommend improvements
        - Decide if iteration is needed
    
    Primary Agent: Farber (critic)
    Supporting:
        - Bayes (statistical review)
        - Greider (biological validity)
        - Feynman (technical correctness)
    
    Outputs:
        - review_report: Comprehensive review
        - issues_list: Categorized issues found
        - recommendations: Suggested improvements
        - verdict: Pass/Revise/Fail
    """
    
    name = "critical_review"
    description = "Quality assessment and improvement recommendations"
    
    required_inputs = ["analysis_results", "validation_report", "final_report"]
    optional_inputs = ["review_focus", "strictness_level"]
    expected_outputs = ["review_report", "issues_list", "recommendations", "verdict"]
    
    primary_agents = ["farber"]
    supporting_agents = ["bayes", "greider", "feynman"]
    
    # Issue severity weights for verdict calculation
    SEVERITY_WEIGHTS = {
        "critical": 10,
        "major": 5,
        "minor": 1,
        "suggestion": 0,
    }
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[WorkflowCheckpoint] = None,
    ) -> WorkflowResult:
        """
        Execute critical review workflow.
        
        Steps:
        1. Methodology review (Farber)
        2. Statistical review (Bayes)
        3. Biological validity check (Greider)
        4. Technical correctness (Feynman)
        5. Compile issues and recommendations
        6. Determine verdict
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
                "final_report": inputs["final_report"],
                "issues": [],
                "reviews": {},
            }
        
        strictness = inputs.get("strictness_level", "standard")  # lenient, standard, strict
        
        self._log_step("Starting critical review")
        
        try:
            # Step 1: Overall methodology review (Farber)
            if self._current_step <= 0:
                self._log_step("Step 1: Methodology review")
                
                method_review = await self._run_agent_task(
                    agent_name="farber",
                    task=f"""Critically review the methodology of this analysis.

Final Report:
{inputs['final_report'][:3000]}...

Validation Report:
{inputs['validation_report'][:1000]}

Evaluate:
1. Is the overall approach sound and appropriate?
2. Are there logical gaps or unjustified assumptions?
3. Are the methods well-suited to the research question?
4. Are there alternative approaches that should have been considered?
5. Is the analysis reproducible from the description?

For each issue found, categorize as:
- CRITICAL: Fundamental flaw that invalidates results
- MAJOR: Significant issue that needs addressing
- MINOR: Small issue or improvement opportunity
- SUGGESTION: Optional enhancement

Format: [SEVERITY] Category: Description | Recommendation

Be thorough but fair. This is constructive criticism.""",
                )
                
                self._state["reviews"]["methodology"] = method_review.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Statistical review (Bayes)
            if self._current_step <= 1:
                self._log_step("Step 2: Statistical review")
                
                stats_review = await self._run_agent_task(
                    agent_name="bayes",
                    task=f"""Review the statistical validity of this analysis.

Validation Report:
{inputs['validation_report']}

Analysis Results:
{json.dumps(inputs['analysis_results'], indent=2) if isinstance(inputs['analysis_results'], dict) else str(inputs['analysis_results'])[:2000]}

Evaluate:
1. Are statistical tests appropriate for the data?
2. Are assumptions validated before using methods?
3. Are effect sizes reported alongside p-values?
4. Is multiple testing correction applied if needed?
5. Are confidence intervals appropriate?
6. Is uncertainty properly quantified?
7. Are there signs of p-hacking or data dredging?

Identify issues with format:
[SEVERITY] Statistics: Description | Recommendation

Be rigorous about statistical best practices.""",
                )
                
                self._state["reviews"]["statistics"] = stats_review.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Biological validity (Greider)
            if self._current_step <= 2:
                self._log_step("Step 3: Biological validity check")
                
                bio_review = await self._run_agent_task(
                    agent_name="greider",
                    task=f"""Review the biological validity of conclusions.

Final Report:
{inputs['final_report'][:3000]}...

Evaluate:
1. Are biological interpretations supported by the data?
2. Are findings consistent with known biology?
3. Are extraordinary claims backed by extraordinary evidence?
4. Are biological mechanisms proposed plausible?
5. Are limitations in biological interpretation noted?
6. Could the findings be confounded by technical artifacts?

Identify issues with format:
[SEVERITY] Biology: Description | Recommendation

Balance skepticism with recognition of genuine discovery.""",
                )
                
                self._state["reviews"]["biology"] = bio_review.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Technical correctness (Feynman)
            if self._current_step <= 3:
                self._log_step("Step 4: Technical correctness review")
                
                tech_review = await self._run_agent_task(
                    agent_name="feynman",
                    task=f"""Review technical correctness of the implementation.

Analysis Results:
{json.dumps(inputs['analysis_results'], indent=2) if isinstance(inputs['analysis_results'], dict) else str(inputs['analysis_results'])[:2000]}

Validation Report:
{inputs['validation_report'][:1000]}

Evaluate:
1. Are computational methods implemented correctly?
2. Are there potential bugs or edge cases?
3. Is data preprocessing appropriate?
4. Are hyperparameters justified?
5. Is code quality sufficient for reproducibility?
6. Are there computational efficiency issues?

Identify issues with format:
[SEVERITY] Technical: Description | Recommendation

Focus on correctness and best practices.""",
                )
                
                self._state["reviews"]["technical"] = tech_review.get("response", "")
                self._current_step = 4
                self.save_checkpoint()
            
            # Step 5: Compile issues (Farber)
            if self._current_step <= 4:
                self._log_step("Step 5: Compiling issues")
                
                all_reviews = "\n\n".join([
                    f"=== {name.upper()} ===\n{content}"
                    for name, content in self._state["reviews"].items()
                ])
                
                compile_result = await self._run_agent_task(
                    agent_name="farber",
                    task=f"""Compile all review feedback into a structured issues list.

All Reviews:
{all_reviews}

Tasks:
1. Consolidate duplicate issues
2. Organize by severity (critical, major, minor, suggestion)
3. Organize by category (methodology, statistics, biology, technical, presentation)
4. Prioritize by impact
5. Ensure recommendations are actionable

Create a structured issues list in JSON format:
```json
[
  {{"severity": "...", "category": "...", "description": "...", "recommendation": "..."}}
]
```

Then summarize the key concerns.""",
                )
                
                self._state["issues_compiled"] = compile_result.get("response", "")
                
                # Parse issues from response
                try:
                    # Extract JSON from response
                    response = compile_result.get("response", "")
                    json_start = response.find("[")
                    json_end = response.rfind("]") + 1
                    if json_start >= 0 and json_end > json_start:
                        issues_json = response[json_start:json_end]
                        self._state["issues"] = json.loads(issues_json)
                except (json.JSONDecodeError, ValueError):
                    self._state["issues"] = []
                
                self._current_step = 5
                self.save_checkpoint()
            
            # Step 6: Determine verdict (Farber)
            if self._current_step <= 5:
                self._log_step("Step 6: Determining verdict")
                
                # Calculate issue score
                issues = self._state.get("issues", [])
                issue_score = sum(
                    self.SEVERITY_WEIGHTS.get(issue.get("severity", "minor"), 1)
                    for issue in issues
                )
                
                # Strictness adjustments
                strictness_thresholds = {
                    "lenient": {"pass": 15, "revise": 30},
                    "standard": {"pass": 10, "revise": 20},
                    "strict": {"pass": 5, "revise": 10},
                }
                thresholds = strictness_thresholds.get(strictness, strictness_thresholds["standard"])
                
                if issue_score <= thresholds["pass"]:
                    preliminary_verdict = "PASS"
                elif issue_score <= thresholds["revise"]:
                    preliminary_verdict = "REVISE"
                else:
                    preliminary_verdict = "MAJOR_REVISION"
                
                # Let Farber make final determination
                verdict_result = await self._run_agent_task(
                    agent_name="farber",
                    task=f"""Make a final verdict on this analysis.

Issues Summary:
{self._state['issues_compiled']}

Calculated Score: {issue_score} (thresholds: pass<={thresholds['pass']}, revise<={thresholds['revise']})
Preliminary Verdict: {preliminary_verdict}

Considering:
1. Severity of issues found
2. Whether critical flaws exist
3. Feasibility of addressing issues
4. Overall quality of work

Provide your final verdict:
- PASS: Ready for use/publication with minor edits
- REVISE: Needs specific improvements before acceptance
- MAJOR_REVISION: Significant work needed
- REJECT: Fundamental flaws require starting over

Start your response with "VERDICT: [verdict]"

Then explain your reasoning and provide prioritized recommendations.""",
                )
                
                verdict_response = verdict_result.get("response", "")
                
                # Parse verdict
                if "VERDICT:" in verdict_response:
                    verdict_line = verdict_response.split("\n")[0]
                    verdict = verdict_line.split(":")[1].strip().upper()
                    if verdict not in ["PASS", "REVISE", "MAJOR_REVISION", "REJECT"]:
                        verdict = preliminary_verdict
                else:
                    verdict = preliminary_verdict
                
                self._state["verdict"] = verdict
                self._state["verdict_explanation"] = verdict_response
                self._current_step = 6
            
            # Write outputs to files
            output_dir = self.project_path / "review"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Full review report
            report_path = output_dir / "review_report.md"
            with open(report_path, "w") as f:
                f.write("# Critical Review Report\n\n")
                f.write(f"## Verdict: {self._state['verdict']}\n\n")
                f.write(self._state.get("verdict_explanation", ""))
                f.write("\n\n---\n\n")
                f.write("## Detailed Reviews\n\n")
                for name, content in self._state.get("reviews", {}).items():
                    f.write(f"### {name.title()} Review\n\n")
                    f.write(content)
                    f.write("\n\n")
            
            # Issues list
            issues_path = output_dir / "issues_list.json"
            with open(issues_path, "w") as f:
                json.dump(self._state.get("issues", []), f, indent=2)
            
            # Recommendations summary
            recs_path = output_dir / "recommendations.md"
            with open(recs_path, "w") as f:
                f.write("# Recommendations\n\n")
                f.write(f"Verdict: {self._state['verdict']}\n\n")
                f.write("## Priority Actions\n\n")
                for issue in self._state.get("issues", []):
                    if issue.get("severity") in ["critical", "major"]:
                        f.write(f"- **[{issue.get('severity', 'unknown').upper()}]** ")
                        f.write(f"{issue.get('description', '')}\n")
                        f.write(f"  - *Recommendation:* {issue.get('recommendation', '')}\n\n")
            
            self._status = WorkflowStatus.COMPLETED
            self._log_step(f"Critical review completed. Verdict: {self._state['verdict']}")
            
            return WorkflowResult(
                status=WorkflowStatus.COMPLETED,
                outputs={
                    "review_report": self._state.get("verdict_explanation", ""),
                    "issues_list": self._state.get("issues", []),
                    "recommendations": self._state.get("issues_compiled", ""),
                    "verdict": self._state.get("verdict", "UNKNOWN"),
                },
                artifacts=[str(report_path), str(issues_path), str(recs_path)],
                summary=f"Critical review complete. Verdict: {self._state['verdict']} with {len(self._state.get('issues', []))} issues identified.",
                metadata={
                    "issue_score": sum(
                        self.SEVERITY_WEIGHTS.get(i.get("severity", "minor"), 1)
                        for i in self._state.get("issues", [])
                    ),
                    "critical_count": len([i for i in self._state.get("issues", []) if i.get("severity") == "critical"]),
                    "major_count": len([i for i in self._state.get("issues", []) if i.get("severity") == "major"]),
                },
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
