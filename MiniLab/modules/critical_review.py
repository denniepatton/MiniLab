"""
CRITICAL_REVIEW Module.

Quality assessment phase for evaluating analysis quality.
Led by Farber (critic/quality assessor).

This provides "peer review"-style scrutiny of all major deliverables.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import json

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


@dataclass
class ReviewIssue:
    """A single issue identified in review."""
    severity: str  # critical, major, minor, suggestion
    category: str  # methodology, statistics, interpretation, presentation
    description: str
    location: str
    recommendation: str


class CriticalReviewModule(Module):
    """
    CRITICAL_REVIEW: Quality assessment and iteration.
    
    Purpose:
        - Critically evaluate the analysis
        - Identify methodological issues
        - Check statistical validity
        - Assess interpretation soundness
        - Recommend improvements
        - Decide if iteration is needed
    
    Primary Agent: Farber (critic)
    Supporting: Bayes, Greider, Feynman
    
    Outputs:
        - eval/critical_review.md: Comprehensive review
        - issues_list: Categorized issues found
        - recommendations: Suggested improvements
        - verdict: Pass/Revise/Fail
    """
    
    name = "critical_review"
    description = "Quality assessment and improvement recommendations"
    module_type = ModuleType.SUBGRAPH
    
    required_inputs = ["analysis_results", "validation_report", "final_report"]
    optional_inputs = ["review_focus", "strictness_level"]
    expected_outputs = ["review_report", "issues_list", "recommendations", "verdict"]
    
    primary_agents = ["farber"]
    supporting_agents = ["bayes", "greider", "feynman"]
    
    SEVERITY_WEIGHTS = {
        "critical": 10,
        "major": 5,
        "minor": 1,
        "suggestion": 0,
    }
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """
        Execute critical review module.
        
        Steps:
        1. Methodology review (Farber)
        2. Statistical review (Bayes)
        3. Biological validity check (Greider)
        4. Technical correctness (Feynman)
        5. Compile issues and recommendations
        6. Determine verdict
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
                "final_report": inputs["final_report"],
                "issues": [],
                "reviews": {},
            }
        
        strictness = inputs.get("strictness_level", "standard")
        
        self._log_step("Starting critical review")
        
        try:
            # Step 1: Methodology review (Farber)
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
1. Is the overall approach sound?
2. Are there logical gaps or unjustified assumptions?
3. Are methods well-suited to the research question?
4. Are there alternative approaches that should have been considered?

For each issue, categorize as:
- CRITICAL: Fundamental flaw that invalidates results
- MAJOR: Significant issue that needs addressing
- MINOR: Small issue or improvement opportunity
- SUGGESTION: Optional enhancement

Format: [SEVERITY] Category: Description | Recommendation""",
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
1. Are statistical tests appropriate?
2. Are assumptions validated?
3. Are effect sizes reported?
4. Is multiple testing corrected?
5. Are confidence intervals appropriate?

Format: [SEVERITY] Statistics: Description | Recommendation""",
                )
                
                self._state["reviews"]["statistics"] = stats_review.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Biological validity (Greider)
            if self._current_step <= 2:
                self._log_step("Step 3: Biological validity check")
                
                bio_review = await self._run_agent_task(
                    agent_name="greider",
                    task=f"""Review biological validity of conclusions.

Final Report:
{inputs['final_report'][:3000]}...

Evaluate:
1. Are biological interpretations supported by data?
2. Are findings consistent with known biology?
3. Are extraordinary claims backed by extraordinary evidence?
4. Are biological mechanisms proposed plausible?

Format: [SEVERITY] Biology: Description | Recommendation""",
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

Format: [SEVERITY] Technical: Description | Recommendation""",
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
2. Organize by severity
3. Prioritize by impact
4. Ensure recommendations are actionable

Create JSON list:
```json
[{{"severity": "...", "category": "...", "description": "...", "recommendation": "..."}}]
```

Then summarize key concerns.""",
                )
                
                self._state["issues_compiled"] = compile_result.get("response", "")
                
                try:
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
                
                issues = self._state.get("issues", [])
                issue_score = sum(
                    self.SEVERITY_WEIGHTS.get(issue.get("severity", "minor"), 1)
                    for issue in issues
                )
                
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
                
                verdict_result = await self._run_agent_task(
                    agent_name="farber",
                    task=f"""Make a final verdict on this analysis.

Issues Summary:
{self._state['issues_compiled']}

Score: {issue_score} (thresholds: pass<={thresholds['pass']}, revise<={thresholds['revise']})
Preliminary: {preliminary_verdict}

Provide final verdict:
- PASS: Ready with minor edits
- REVISE: Needs specific improvements
- MAJOR_REVISION: Significant work needed
- REJECT: Fundamental flaws

Start with "VERDICT: [verdict]" then explain.""",
                )
                
                verdict_response = verdict_result.get("response", "")
                
                if "VERDICT:" in verdict_response:
                    verdict_line = verdict_response.split("VERDICT:")[1].split("\n")[0].strip()
                    for v in ["PASS", "REVISE", "MAJOR_REVISION", "REJECT"]:
                        if v in verdict_line.upper():
                            self._state["verdict"] = v
                            break
                    else:
                        self._state["verdict"] = preliminary_verdict
                else:
                    self._state["verdict"] = preliminary_verdict
                
                self._state["verdict_reasoning"] = verdict_response
                self._current_step = 6
            
            # Write outputs
            self._write_outputs()
            
            self._status = ModuleStatus.COMPLETED
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "review_report": self._state.get("issues_compiled", ""),
                    "issues_list": self._state.get("issues", []),
                    "recommendations": self._state.get("verdict_reasoning", ""),
                    "verdict": self._state.get("verdict", ""),
                },
                artifacts=[
                    str(self.project_path / "eval" / "critical_review.md"),
                ],
                summary=f"Critical review complete. Verdict: {self._state.get('verdict', 'UNKNOWN')}",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
    
    def _write_outputs(self) -> None:
        """Write outputs to eval directory."""
        eval_dir = self.project_path / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        with open(eval_dir / "critical_review.md", "w") as f:
            f.write("# Critical Review\n\n")
            f.write(f"## Verdict: {self._state.get('verdict', 'UNKNOWN')}\n\n")
            f.write("## Review Summary\n\n")
            f.write(self._state.get("issues_compiled", "") + "\n\n")
            f.write("## Individual Reviews\n\n")
            for name, content in self._state.get("reviews", {}).items():
                f.write(f"### {name.title()} Review\n\n")
                f.write(content + "\n\n")
            f.write("## Verdict Reasoning\n\n")
            f.write(self._state.get("verdict_reasoning", ""))
