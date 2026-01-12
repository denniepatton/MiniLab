"""
INTERPRET_STATS Module.

Linear module for statistical result interpretation.
"""

from typing import Any, Optional
from pathlib import Path

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


class InterpretStatsModule(Module):
    """
    INTERPRET_STATS: Write prose from stat output.
    
    Linear flow:
    1. Parse statistical output
    2. Identify key findings
    3. Assess statistical significance and effect sizes
    4. Write interpretive prose
    5. Flag any concerns
    
    Primary Agent: Bayes
    
    Outputs:
        - Statistical interpretation prose
        - Key findings summary
        - Concerns/limitations
    """
    
    name = "interpret_stats"
    description = "Write interpretive prose from statistical output"
    module_type = ModuleType.LINEAR
    
    required_inputs = ["stat_output", "analysis_context"]
    optional_inputs = ["hypotheses", "significance_threshold", "prior_expectations"]
    expected_outputs = ["interpretation", "key_findings", "concerns"]
    
    primary_agents = ["bayes"]
    supporting_agents = ["farber"]
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """Execute statistical interpretation."""
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
                "stat_output": inputs["stat_output"],
                "analysis_context": inputs["analysis_context"],
                "hypotheses": inputs.get("hypotheses", []),
                "alpha": inputs.get("significance_threshold", 0.05),
                "prior_expectations": inputs.get("prior_expectations", ""),
            }
        
        self._log_step("Interpreting statistical results")
        
        try:
            # Step 1: Parse and identify key statistics
            if self._current_step <= 0:
                self._log_step("Step 1: Parsing statistical output")
                
                parse_result = await self._run_agent_task(
                    agent_name="bayes",
                    task=f"""Parse these statistical results and identify key values.

Statistical Output:
{inputs['stat_output'][:3000]}

Analysis Context: {inputs['analysis_context']}
Significance Threshold (α): {self._state['alpha']}

Extract and organize:
1. Test statistics (t, F, χ², etc.)
2. P-values
3. Confidence intervals
4. Effect sizes (if available)
5. Sample sizes
6. Model fit metrics (R², AIC, etc.)

Assess which results are statistically significant.""",
                )
                
                self._state["parsed_stats"] = parse_result.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Identify key findings
            if self._current_step <= 1:
                self._log_step("Step 2: Identifying key findings")
                
                hypotheses_text = ""
                if self._state.get("hypotheses"):
                    hypotheses_text = f"\n\nHypotheses being tested:\n" + "\n".join(
                        f"- {h}" for h in self._state["hypotheses"]
                    )
                
                findings_result = await self._run_agent_task(
                    agent_name="bayes",
                    task=f"""Identify key findings from the statistical analysis.

Parsed Statistics:
{self._state['parsed_stats']}

Context: {inputs['analysis_context']}
{hypotheses_text}

Prior Expectations: {self._state.get('prior_expectations', 'None specified')}

For each key finding:
1. State the finding clearly
2. Statistical evidence (p-value, CI, effect size)
3. Practical significance (is the effect meaningful?)
4. How it relates to hypotheses
5. Whether it aligns with or contradicts expectations

Prioritize by importance.""",
                )
                
                self._state["key_findings"] = findings_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Write interpretation
            if self._current_step <= 2:
                self._log_step("Step 3: Writing interpretation")
                
                interp_result = await self._run_agent_task(
                    agent_name="bayes",
                    task=f"""Write clear, accessible prose interpreting these results.

Key Findings:
{self._state['key_findings']}

Context: {inputs['analysis_context']}

Write interpretation suitable for a scientific paper:
1. Start with the main finding
2. Support with specific statistics
3. Discuss effect sizes and practical significance
4. Compare to prior expectations
5. Note any surprising results

Use precise statistical language but remain accessible.
Avoid p-hacking language ("trending toward significance").
Report effect sizes, not just p-values.""",
                )
                
                self._state["interpretation"] = interp_result.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Flag concerns
            if self._current_step <= 3:
                self._log_step("Step 4: Flagging concerns and limitations")
                
                concerns_result = await self._run_agent_task(
                    agent_name="bayes",
                    task=f"""Identify any statistical concerns or limitations.

Statistical Output:
{inputs['stat_output'][:2000]}

Interpretation:
{self._state['interpretation'][:1500]}

Check for:
1. Multiple comparisons issues
2. Small sample sizes
3. Violated assumptions
4. Missing data impact
5. Potential confounders
6. Generalizability limits
7. Effect sizes that are statistically but not practically significant

List concerns with severity and suggested mitigations.""",
                )
                
                self._state["concerns"] = concerns_result.get("response", "")
                self._current_step = 4
            
            self._status = ModuleStatus.COMPLETED
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "interpretation": self._state.get("interpretation", ""),
                    "key_findings": self._state.get("key_findings", ""),
                    "concerns": self._state.get("concerns", ""),
                },
                summary="Statistical interpretation complete.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
