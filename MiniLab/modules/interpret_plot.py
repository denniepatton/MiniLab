"""
INTERPRET_PLOT Module.

Linear module for figure/plot interpretation.
"""

from typing import Any, Optional
from pathlib import Path

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


class InterpretPlotModule(Module):
    """
    INTERPRET_PLOT: Write caption and prose from figure.
    
    Linear flow:
    1. Analyze figure content
    2. Write descriptive caption
    3. Write interpretive prose
    4. Check for figure quality issues
    
    Primary Agent: Gould (visual communication)
    Supporting: Farber (claims discipline)
    
    Outputs:
        - Figure caption
        - Interpretive prose
        - Quality assessment
    """
    
    name = "interpret_plot"
    description = "Write caption and prose from figure"
    module_type = ModuleType.LINEAR
    
    required_inputs = ["figure_path", "figure_context"]
    optional_inputs = ["expected_patterns", "statistical_context", "audience"]
    expected_outputs = ["caption", "interpretation", "quality_issues"]
    
    primary_agents = ["gould"]
    supporting_agents = ["farber"]
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """Execute plot interpretation."""
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        figure_path = Path(inputs["figure_path"])
        # Note: Figure may not exist yet if being described from memory
        
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = ModuleStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "figure_path": str(figure_path),
                "figure_context": inputs["figure_context"],
                "expected_patterns": inputs.get("expected_patterns", []),
                "statistical_context": inputs.get("statistical_context", ""),
                "audience": inputs.get("audience", "scientific"),
            }
        
        self._log_step(f"Interpreting figure: {figure_path.name if figure_path.suffix else 'described figure'}")
        
        try:
            # Step 1: Analyze figure content
            if self._current_step <= 0:
                self._log_step("Step 1: Analyzing figure content")
                
                expected = self._state.get("expected_patterns", [])
                expected_text = "\n".join(f"- {p}" for p in expected) if expected else "No specific patterns expected"
                
                analyze_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Analyze this figure for content and patterns.

Figure: {inputs['figure_path']}
Context: {inputs['figure_context']}

Expected patterns:
{expected_text}

Statistical context: {self._state.get('statistical_context', 'Not provided')}

Describe:
1. Figure type (scatter, bar, heatmap, etc.)
2. Axes and what they represent
3. Key visual patterns
4. Color coding and legends
5. Statistical annotations (if any)
6. Main message the figure conveys""",
                )
                
                self._state["figure_analysis"] = analyze_result.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Write caption
            if self._current_step <= 1:
                self._log_step("Step 2: Writing figure caption")
                
                caption_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Write a scientific figure caption.

Figure Analysis:
{self._state['figure_analysis']}

Context: {inputs['figure_context']}
Audience: {self._state.get('audience', 'scientific')}

Write a caption that:
1. Starts with a brief title/summary (bolded)
2. Describes what is shown (axes, groups, conditions)
3. Defines any abbreviations
4. Notes sample sizes if relevant
5. Mentions statistical tests if applicable

Format: **Title.** Descriptive text. (n=X per group)""",
                )
                
                self._state["caption"] = caption_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Write interpretive prose
            if self._current_step <= 2:
                self._log_step("Step 3: Writing interpretation")
                
                interp_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Write interpretive prose for results section.

Figure Analysis:
{self._state['figure_analysis']}

Caption:
{self._state['caption']}

Context: {inputs['figure_context']}
Statistical context: {self._state.get('statistical_context', '')}

Write 1-2 paragraphs for a results section:
1. Reference the figure ("Figure X shows...")
2. State the main finding
3. Note specific patterns or comparisons
4. Connect to the broader analysis
5. Avoid over-interpretation

Use present tense for describing what the figure shows.""",
                )
                
                self._state["interpretation"] = interp_result.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Quality check (using Farber for claims discipline)
            if self._current_step <= 3:
                self._log_step("Step 4: Quality assessment")
                
                quality_result = await self._run_agent_task(
                    agent_name="farber",
                    task=f"""Review this figure interpretation for quality and claims discipline.

Caption:
{self._state['caption']}

Interpretation:
{self._state['interpretation']}

Context: {inputs['figure_context']}

Check for:
1. Caption completeness (all elements described?)
2. Over-claims (saying more than data shows?)
3. Under-claims (missing important patterns?)
4. Clarity issues
5. Missing statistical context
6. Figure design issues (if apparent from description)

Provide specific feedback for improvement.""",
                )
                
                self._state["quality_issues"] = quality_result.get("response", "")
                self._current_step = 4
            
            self._status = ModuleStatus.COMPLETED
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "caption": self._state.get("caption", ""),
                    "interpretation": self._state.get("interpretation", ""),
                    "quality_issues": self._state.get("quality_issues", ""),
                },
                summary="Figure interpretation complete.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
