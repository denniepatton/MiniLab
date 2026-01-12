"""
ONE_ON_ONE Module.

Linear module for deep dive with a specific expert agent.
"""

from typing import Any, Optional
from pathlib import Path

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


class OneOnOneModule(Module):
    """
    ONE_ON_ONE: Deep dive with a specific expert.
    
    Goal: Get targeted recommendations, pitfalls, and next actions
    from a specific expert agent.
    
    Linear flow:
    1. Bohr frames question with context
    2. Expert responds with detailed analysis
    3. Bohr summarizes actionable recommendations
    
    Primary Agent: Dynamic (specified in inputs)
    
    Outputs:
        - Targeted recommendations
        - Identified pitfalls
        - Suggested next actions
    """
    
    name = "one_on_one"
    description = "Deep dive consultation with specific expert"
    module_type = ModuleType.LINEAR
    
    required_inputs = ["target_agent", "topic", "context"]
    optional_inputs = ["specific_questions", "artifacts_to_review"]
    expected_outputs = ["recommendations", "pitfalls", "next_actions"]
    
    primary_agents = ["bohr"]
    supporting_agents = ["feynman", "hinton", "dayhoff", "bayes", "shannon", "greider", "gould", "farber"]
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """Execute one-on-one consultation."""
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        target_agent = inputs["target_agent"].lower()
        if target_agent not in self.agents:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Target agent '{target_agent}' not available",
            )
        
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = ModuleStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "target_agent": target_agent,
                "topic": inputs["topic"],
                "context": inputs["context"],
                "specific_questions": inputs.get("specific_questions", []),
            }
        
        self._log_step(f"Starting one-on-one with {target_agent}")
        
        try:
            # Step 1: Bohr frames the question
            if self._current_step <= 0:
                self._log_step("Step 1: Framing the consultation")
                
                specific_q = self._state.get("specific_questions", [])
                questions_text = "\n".join(f"- {q}" for q in specific_q) if specific_q else "General consultation"
                
                framing_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Frame a consultation question for {target_agent}.

Topic: {inputs['topic']}

Context:
{inputs['context'][:2000]}

Specific Questions:
{questions_text}

Create a clear, focused brief for {target_agent} that:
1. States the problem concisely
2. Provides necessary context
3. Asks specific questions
4. Identifies what decisions depend on their input""",
                )
                
                self._state["framed_question"] = framing_result.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
                
                console.agent_message("BOHR", f"Consulting {target_agent.upper()}...")
            
            # Step 2: Expert responds
            if self._current_step <= 1:
                self._log_step(f"Step 2: {target_agent} analysis")
                
                expert_result = await self._run_agent_task(
                    agent_name=target_agent,
                    task=f"""You are being consulted on a specific topic.

Consultation Brief:
{self._state['framed_question']}

Provide a detailed response including:
1. Your expert assessment of the situation
2. Key recommendations (be specific)
3. Potential pitfalls or risks to watch for
4. Suggested next steps
5. Any assumptions you're making

Draw on your domain expertise to give actionable guidance.""",
                )
                
                self._state["expert_response"] = expert_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
                
                console.agent_message(target_agent.upper(), self._state["expert_response"])
            
            # Step 3: Bohr summarizes
            if self._current_step <= 2:
                self._log_step("Step 3: Summarizing recommendations")
                
                summary_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Summarize the expert consultation.

Expert ({target_agent}):
{self._state['expert_response']}

Create a concise summary with:
1. KEY RECOMMENDATIONS: Bullet list of actionable items
2. PITFALLS: What to avoid
3. NEXT ACTIONS: Prioritized list of next steps
4. DECISION POINTS: What needs to be decided

Keep it actionable and clear.""",
                )
                
                self._state["summary"] = summary_result.get("response", "")
                self._current_step = 3
            
            self._status = ModuleStatus.COMPLETED
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "recommendations": self._state.get("expert_response", ""),
                    "pitfalls": "",  # Extracted in summary
                    "next_actions": self._state.get("summary", ""),
                    "expert": target_agent,
                },
                summary=f"One-on-one with {target_agent} complete.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
