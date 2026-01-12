"""
CORE_INPUT Module.

Subgraph for getting coherent answer from core subgroups.

Cores:
- Synthesis Core: Bohr + Farber + Gould (feasibility, claims discipline, narrative)
- Theory Core: Feynman + Shannon + Greider (concepts, abstractions, mechanism)
- Implementation Core: Dayhoff + Hinton + Bayes (pipelines, compute, stats)
"""

from typing import Any, Optional, Literal
from pathlib import Path

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


CoreType = Literal["synthesis", "theory", "implementation"]


class CoreInputModule(Module):
    """
    CORE_INPUT: Get coherent answer from core subgroup.
    
    Cores represent complementary expertise groups:
    
    - Synthesis Core: Bohr + Farber + Gould
      For: feasibility assessment, claims discipline, narrative quality
      
    - Theory Core: Feynman + Shannon + Greider
      For: conceptual questions, abstractions, mechanism
      
    - Implementation Core: Dayhoff + Hinton + Bayes
      For: pipeline design, computation, statistics
    
    The module orchestrates a focused discussion within the core
    and produces a consolidated recommendation.
    
    Outputs:
        - Consolidated core recommendation
        - Points of agreement
        - Points of disagreement
        - Suggested approach
    """
    
    name = "core_input"
    description = "Get coherent answer from expert core subgroup"
    module_type = ModuleType.SUBGRAPH
    
    required_inputs = ["core_type", "question", "context"]
    optional_inputs = ["artifacts"]
    expected_outputs = ["recommendation", "agreements", "disagreements", "approach"]
    
    CORES = {
        "synthesis": {
            "agents": ["bohr", "farber", "gould"],
            "focus": "feasibility, claims discipline, narrative quality",
        },
        "theory": {
            "agents": ["feynman", "shannon", "greider"],
            "focus": "concepts, abstractions, mechanisms",
        },
        "implementation": {
            "agents": ["dayhoff", "hinton", "bayes"],
            "focus": "pipelines, computation, statistics",
        },
    }
    
    primary_agents = ["bohr"]
    supporting_agents = ["feynman", "hinton", "dayhoff", "bayes", "shannon", "greider", "gould", "farber"]
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """Execute core input consultation."""
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        core_type = inputs["core_type"].lower()
        if core_type not in self.CORES:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Invalid core type '{core_type}'. Must be one of: {list(self.CORES.keys())}",
            )
        
        core_info = self.CORES[core_type]
        core_agents = [a for a in core_info["agents"] if a in self.agents]
        
        if len(core_agents) < 2:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Insufficient agents available for {core_type} core",
            )
        
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = ModuleStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "core_type": core_type,
                "core_agents": core_agents,
                "question": inputs["question"],
                "context": inputs["context"],
                "responses": {},
            }
        
        self._log_step(f"Starting {core_type} core consultation")
        console.info(f"Consulting {core_type.upper()} CORE: {', '.join(a.title() for a in core_agents)}")
        
        try:
            # Step 1: Each core member responds
            if self._current_step <= 0:
                self._log_step("Step 1: Gathering core member perspectives")
                
                for agent_name in core_agents:
                    response = await self._run_agent_task(
                        agent_name=agent_name,
                        task=f"""You are part of the {core_type.upper()} CORE.
Core focus: {core_info['focus']}
Core members: {', '.join(core_agents)}

Question:
{inputs['question']}

Context:
{inputs['context'][:1500]}

Provide your expert perspective on this question.
Be concise (2-3 paragraphs) but substantive.
Note any concerns, assumptions, or prerequisites.""",
                    )
                    
                    self._state["responses"][agent_name] = response.get("response", "")
                    console.agent_message(agent_name.upper(), self._state["responses"][agent_name])
                
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Identify agreements and disagreements
            if self._current_step <= 1:
                self._log_step("Step 2: Synthesizing perspectives")
                
                responses_text = "\n\n".join([
                    f"=== {agent.upper()} ===\n{resp}"
                    for agent, resp in self._state["responses"].items()
                ])
                
                # Use the first core agent to synthesize (or bohr if available)
                synthesizer = "bohr" if "bohr" in core_agents else core_agents[0]
                
                synthesis_result = await self._run_agent_task(
                    agent_name=synthesizer,
                    task=f"""Synthesize the core members' perspectives.

Question:
{inputs['question']}

Core Member Responses:
{responses_text}

Create a synthesis including:
1. POINTS OF AGREEMENT: What all members agree on
2. POINTS OF DISAGREEMENT: Where views differ
3. CONSOLIDATED RECOMMENDATION: The core's unified answer
4. SUGGESTED APPROACH: Recommended next steps

Aim for consensus where possible, but note genuine disagreements.""",
                )
                
                self._state["synthesis"] = synthesis_result.get("response", "")
                self._current_step = 2
            
            self._status = ModuleStatus.COMPLETED
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "recommendation": self._state.get("synthesis", ""),
                    "agreements": "",  # Embedded in synthesis
                    "disagreements": "",  # Embedded in synthesis
                    "approach": "",  # Embedded in synthesis
                    "core_type": core_type,
                    "core_agents": core_agents,
                    "individual_responses": self._state.get("responses", {}),
                },
                summary=f"{core_type.title()} core consultation complete.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
