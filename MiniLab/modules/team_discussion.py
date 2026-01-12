"""
TEAM_DISCUSSION Module (formerly PlanningCommitteeModule).

Multi-agent deliberation phase for planning analysis approach.
Implements "open dialogue" protocol with context-based speaker selection.

This module corresponds to TEAM_DISCUSSION from the outline:
- Bohr runs a structured discussion with relevant agents
- Outputs risks/unknowns, dependency notes, verification requirements
- Recommends task graph structure
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import json

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from .plan_dissemination import extract_agent_responsibilities
from ..utils import console


@dataclass
class DialogueTurn:
    """A single turn in the open dialogue."""
    speaker: str
    message: str
    turn_number: int
    topic: str
    references_turns: list[int] = field(default_factory=list)


class TeamDiscussionModule(Module):
    """
    TEAM_DISCUSSION: Multi-agent deliberation on analysis approach.
    
    Goal: Structured multi-agent feedback producing risks, dependencies,
    verification requirements, and recommended task graph structure.
    
    Protocol: Open Dialogue
        - Context-based speaker selection (not round-robin)
        - LLM decides who should speak next based on conversation flow
        - Natural transitions between experts
        - Convergence toward actionable plan
    
    All Agents Participate:
        - Bohr: Facilitates, synthesizes
        - Feynman: Technical approach
        - Hinton: ML/model considerations
        - Dayhoff: Data processing
        - Bayes: Statistical rigor
        - Shannon: Information theory
        - Greider: Biological context
        - Gould: Literature grounding
        - Farber: Critical assessment
    
    Outputs:
        - analysis_plan: Detailed step-by-step plan
        - responsibilities: Agent assignments
        - decision_rationale: Why this approach
        - risks_unknowns: Identified risks
        - verification_requirements: How to verify
        - dialogue_transcript: Full deliberation record
    """
    
    name = "team_discussion"
    description = "Multi-agent deliberation for analysis planning"
    module_type = ModuleType.LINEAR
    
    required_inputs = ["project_spec"]
    optional_inputs = ["literature_summary", "constraints", "prior_attempts", "max_turns", "token_budget"]
    expected_outputs = ["analysis_plan", "responsibilities", "decision_rationale", "risks_unknowns", "dialogue_transcript"]
    
    primary_agents = ["bohr"]
    supporting_agents = ["feynman", "hinton", "dayhoff", "bayes", "shannon", "greider", "gould", "farber"]
    
    # Agent expertise mapping for speaker selection
    AGENT_EXPERTISE = {
        "bohr": ["synthesis", "coordination", "project management", "user communication"],
        "feynman": ["technical implementation", "algorithms", "computation", "debugging"],
        "hinton": ["machine learning", "neural networks", "model selection", "optimization"],
        "dayhoff": ["data processing", "pipelines", "databases", "preprocessing"],
        "bayes": ["statistics", "inference", "uncertainty", "experimental design"],
        "shannon": ["information theory", "feature selection", "dimensionality", "encoding"],
        "greider": ["biology", "mechanisms", "pathways", "biological interpretation"],
        "gould": ["literature", "citations", "prior work", "context"],
        "farber": ["critique", "limitations", "risks", "quality"],
    }
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """
        Execute team discussion module.
        
        Steps:
        1. Frame the planning problem (Bohr)
        2. Open dialogue with context-based speaker selection
        3. Iterate until consensus or max turns
        4. Synthesize into analysis plan
        5. Extract risks, verification requirements
        6. Assign responsibilities
        """
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        token_budget = inputs.get("token_budget")
        self._init_budget_tracking(token_budget)
        
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = ModuleStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "project_spec": inputs["project_spec"],
                "literature_summary": inputs.get("literature_summary")
                or "No literature review summary available yet.",
                "dialogue": [],
                "current_speaker": "bohr",
                "topics_discussed": [],
                "consensus_points": [],
                "risks_unknowns": [],
            }
        
        max_turns = inputs.get("max_turns", 15)
        if token_budget and token_budget < 300_000:
            max_turns = min(max_turns, 8)
        
        self._log_step("Starting team discussion deliberation")
        
        try:
            # Step 1: Frame the problem (Bohr opens)
            if self._current_step <= 0:
                self._log_step("Step 1: Framing the planning problem")
                
                framing_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Open a team discussion meeting for this project. BE CONCISE.

Project Specification:
{inputs['project_spec'][:2000]}

Literature Background:
{self._state.get('literature_summary', '')}

Frame the key questions we need to answer:
1. What is our overall analytical approach?
2. What data processing is needed?
3. What methods/models should we use?
4. How do we validate results?
5. What are the key risks?

Present this to the team and identify which expert should speak first.""",
                )
                
                opening = DialogueTurn(
                    speaker="bohr",
                    message=framing_result.get("response", ""),
                    turn_number=0,
                    topic="framing",
                )
                self._state["dialogue"].append(opening.__dict__)
                self._state["topics_discussed"].append("framing")
                self._current_step = 1
                self.save_checkpoint()
                
                console.agent_message("BOHR", framing_result.get("response", ""))
            
            # Step 2: Open dialogue loop
            if self._current_step <= 1:
                self._log_step("Step 2: Open dialogue deliberation")
                
                turn_number = len(self._state["dialogue"])
                
                while turn_number < max_turns:
                    within_budget, budget_pct = self._check_module_budget()
                    if budget_pct >= 90:
                        self._log_step(f"Budget limit reached at turn {turn_number}")
                        break
                    
                    next_speaker = await self._select_next_speaker()
                    
                    if next_speaker == "CONSENSUS":
                        self._log_step(f"Consensus reached at turn {turn_number}")
                        break
                    
                    dialogue_history = self._format_dialogue_history()
                    
                    budget_note = ""
                    if budget_pct >= 70:
                        budget_note = "\n\n⚠️ Budget is limited. Be VERY CONCISE (2-3 sentences max)."
                    
                    role_boundary = self._get_role_boundary(next_speaker)
                    
                    contribution = await self._run_agent_task(
                        agent_name=next_speaker,
                        task=f"""You are participating in a team discussion meeting.

Project Context:
{inputs['project_spec'][:500]}...

Discussion So Far:
{dialogue_history[-2000:]}

As {next_speaker}, provide your expert input on the current discussion.
Consider your domain expertise: {', '.join(self.AGENT_EXPERTISE.get(next_speaker, []))}

🎯 THIS IS A PLANNING DISCUSSION, NOT EXECUTION. Your job:
- Share your expert perspective on the approach being discussed
- Raise concerns or endorse suggestions from your domain expertise
- Identify risks or unknowns in your area

FORMAT: 2-4 sentences MAX. One key point, with brief rationale.
If you believe we're ready to converge on a plan, say "I believe we have consensus on...".{role_boundary}{budget_note}""",
                    )
                    
                    turn = DialogueTurn(
                        speaker=next_speaker,
                        message=contribution.get("response", ""),
                        turn_number=turn_number,
                        topic=self._extract_topic(contribution.get("response", "")),
                    )
                    self._state["dialogue"].append(turn.__dict__)
                    turn_number += 1
                    
                    console.agent_message(next_speaker.upper(), turn.message)
                    
                    # Track consensus and risks
                    if "consensus" in turn.message.lower():
                        self._state["consensus_points"].append(turn.message)
                    if any(kw in turn.message.lower() for kw in ["risk", "concern", "unknown", "uncertain"]):
                        self._state["risks_unknowns"].append(f"[{next_speaker}] {turn.message}")
                    
                    if turn_number % 3 == 0:
                        self.save_checkpoint()
                
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Synthesize analysis plan (Bohr)
            if self._current_step <= 2:
                self._log_step("Step 3: Synthesizing analysis plan")
                
                dialogue_history = self._format_dialogue_history()
                
                plan_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Synthesize the team discussion into a concrete analysis plan.

Full Discussion:
{dialogue_history}

Consensus Points:
{chr(10).join(self._state['consensus_points'])}

Identified Risks/Unknowns:
{chr(10).join(self._state['risks_unknowns'])}

Create a detailed analysis plan with:
1. OBJECTIVE: Clear statement of what we're doing
2. DATA PREPARATION: Steps for data processing
3. METHODOLOGY: Specific methods and models
4. VALIDATION: How we'll verify results
5. TIMELINE: Logical sequence with dependencies
6. EXPECTED OUTPUTS: What files will be created where
7. RISKS & MITIGATIONS: Key risks and how to address them
8. VERIFICATION REQUIREMENTS: How to verify correctness""",
                )
                
                self._state["analysis_plan"] = plan_result.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Extract responsibilities
            if self._current_step <= 3:
                self._log_step("Step 4: Extracting responsibilities")
                
                responsibilities_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Based on the analysis plan, assign specific responsibilities to each team member.

Analysis Plan:
{self._state['analysis_plan']}

For each agent who has work to do, specify:
- Agent name
- Their specific tasks
- What outputs they must produce
- Dependencies on other agents' work

Be explicit about who does what.""",
                )
                
                self._state["responsibilities"] = responsibilities_result.get("response", "")
                self._current_step = 4
            
            # Write outputs to artifacts
            self._write_outputs()
            
            self._status = ModuleStatus.COMPLETED
            self._record_module_usage()
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "analysis_plan": self._state.get("analysis_plan", ""),
                    "responsibilities": self._state.get("responsibilities", ""),
                    "decision_rationale": "\n".join(self._state.get("consensus_points", [])),
                    "risks_unknowns": self._state.get("risks_unknowns", []),
                    "dialogue_transcript": self._state.get("dialogue", []),
                },
                artifacts=[
                    str(self.project_path / "artifacts" / "decisions.md"),
                ],
                summary=f"Team discussion complete with {len(self._state['dialogue'])} turns.",
            )
            
        except Exception as e:
            self._record_module_usage()
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
    
    async def _select_next_speaker(self) -> str:
        """Select the next speaker based on dialogue context."""
        dialogue = self._state.get("dialogue", [])
        if not dialogue:
            return "bohr"
        
        last_turn = dialogue[-1]
        last_speaker = last_turn.get("speaker", "")
        last_message = last_turn.get("message", "")
        
        # Check for consensus signal
        if "consensus" in last_message.lower() and len(dialogue) > 3:
            consensus_count = sum(1 for t in dialogue[-3:] if "consensus" in t.get("message", "").lower())
            if consensus_count >= 2:
                return "CONSENSUS"
        
        # Use LLM to select next speaker
        bohr = self.agents.get("bohr")
        if not bohr:
            # Fallback to round-robin
            available = [a for a in self.AGENT_EXPERTISE.keys() if a != last_speaker]
            return available[len(dialogue) % len(available)]
        
        selection = await bohr.simple_query(
            query=f"""Based on the discussion so far, who should speak next?

Last speaker: {last_speaker}
Last message: {last_message[:200]}...

Available experts and their areas:
{json.dumps(self.AGENT_EXPERTISE, indent=2)}

Who would add most value to the discussion right now?
Return ONLY the agent name (lowercase) or "CONSENSUS" if we're ready to conclude.""",
            context="Speaker selection"
        )
        
        selection = selection.strip().lower()
        if selection in self.AGENT_EXPERTISE:
            return selection
        elif "consensus" in selection:
            return "CONSENSUS"
        else:
            available = [a for a in self.AGENT_EXPERTISE.keys() if a != last_speaker]
            return available[len(dialogue) % len(available)]
    
    def _get_role_boundary(self, agent_name: str) -> str:
        """Get role boundary reminder for agent."""
        if agent_name in ["bayes", "shannon", "greider"]:
            return "\n\n📌 ROLE: You are ADVISING on methodology. Brief rationale only."
        elif agent_name in ["hinton", "dayhoff"]:
            return "\n\n📌 ROLE: You are assessing FEASIBILITY. Note concerns briefly."
        elif agent_name in ["farber", "gould"]:
            return "\n\n📌 ROLE: You are providing CRITICAL PERSPECTIVE. Be specific but brief."
        return ""
    
    def _format_dialogue_history(self) -> str:
        """Format dialogue history for prompts."""
        lines = []
        for turn in self._state.get("dialogue", []):
            speaker = turn.get("speaker", "unknown").upper()
            message = turn.get("message", "")
            lines.append(f"[{speaker}]: {message}")
        return "\n\n".join(lines)
    
    def _extract_topic(self, message: str) -> str:
        """Extract topic from message."""
        keywords = ["data", "model", "method", "risk", "validation", "approach", "analysis"]
        for kw in keywords:
            if kw in message.lower():
                return kw
        return "general"
    
    def _write_outputs(self) -> None:
        """Write outputs to artifacts directory."""
        artifacts_dir = self.project_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Write decisions.md
        with open(artifacts_dir / "decisions.md", "w") as f:
            f.write("# Project Decisions\n\n")
            f.write("## Analysis Plan\n\n")
            f.write(self._state.get("analysis_plan", "") + "\n\n")
            f.write("## Agent Responsibilities\n\n")
            f.write(self._state.get("responsibilities", "") + "\n\n")
            f.write("## Consensus Points\n\n")
            for point in self._state.get("consensus_points", []):
                f.write(f"- {point}\n")
            f.write("\n## Risks & Unknowns\n\n")
            for risk in self._state.get("risks_unknowns", []):
                f.write(f"- {risk}\n")


# Backward compatibility alias
PlanningCommitteeModule = TeamDiscussionModule
