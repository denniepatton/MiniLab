"""
PLANNING COMMITTEE Workflow Module.

Multi-agent deliberation phase for planning analysis approach.
Implements "open dialogue" protocol with context-based speaker selection.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import json
import re

from .base import WorkflowModule, WorkflowResult, WorkflowCheckpoint, WorkflowStatus
from ..utils import console


@dataclass
class DialogueTurn:
    """A single turn in the open dialogue."""
    speaker: str
    message: str
    turn_number: int
    topic: str
    references_turns: list[int] = field(default_factory=list)


class PlanningCommitteeModule(WorkflowModule):
    """
    PLANNING COMMITTEE: Multi-agent deliberation on analysis approach.
    
    Purpose:
        - Deliberate on best approach to the problem
        - Each agent contributes domain expertise
        - Build consensus on methodology
        - Create detailed analysis plan
        - Assign responsibilities
    
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
        - dialogue_transcript: Full deliberation record
    """
    
    name = "planning_committee"
    description = "Multi-agent deliberation for analysis planning"
    
    required_inputs = ["project_spec", "literature_summary"]
    optional_inputs = ["constraints", "prior_attempts", "max_turns"]
    expected_outputs = ["analysis_plan", "responsibilities", "decision_rationale", "dialogue_transcript"]
    
    primary_agents = ["bohr"]  # Facilitator
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
        checkpoint: Optional[WorkflowCheckpoint] = None,
    ) -> WorkflowResult:
        """
        Execute planning committee workflow.
        
        Steps:
        1. Frame the planning problem (Bohr)
        2. Open dialogue with context-based speaker selection
        3. Iterate until consensus or max turns
        4. Synthesize into analysis plan
        5. Assign responsibilities
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
                "project_spec": inputs["project_spec"],
                "literature_summary": inputs["literature_summary"],
                "dialogue": [],
                "current_speaker": "bohr",
                "topics_discussed": [],
                "consensus_points": [],
            }
        
        max_turns = inputs.get("max_turns", 15)
        
        self._log_step("Starting planning committee deliberation")
        
        try:
            # Step 1: Frame the problem (Bohr opens)
            if self._current_step <= 0:
                self._log_step("Step 1: Framing the planning problem")
                
                framing_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Open a planning committee meeting for this project.

Project Specification:
{inputs['project_spec']}

Literature Background:
{inputs['literature_summary']}

Frame the key questions we need to answer:
1. What is our overall analytical approach?
2. What data processing is needed?
3. What methods/models should we use?
4. How do we validate results?
5. What are the key risks?

Present this to the committee and identify which expert should speak first.""",
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
            
            # Step 2: Open dialogue loop
            if self._current_step <= 1:
                self._log_step("Step 2: Open dialogue deliberation")
                
                turn_number = len(self._state["dialogue"])
                
                while turn_number < max_turns:
                    # Select next speaker based on context
                    next_speaker = await self._select_next_speaker()
                    
                    if next_speaker == "CONSENSUS":
                        self._log_step(f"Consensus reached at turn {turn_number}")
                        break
                    
                    # Get contribution from selected speaker
                    dialogue_history = self._format_dialogue_history()
                    
                    contribution = await self._run_agent_task(
                        agent_name=next_speaker,
                        task=f"""You are participating in a planning committee meeting.

Project Context:
{inputs['project_spec'][:500]}...

Discussion So Far:
{dialogue_history}

As {next_speaker}, provide your expert input on the current discussion.
Consider your domain expertise: {', '.join(self.AGENT_EXPERTISE.get(next_speaker, []))}

Be concise but substantive. Build on or respectfully disagree with previous points.
If you believe we're ready to converge on a plan, say "I believe we have consensus on...".""",
                    )
                    
                    turn = DialogueTurn(
                        speaker=next_speaker,
                        message=contribution.get("response", ""),
                        turn_number=turn_number,
                        topic=self._extract_topic(contribution.get("response", "")),
                    )
                    self._state["dialogue"].append(turn.__dict__)
                    turn_number += 1
                    
                    # Check for consensus signals
                    if "consensus" in turn.message.lower():
                        self._state["consensus_points"].append(turn.message)
                    
                    # Checkpoint every 3 turns
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
                    task=f"""Synthesize the planning committee discussion into a concrete analysis plan.

Full Discussion:
{dialogue_history}

Consensus Points:
{chr(10).join(self._state['consensus_points'])}

Create a detailed analysis plan with:
1. OBJECTIVE: Clear statement of what we're doing
2. DATA PREPARATION: Steps for data processing
3. METHODOLOGY: Specific methods and models to use
4. VALIDATION: How we'll verify results
5. TIMELINE: Logical sequence of steps
6. RISKS: Key risks and mitigations

Format as a structured plan document.""",
                )
                
                self._state["analysis_plan"] = plan_result.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Assign responsibilities
            if self._current_step <= 3:
                self._log_step("Step 4: Assigning responsibilities")
                
                assign_result = await self._run_agent_task(
                    agent_name="bohr",
                    task=f"""Based on the analysis plan, assign specific responsibilities to each agent.

Analysis Plan:
{self._state['analysis_plan']}

Available Agents and Expertise:
{json.dumps(self.AGENT_EXPERTISE, indent=2)}

Assign tasks matching expertise:
- Data processing tasks -> dayhoff
- Statistical analysis -> bayes
- ML/modeling -> hinton
- Feature engineering -> shannon
- Biological interpretation -> greider
- Documentation/references -> gould
- Quality review -> farber
- Technical implementation -> feynman
- Coordination -> bohr

List each agent with their specific assigned tasks.""",
                )
                
                self._state["responsibilities"] = assign_result.get("response", "")
                self._current_step = 4
            
            # Step 5: Decision rationale (Farber provides critical summary)
            if self._current_step <= 4:
                self._log_step("Step 5: Documenting decision rationale")
                
                rationale_result = await self._run_agent_task(
                    agent_name="farber",
                    task=f"""Document the rationale for the planning decisions made.

Analysis Plan:
{self._state['analysis_plan']}

Key Discussion Points:
{self._format_dialogue_history()[-2000:]}

Provide:
1. Why this approach was chosen
2. Alternatives that were considered
3. Key tradeoffs made
4. Remaining uncertainties
5. What could go wrong

Be honest about limitations and assumptions.""",
                )
                
                self._state["decision_rationale"] = rationale_result.get("response", "")
                self._current_step = 5
            
            # Write outputs to files
            output_dir = self.project_path / "planning"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Analysis plan
            plan_path = output_dir / "analysis_plan.md"
            with open(plan_path, "w") as f:
                f.write("# Analysis Plan\n\n")
                f.write(self._state["analysis_plan"])
            
            # Responsibilities
            resp_path = output_dir / "responsibilities.md"
            with open(resp_path, "w") as f:
                f.write("# Agent Responsibilities\n\n")
                f.write(self._state["responsibilities"])
            
            # Decision rationale
            rationale_path = output_dir / "decision_rationale.md"
            with open(rationale_path, "w") as f:
                f.write("# Decision Rationale\n\n")
                f.write(self._state["decision_rationale"])
            
            # Dialogue transcript
            transcript_path = output_dir / "planning_transcript.json"
            with open(transcript_path, "w") as f:
                json.dump(self._state["dialogue"], f, indent=2)
            
            self._status = WorkflowStatus.COMPLETED
            self._log_step("Planning committee completed successfully")
            
            return WorkflowResult(
                status=WorkflowStatus.COMPLETED,
                outputs={
                    "analysis_plan": self._state["analysis_plan"],
                    "responsibilities": self._state["responsibilities"],
                    "decision_rationale": self._state["decision_rationale"],
                    "dialogue_transcript": self._state["dialogue"],
                },
                artifacts=[str(plan_path), str(resp_path), str(rationale_path), str(transcript_path)],
                summary=f"Planning complete after {len(self._state['dialogue'])} dialogue turns.",
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
    
    async def _select_next_speaker(self) -> str:
        """
        Select next speaker using JSON-structured LLM decision.
        
        Returns agent name or "CONSENSUS" if deliberation should end.
        """
        if not self._state["dialogue"]:
            return "feynman"  # Technical lead starts after Bohr
        
        dialogue_history = self._format_dialogue_history()
        recent_speakers = [t["speaker"] for t in self._state["dialogue"][-3:]]
        
        # Use Bohr (as facilitator) to decide next speaker with JSON response
        selection_result = await self._run_agent_task(
            agent_name="bohr",
            task=f"""As meeting facilitator, decide who should speak next.

Recent Discussion:
{dialogue_history[-1500:]}

Recent Speakers: {', '.join(recent_speakers)}

Available Experts and Their Domains:
{json.dumps(self.AGENT_EXPERTISE, indent=2)}

RESPOND WITH A JSON OBJECT ONLY:
{{"next_speaker": "agent_name", "reasoning": "why this expert should speak", "consensus_reached": false}}

Or if the discussion has covered all key points:
{{"next_speaker": null, "reasoning": "why we have consensus", "consensus_reached": true}}

Rules:
- Avoid same speaker twice in a row
- Choose based on what expertise is needed next
- Set consensus_reached=true only when all major topics are addressed""",
        )
        
        response = selection_result.get("response", "").strip()
        
        # Parse JSON response
        try:
            # Handle markdown code blocks
            if "```" in response:
                json_match = re.search(r'```(?:json)?\s*(\{[^`]+\})\s*```', response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)
            
            data = json.loads(response)
            
            if data.get("consensus_reached", False):
                return "CONSENSUS"
            
            next_speaker = data.get("next_speaker", "").lower()
            if next_speaker in self.AGENT_EXPERTISE:
                return next_speaker
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: look for agent names in response
        response_lower = response.lower()
        if "consensus" in response_lower:
            return "CONSENSUS"
        
        for agent in self.AGENT_EXPERTISE:
            if agent in response_lower:
                return agent
        
        # Default to round-robin if parsing fails
        all_agents = list(self.AGENT_EXPERTISE.keys())
        last_speaker = self._state["dialogue"][-1]["speaker"] if self._state["dialogue"] else "bohr"
        try:
            idx = all_agents.index(last_speaker)
            return all_agents[(idx + 1) % len(all_agents)]
        except ValueError:
            return "feynman"
    
    def _format_dialogue_history(self) -> str:
        """Format dialogue history for context."""
        lines = []
        for turn in self._state["dialogue"]:
            lines.append(f"[{turn['speaker'].upper()}]: {turn['message']}")
            lines.append("")
        return "\n".join(lines)
    
    def _extract_topic(self, message: str) -> str:
        """Extract main topic from a message."""
        # Simple heuristic - look for key terms
        topics = {
            "data": "data_processing",
            "model": "modeling",
            "statistic": "statistics",
            "feature": "features",
            "valid": "validation",
            "risk": "risks",
            "method": "methodology",
            "literature": "literature",
            "biology": "biology",
        }
        
        message_lower = message.lower()
        for keyword, topic in topics.items():
            if keyword in message_lower:
                return topic
        
        return "general"
