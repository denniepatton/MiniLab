"""
CONSULT_EXTERNAL_EXPERT Module.

Linear module for consulting external knowledge sources.
"""

from typing import Any, Optional
from pathlib import Path

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


class ConsultExternalExpertModule(Module):
    """
    CONSULT_EXTERNAL_EXPERT: Query external knowledge sources.
    
    Linear flow:
    1. Frame the expert query
    2. Search relevant sources (web, literature, docs)
    3. Synthesize findings
    4. Validate and summarize
    
    Primary Agent: Shannon (information/communication)
    Supporting: Gould (literature)
    
    Outputs:
        - Expert consultation summary
        - Source references
        - Key recommendations
    """
    
    name = "consult_external_expert"
    description = "Query external knowledge sources for expert input"
    module_type = ModuleType.LINEAR
    
    required_inputs = ["query", "domain"]
    optional_inputs = ["sources", "depth", "specific_questions"]
    expected_outputs = ["summary", "references", "recommendations"]
    
    primary_agents = ["shannon"]
    supporting_agents = ["gould"]
    
    DOMAINS = {
        "statistics": ["stack exchange", "statistical documentation", "textbooks"],
        "bioinformatics": ["biostars", "pubmed", "software documentation"],
        "machine_learning": ["arxiv", "papers with code", "framework docs"],
        "biology": ["pubmed", "textbooks", "review articles"],
        "programming": ["stack overflow", "documentation", "github"],
        "methods": ["protocol papers", "software documentation"],
    }
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """Execute external expert consultation."""
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        domain = inputs["domain"].lower()
        suggested_sources = self.DOMAINS.get(domain, ["web search", "documentation"])
        
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = ModuleStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "query": inputs["query"],
                "domain": domain,
                "sources": inputs.get("sources", suggested_sources),
                "depth": inputs.get("depth", "moderate"),
                "specific_questions": inputs.get("specific_questions", []),
            }
        
        self._log_step(f"Consulting external experts on: {inputs['query'][:50]}...")
        
        try:
            # Step 1: Frame the query
            if self._current_step <= 0:
                self._log_step("Step 1: Framing expert query")
                
                specific_q = self._state.get("specific_questions", [])
                questions_text = "\n".join(f"- {q}" for q in specific_q) if specific_q else ""
                
                frame_result = await self._run_agent_task(
                    agent_name="shannon",
                    task=f"""Frame this query for external expert consultation.

Original Query: {inputs['query']}
Domain: {domain}
Depth: {self._state['depth']}

Specific Questions:
{questions_text if questions_text else 'General consultation'}

Create:
1. A clear, searchable version of the query
2. Key terms to search for
3. Specific questions to answer
4. Success criteria (what would a good answer include?)

Make it specific enough to get useful results.""",
                )
                
                self._state["framed_query"] = frame_result.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Search sources
            if self._current_step <= 1:
                self._log_step("Step 2: Searching external sources")
                
                sources_text = ", ".join(self._state.get("sources", []))
                
                search_result = await self._run_agent_task(
                    agent_name="shannon",
                    task=f"""Search for expert information on this topic.

Framed Query:
{self._state['framed_query']}

Target Sources: {sources_text}

Use web search and literature tools to find:
1. Authoritative answers
2. Best practices
3. Common pitfalls
4. Example implementations

Document sources with enough detail to cite.""",
                )
                
                self._state["search_results"] = search_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Synthesize findings
            if self._current_step <= 2:
                self._log_step("Step 3: Synthesizing findings")
                
                synth_result = await self._run_agent_task(
                    agent_name="shannon",
                    task=f"""Synthesize the search results into actionable guidance.

Original Query: {inputs['query']}

Search Results:
{self._state['search_results'][:3000]}

Create a synthesis that:
1. Answers the original question directly
2. Provides context and rationale
3. Notes any caveats or limitations
4. Lists specific recommendations
5. Cites sources

Make it practical and actionable.""",
                )
                
                self._state["synthesis"] = synth_result.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Validate with domain expert
            if self._current_step <= 3:
                self._log_step("Step 4: Validating synthesis")
                
                validate_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Validate this expert consultation synthesis.

Original Query: {inputs['query']}
Domain: {domain}

Synthesis:
{self._state['synthesis']}

Check:
1. Are the claims well-supported?
2. Are sources credible?
3. Any important caveats missing?
4. Is the advice practical?

Provide any corrections or additions needed.""",
                )
                
                self._state["validation"] = validate_result.get("response", "")
                self._current_step = 4
            
            self._status = ModuleStatus.COMPLETED
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "summary": self._state.get("synthesis", ""),
                    "references": self._state.get("search_results", ""),
                    "recommendations": self._state.get("framed_query", ""),
                    "validation": self._state.get("validation", ""),
                },
                summary=f"External expert consultation complete for {domain}.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
