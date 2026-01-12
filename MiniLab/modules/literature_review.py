"""
LITERATURE_REVIEW Module.

Background research phase for gathering context, prior work,
and methodological guidance. Led by Gould (librarian/scholar).

This module combines search and evidence packet creation.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


@dataclass
class LiteratureEntry:
    """A single literature reference."""
    title: str
    authors: list[str]
    year: int
    source: str
    abstract: str
    relevance: str
    key_findings: list[str]
    pmid: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None


class LiteratureReviewModule(Module):
    """
    LITERATURE_REVIEW: Background research and context gathering.
    
    Task from outline:
    - Gather, filter, and synthesize literature
    - Produce evidence-backed narrative
    
    Scope and depth are AGENT-DRIVEN based on project needs.
    Gould assesses topic complexity and decides how thorough to be.
    
    Outputs:
        - artifacts/evidence.md: Evidence packets with citations
        - reports/review.md: Synthesis narrative
        - memory/sources/: Bibliography records
    """
    
    name = "literature_review"
    description = "Background research and literature synthesis"
    module_type = ModuleType.SUBGRAPH
    
    required_inputs = ["research_topic", "project_spec"]
    optional_inputs = ["specific_queries", "max_papers", "target_citations", "token_budget", "review_mode", "user_preferences"]
    expected_outputs = ["bibliography", "literature_summary", "methodology_notes", "knowledge_gaps"]
    
    primary_agents = ["gould"]
    supporting_agents = ["farber", "feynman"]
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """
        Execute literature review module.
        
        AGENT-DRIVEN: Gould assesses the topic and decides scope dynamically.
        """
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        token_budget = inputs.get("token_budget")
        user_preferences = inputs.get("user_preferences", "")
        explicit_mode = inputs.get("review_mode")
        
        self._init_budget_tracking(token_budget)
        
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = ModuleStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "research_topic": inputs["research_topic"],
                "project_spec": inputs["project_spec"],
                "user_preferences": user_preferences,
                "token_budget": token_budget,
                "search_queries": [],
                "pubmed_results": [],
                "arxiv_results": [],
                "analyzed_papers": [],
                "bibliography": [],
            }
        
        return await self._execute_agent_driven_review(inputs, explicit_mode)
    
    async def _execute_agent_driven_review(
        self, 
        inputs: dict[str, Any],
        explicit_mode: Optional[str] = None
    ) -> ModuleResult:
        """Execute literature review with agent-driven scope decisions."""
        try:
            # Step 1: Gould assesses and plans
            if self._current_step <= 0:
                console.info("Gould assessing topic and planning review scope...")
                
                user_prefs = self._state.get("user_preferences", "")
                budget_context = ""
                if self._state.get("token_budget"):
                    module_budget = int(self._state['token_budget'] * self.get_budget_allocation(self.name))
                    budget_context = f"\nBUDGET CONTEXT: ~{module_budget:,} tokens allocated."
                
                planning_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Assess this research topic and plan an appropriate literature review.

Research Topic: {inputs['research_topic']}

Project Context:
{inputs['project_spec'][:1500]}

User Preferences: {user_prefs if user_prefs else "None specified"}
{budget_context}
{"User explicitly requested: " + explicit_mode + " mode" if explicit_mode else ""}

Plan your approach and output as JSON:
```json
{{"num_search_queries": N, "papers_per_query": N, "include_methodology_analysis": bool, "include_critical_assessment": bool, "search_sources": ["pubmed", "arxiv"], "queries": [...]}}
```""",
                )
                
                plan = self._parse_review_plan(planning_result.get("response", ""))
                self._state["review_plan"] = plan
                self._state["search_queries"] = plan.get("queries", [])
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Execute searches
            if self._current_step <= 1:
                plan = self._state.get("review_plan", {})
                queries = self._state.get("search_queries", [])
                sources = plan.get("search_sources", ["pubmed", "arxiv"])
                papers_per_query = plan.get("papers_per_query", 5)
                
                console.info(f"Searching {len(queries)} queries across {', '.join(sources)}...")
                
                search_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Execute these literature searches:

Queries: {queries}
Sources to search: {sources}
Results per query: {papers_per_query}

For each query, search and capture: Title, Authors, Year, ID, relevance note.""",
                )
                
                self._state["search_results"] = search_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Optional methodology analysis
            plan = self._state.get("review_plan", {})
            within_budget, budget_pct = self._check_module_budget()
            
            if self._current_step <= 2 and plan.get("include_methodology_analysis", False):
                if budget_pct >= 90:
                    console.info("Methodology analysis skipped (budget critical)")
                    self._state["methodology_notes"] = "(Skipped - budget critical)"
                else:
                    console.info("Feynman analyzing methodological papers...")
                    
                    budget_guidance = self._get_budget_guidance()
                    method_result = await self._run_agent_task(
                        agent_name="feynman",
                        task=f"""Review methodological aspects of these papers.
{budget_guidance}
Research Topic: {inputs['research_topic']}

Papers Found:
{self._state['search_results'][:4000]}

Identify key computational/statistical methods and their relevance.""",
                    )
                    
                    self._state["methodology_notes"] = method_result.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            elif self._current_step <= 2:
                self._current_step = 3
            
            # Step 4: Optional critical assessment
            within_budget, budget_pct = self._check_module_budget()
            
            if self._current_step <= 3 and plan.get("include_critical_assessment", False):
                if budget_pct >= 90:
                    console.info("Critical assessment skipped (budget critical)")
                    self._state["critical_assessment"] = "(Skipped - budget critical)"
                else:
                    console.info("Farber providing critical assessment...")
                    
                    budget_guidance = self._get_budget_guidance()
                    critique_result = await self._run_agent_task(
                        agent_name="farber",
                        task=f"""Critically assess the relevance and quality of this literature.
{budget_guidance}

Research Topic: {inputs['research_topic']}

Papers Found:
{self._state['search_results'][:3000]}

Assess coverage and note any gaps or quality concerns.""",
                    )
                    
                    self._state["critical_assessment"] = critique_result.get("response", "")
                self._current_step = 4
                self.save_checkpoint()
            elif self._current_step <= 3:
                self._current_step = 4
            
            # Step 5: Synthesize
            if self._current_step <= 4:
                console.info("Synthesizing findings...")
                
                extra_context = ""
                if self._state.get("methodology_notes"):
                    extra_context += f"\n\nMethodology Analysis:\n{self._state['methodology_notes']}"
                if self._state.get("critical_assessment"):
                    extra_context += f"\n\nCritical Assessment:\n{self._state['critical_assessment']}"
                
                synthesis_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Synthesize the literature into a coherent summary.

Research Topic: {inputs['research_topic']}

Search Results:
{self._state['search_results'][:4000]}
{extra_context}

Create:
1. Executive summary (2-3 paragraphs)
2. Key findings organized by theme
3. Identified gaps in the literature
4. Formatted bibliography""",
                )
                
                self._state["synthesis"] = synthesis_result.get("response", "")
                self._current_step = 5
            
            # Write outputs
            self._write_outputs()
            
            self._status = ModuleStatus.COMPLETED
            self._record_module_usage()
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "bibliography": self._state.get("search_results", ""),
                    "literature_summary": self._state.get("synthesis", ""),
                    "methodology_notes": self._state.get("methodology_notes", ""),
                    "knowledge_gaps": self._state.get("critical_assessment", ""),
                },
                artifacts=[
                    str(self.project_path / "artifacts" / "evidence.md"),
                    str(self.project_path / "reports" / "literature_review.md"),
                ],
                summary="Literature review complete.",
            )
            
        except Exception as e:
            self._record_module_usage()
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
    
    def _parse_review_plan(self, response: str) -> dict:
        """Parse Gould's review plan from response."""
        import json
        import re
        
        default_plan = {
            "num_search_queries": 3,
            "papers_per_query": 5,
            "include_methodology_analysis": False,
            "include_critical_assessment": False,
            "search_sources": ["pubmed", "arxiv"],
            "queries": ["general topic search"],
        }
        
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            json_match = re.search(r'\{[^{}]*"queries"[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return default_plan
    
    def _write_outputs(self) -> None:
        """Write outputs to appropriate directories."""
        # Create directories
        artifacts_dir = self.project_path / "artifacts"
        reports_dir = self.project_path / "reports"
        memory_dir = self.project_path / "memory" / "sources"
        
        for d in [artifacts_dir, reports_dir, memory_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Write evidence.md
        with open(artifacts_dir / "evidence.md", "w") as f:
            f.write("# Evidence Packets\n\n")
            f.write("## Search Results\n\n")
            f.write(self._state.get("search_results", "") + "\n\n")
            if self._state.get("methodology_notes"):
                f.write("## Methodology Notes\n\n")
                f.write(self._state.get("methodology_notes", "") + "\n\n")
            if self._state.get("critical_assessment"):
                f.write("## Critical Assessment\n\n")
                f.write(self._state.get("critical_assessment", "") + "\n")
        
        # Write literature review report
        with open(reports_dir / "literature_review.md", "w") as f:
            f.write("# Literature Review\n\n")
            f.write(self._state.get("synthesis", ""))
