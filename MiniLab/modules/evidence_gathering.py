"""
EVIDENCE_GATHERING Module.

Subgraph for consistent searching and evidence packet creation.
"""

from typing import Any, Optional
from pathlib import Path
import json

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


class EvidenceGatheringModule(Module):
    """
    EVIDENCE_GATHERING: Search + evidence packet creation.
    
    Subgraph pattern:
    1. Generate search plan (queries + inclusion/exclusion)
    2. Run search tools
    3. Triage results (relevance, recency, quality)
    4. Write artifacts/evidence.md (claims → supporting sources)
    5. Create/update bibliography records in memory/sources/
    
    Primary Agent: Gould
    
    Outputs:
        - artifacts/evidence.md: Claims to sources mapping
        - memory/sources/: Bibliography records
        - Search results summary
    """
    
    name = "evidence_gathering"
    description = "Consistent searching and evidence packet creation"
    module_type = ModuleType.SUBGRAPH
    
    required_inputs = ["research_questions"]
    optional_inputs = ["inclusion_criteria", "exclusion_criteria", "max_results", "sources"]
    expected_outputs = ["evidence_packets", "bibliography", "search_summary"]
    
    primary_agents = ["gould"]
    supporting_agents = ["farber"]
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """Execute evidence gathering."""
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
                "research_questions": inputs["research_questions"],
                "inclusion_criteria": inputs.get("inclusion_criteria", ""),
                "exclusion_criteria": inputs.get("exclusion_criteria", ""),
                "max_results": inputs.get("max_results", 20),
                "sources": inputs.get("sources", ["pubmed", "arxiv"]),
                "search_queries": [],
                "raw_results": [],
                "triaged_results": [],
                "evidence_packets": [],
            }
        
        self._log_step("Starting evidence gathering")
        
        try:
            # Step 1: Generate search plan
            if self._current_step <= 0:
                self._log_step("Step 1: Generating search plan")
                
                plan_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Create a search plan for gathering evidence.

Research Questions:
{json.dumps(inputs['research_questions'], indent=2) if isinstance(inputs['research_questions'], list) else inputs['research_questions']}

Inclusion Criteria: {self._state['inclusion_criteria'] or 'Not specified'}
Exclusion Criteria: {self._state['exclusion_criteria'] or 'Not specified'}
Max Results: {self._state['max_results']}
Sources: {', '.join(self._state['sources'])}

Generate:
1. Search queries (3-7 queries)
2. For each query: target source, expected result type
3. Triage criteria for relevance assessment

Output as JSON:
```json
{{"queries": [{{"query": "...", "source": "pubmed/arxiv", "rationale": "..."}}], "triage_criteria": ["..."]}}
```""",
                )
                
                # Parse search plan
                response = plan_result.get("response", "")
                try:
                    import re
                    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                    if json_match:
                        plan = json.loads(json_match.group(1))
                    else:
                        plan = {"queries": [{"query": inputs["research_questions"], "source": "pubmed"}]}
                    
                    self._state["search_plan"] = plan
                    self._state["search_queries"] = plan.get("queries", [])
                except (json.JSONDecodeError, Exception):
                    self._state["search_queries"] = [{"query": str(inputs["research_questions"]), "source": "pubmed"}]
                
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Run searches
            if self._current_step <= 1:
                self._log_step("Step 2: Executing searches")
                
                queries_text = "\n".join([
                    f"- [{q.get('source', 'pubmed')}] {q.get('query', '')}"
                    for q in self._state.get("search_queries", [])
                ])
                
                search_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Execute these literature searches.

Search Queries:
{queries_text}

Use the search tools (pubmed, arxiv) to find relevant papers.
For each result capture:
- Title, Authors, Year
- PMID/arXiv ID/DOI
- Abstract snippet
- Initial relevance assessment""",
                )
                
                self._state["raw_results"] = search_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Triage results
            if self._current_step <= 2:
                self._log_step("Step 3: Triaging results")
                
                triage_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Triage the search results for relevance and quality.

Research Questions:
{inputs['research_questions']}

Search Results:
{self._state['raw_results'][:4000]}

Inclusion Criteria: {self._state['inclusion_criteria'] or 'Relevant to research questions'}
Exclusion Criteria: {self._state['exclusion_criteria'] or 'None specified'}

For each result:
1. Assess relevance (high/medium/low)
2. Assess quality (high/medium/low)
3. Note key findings relevant to research questions
4. Decide: INCLUDE or EXCLUDE

Return triaged list with rationale.""",
                )
                
                self._state["triaged_results"] = triage_result.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Create evidence packets
            if self._current_step <= 3:
                self._log_step("Step 4: Creating evidence packets")
                
                packets_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Create evidence packets mapping claims to sources.

Research Questions:
{inputs['research_questions']}

Triaged Results:
{self._state['triaged_results'][:4000]}

For each research question, create an evidence packet:
1. State the claim/question
2. List supporting sources with:
   - Citation (Author, Year, Title)
   - Key finding from that source
   - Strength of evidence (strong/moderate/weak)
3. Identify any gaps in evidence

Format as structured markdown for artifacts/evidence.md""",
                )
                
                self._state["evidence_packets"] = packets_result.get("response", "")
                self._current_step = 4
            
            # Write outputs
            self._write_outputs()
            
            self._status = ModuleStatus.COMPLETED
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "evidence_packets": self._state.get("evidence_packets", ""),
                    "bibliography": self._state.get("raw_results", ""),
                    "search_summary": self._state.get("triaged_results", ""),
                },
                artifacts=[
                    str(self.project_path / "artifacts" / "evidence.md"),
                ],
                summary="Evidence gathering complete.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
    
    def _write_outputs(self) -> None:
        """Write evidence outputs."""
        artifacts_dir = self.project_path / "artifacts"
        memory_dir = self.project_path / "memory" / "sources"
        
        for d in [artifacts_dir, memory_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Write evidence.md
        with open(artifacts_dir / "evidence.md", "w") as f:
            f.write("# Evidence Packets\n\n")
            f.write(self._state.get("evidence_packets", ""))
            f.write("\n\n## Raw Search Results\n\n")
            f.write(self._state.get("raw_results", ""))
        
        # Write bibliography to memory
        with open(memory_dir / "bibliography.md", "w") as f:
            f.write("# Bibliography\n\n")
            f.write(self._state.get("triaged_results", ""))
