"""
LITERATURE REVIEW Workflow Module.

Background research phase for gathering context, prior work,
and methodological guidance. Led by Gould (librarian/scholar).
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

from .base import WorkflowModule, WorkflowResult, WorkflowCheckpoint, WorkflowStatus
from ..utils import console


@dataclass
class LiteratureEntry:
    """A single literature reference."""
    title: str
    authors: list[str]
    year: int
    source: str  # journal, arxiv, etc.
    abstract: str
    relevance: str  # Why this is relevant
    key_findings: list[str]
    pmid: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None


class LiteratureReviewModule(WorkflowModule):
    """
    LITERATURE REVIEW: Background research and context gathering.
    
    Supports two modes:
    - QUICK: 3 steps, ~10 papers, 1 agent (Gould)
    - COMPREHENSIVE: 7 steps, ~30 papers, 3 agents (Gould, Farber, Feynman)
    
    Mode is determined by token budget:
    - Budget < 300K: Quick mode
    - Budget >= 300K: Comprehensive mode
    
    Purpose:
        - Search relevant literature (PubMed, arXiv)
        - Identify key prior work and methodologies
        - Build bibliography for the project
        - Synthesize background knowledge
        - Identify gaps and opportunities
    
    Primary Agent: Gould (librarian, citation expert)
    Supporting: Farber (critical assessment), Feynman (technical papers)
    
    Outputs:
        - bibliography: Formatted reference list
        - literature_summary: Synthesis of key findings
        - methodology_notes: Relevant methods identified (comprehensive only)
        - knowledge_gaps: Identified gaps in literature (comprehensive only)
    """
    
    name = "literature_review"
    description = "Background research and literature synthesis"
    
    required_inputs = ["research_topic", "project_spec"]
    optional_inputs = ["specific_queries", "max_papers", "target_citations", "token_budget", "review_mode"]
    expected_outputs = ["bibliography", "literature_summary", "methodology_notes", "knowledge_gaps"]
    
    primary_agents = ["gould"]
    supporting_agents = ["farber", "feynman"]
    
    # Mode constants
    MODE_QUICK = "quick"
    MODE_COMPREHENSIVE = "comprehensive"
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[WorkflowCheckpoint] = None,
    ) -> WorkflowResult:
        """
        Execute literature review workflow.
        
        Quick Mode (3 steps, ~5-10 papers):
        1. Generate search queries + Search both PubMed and arXiv
        2. Synthesize findings into summary
        3. Compile bibliography
        
        Comprehensive Mode (7 steps, ~20-30 papers):
        1. Generate search queries from topic
        2. Search PubMed for biomedical literature
        3. Search arXiv for computational methods
        4. Critical assessment of relevance (Farber)
        5. Technical methodology analysis (Feynman)
        6. Synthesize findings into summary
        7. Compile bibliography
        """
        # Validate inputs
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        # Determine mode based on token budget or explicit setting
        token_budget = inputs.get("token_budget", 500_000)
        review_mode = inputs.get("review_mode")
        
        if review_mode:
            mode = review_mode
        elif token_budget and token_budget < 300_000:
            mode = self.MODE_QUICK
        else:
            mode = self.MODE_COMPREHENSIVE
        
        # Restore or initialize state
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = WorkflowStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "research_topic": inputs["research_topic"],
                "project_spec": inputs["project_spec"],
                "mode": mode,
                "search_queries": [],
                "pubmed_results": [],
                "arxiv_results": [],
                "analyzed_papers": [],
                "bibliography": [],
            }
        
        # Set paper limits based on mode
        if mode == self.MODE_QUICK:
            max_papers = inputs.get("max_papers", 10)
            self._log_step(f"Starting QUICK literature review (~{max_papers} papers)")
            return await self._execute_quick_mode(inputs, max_papers)
        else:
            max_papers = inputs.get("max_papers", 25)
            self._log_step(f"Starting COMPREHENSIVE literature review (~{max_papers} papers)")
            return await self._execute_comprehensive_mode(inputs, max_papers)
    
    async def _execute_quick_mode(
        self, 
        inputs: dict[str, Any],
        max_papers: int
    ) -> WorkflowResult:
        """
        Execute quick 3-step literature review.
        
        Single agent (Gould), combined searches, faster synthesis.
        """
        from ..utils import console
        
        try:
            # Step 1: Generate queries and search both sources in one task
            if self._current_step <= 0:
                console.info("Quick review: Searching PubMed and arXiv")
                
                search_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Conduct a focused literature search on this topic:

Research Topic: {inputs['research_topic']}

Project Context:
{inputs['project_spec'][:1000]}

Instructions:
1. Generate 2-3 targeted search queries
2. Use pubmed.search for each query (max_results=5 per query)
3. Use arxiv.search for each query (max_results=5 per query)
4. Emphasize recent publications (last 3-5 years) but include key foundational works
5. For each paper, capture: Title, Authors, Year, PMID/arXiv ID, and key relevance

Target: {max_papers} high-quality papers total. Focus on the MOST relevant work.""",
                )
                
                self._state["combined_results"] = search_result.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Quick synthesis
            if self._current_step <= 1:
                console.info("Quick review: Synthesizing findings")
                
                synthesis_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Create a concise literature summary.

Research Topic: {inputs['research_topic']}

Search Results:
{self._state['combined_results']}

Write a focused summary (3-5 paragraphs) that:
1. Introduces the key research area
2. Highlights the most important findings from relevant papers
3. Notes any obvious gaps or opportunities
4. Briefly mentions relevant methodological approaches

Keep it concise but informative - this is a quick review, not a comprehensive analysis.""",
                )
                
                self._state["literature_summary"] = synthesis_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Quick bibliography
            if self._current_step <= 2:
                console.info("Quick review: Compiling bibliography")
                
                bib_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Compile a brief bibliography from the search results.

IMPORTANT: Only include papers that were actually found during searches. Do NOT invent references.

Search Results:
{self._state['combined_results']}

Format as Markdown with:
- Author list, Year, Title
- Journal/venue or arXiv ID
- PMID or DOI where available

Sort by relevance to our research topic.""",
                )
                
                self._state["bibliography"] = bib_result.get("response", "")
                self._current_step = 3
            
            # Write outputs
            return self._write_quick_outputs(inputs)
            
        except Exception as e:
            self._status = WorkflowStatus.FAILED
            self._log_step(f"Error: {str(e)}")
            self.save_checkpoint()
            return WorkflowResult(status=WorkflowStatus.FAILED, error=str(e))
    
    async def _execute_comprehensive_mode(
        self,
        inputs: dict[str, Any], 
        max_papers: int
    ) -> WorkflowResult:
        """
        Execute comprehensive 7-step literature review.
        
        Multiple agents, thorough assessment, detailed methodology analysis.
        """
        
        target_citations = inputs.get("target_citations", max_papers)
        
        from ..utils import console
        
        try:
            # Step 1: Generate search queries (Gould)
            if self._current_step <= 0:
                console.info("Step 1/7: Generating search queries")
                
                query_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Generate search queries for a literature review on this topic:

Research Topic: {inputs['research_topic']}

Project Context:
{inputs['project_spec']}

Generate:
1. 3-5 PubMed search queries (MeSH terms preferred)
2. 3-5 arXiv search queries (cs.LG, stat.ML, q-bio relevant)

Format each query on a new line, prefixed with [PUBMED] or [ARXIV].""",
                )
                
                # Parse queries from response
                response = query_result.get("response", "")
                pubmed_queries = []
                arxiv_queries = []
                
                for line in response.split("\n"):
                    if "[PUBMED]" in line:
                        query = line.replace("[PUBMED]", "").strip()
                        if query:
                            pubmed_queries.append(query)
                    elif "[ARXIV]" in line:
                        query = line.replace("[ARXIV]", "").strip()
                        if query:
                            arxiv_queries.append(query)
                
                self._state["pubmed_queries"] = pubmed_queries or [inputs["research_topic"]]
                self._state["arxiv_queries"] = arxiv_queries or [inputs["research_topic"]]
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Search PubMed (Gould uses pubmed tool)
            if self._current_step <= 1:
                console.info("Step 2/7: Searching PubMed")
                
                pubmed_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Search PubMed for relevant papers using these queries.

IMPORTANT: You MUST use the pubmed.search tool for each query. Do not summarize - actually call the tool.

Queries to search:
{chr(10).join(self._state['pubmed_queries'])}

Instructions:
1. Use pubmed.search for each query (max_results=10 per query)
2. Emphasize recent publications (last 3-5 years) while including seminal/foundational works
3. Use pubmed.fetch to get abstracts for the most relevant papers
4. For each paper found, provide: Title, Authors, Year, PMID, and Abstract

Target: Retrieve up to {max_papers // 2} high-quality papers total.""",
                )
                
                # Validate that actual tool calls were made
                response = pubmed_result.get("response", "")
                tool_calls = pubmed_result.get("tool_calls", [])
                if not tool_calls and "PMID:" not in response and "pmid" not in response.lower():
                    self._log_step("WARNING: PubMed search may not have used tools - results may be incomplete")
                
                self._state["pubmed_results"] = response
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Search arXiv (Gould uses arxiv tool)
            if self._current_step <= 2:
                console.info("Step 3/7: Searching arXiv")
                
                arxiv_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Search arXiv for relevant computational/methods papers.

IMPORTANT: You MUST use the arxiv.search tool for each query. Do not summarize - actually call the tool.

Queries to search:
{chr(10).join(self._state['arxiv_queries'])}

Instructions:
1. Use arxiv.search for each query (max_results=10 per query)
2. Focus on recent preprints but include influential older works
3. Categories: cs.LG, stat.ML, q-bio.* are most relevant
4. For each paper found, provide: Title, Authors, Year, arXiv ID, and Abstract

Target: Retrieve up to {max_papers // 2} high-quality papers.""",
                )
                
                # Validate that actual tool calls were made
                response = arxiv_result.get("response", "")
                tool_calls = arxiv_result.get("tool_calls", [])
                if not tool_calls and "arxiv:" not in response.lower() and "arXiv:" not in response:
                    self._log_step("WARNING: arXiv search may not have used tools - results may be incomplete")
                
                self._state["arxiv_results"] = response
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Critical assessment of relevance (Farber)
            if self._current_step <= 3:
                console.info("Step 4/7: Critical assessment (Farber)")
                
                assessment_result = await self._run_agent_task(
                    agent_name="farber",
                    task=f"""Critically assess these literature search results for relevance
to our research topic.

Research Topic: {inputs['research_topic']}

PubMed Results:
{self._state['pubmed_results']}

arXiv Results:
{self._state['arxiv_results']}

For each paper:
1. Rate relevance (High/Medium/Low)
2. Identify key contributions
3. Note methodological approaches
4. Flag any concerns (outdated, superseded, etc.)

Prioritize the most relevant papers for detailed review.""",
                )
                
                self._state["assessment"] = assessment_result.get("response", "")
                self._current_step = 4
                self.save_checkpoint()
            
            # Step 5: Technical methodology analysis (Feynman)
            if self._current_step <= 4:
                console.info("Step 5/7: Methodology analysis (Feynman)")
                
                methods_result = await self._run_agent_task(
                    agent_name="feynman",
                    task=f"""Analyze the methodological approaches in this literature.

Literature Assessment:
{self._state['assessment']}

Identify:
1. Common methodological approaches
2. State-of-the-art techniques
3. Computational requirements
4. Potential pitfalls mentioned
5. Methods we should consider for our project

Focus on practical implementation insights.""",
                )
                
                self._state["methodology_notes"] = methods_result.get("response", "")
                self._current_step = 5
                self.save_checkpoint()
            
            # Step 6: Synthesize literature summary (Gould)
            if self._current_step <= 5:
                console.info("Step 6/7: Synthesizing findings")
                
                synthesis_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Synthesize a comprehensive literature summary.

Research Topic: {inputs['research_topic']}

Search Results:
{self._state['pubmed_results']}
{self._state['arxiv_results']}

Critical Assessment:
{self._state['assessment']}

Methodological Analysis:
{self._state['methodology_notes']}

Write a literature review summary that:
1. Introduces the research landscape
2. Summarizes key findings from relevant papers
3. Identifies consensus and controversies
4. Notes gaps in current knowledge
5. Recommends directions for our project

This will be saved to the project documentation.""",
                )
                
                self._state["literature_summary"] = synthesis_result.get("response", "")
                self._current_step = 6
                self.save_checkpoint()
            
            # Step 7: Compile bibliography (Gould)
            if self._current_step <= 6:
                console.info("Step 7/7: Compiling bibliography")
                
                bib_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Compile a formatted bibliography from the literature review.

IMPORTANT: Only include papers that were actually found during the searches above.
Do NOT invent or hallucinate references.

Include papers from:
{self._state['pubmed_results']}
{self._state['arxiv_results']}

Format as a Markdown reference list with:
- Full author list
- Publication year
- Title
- Journal/venue (or arXiv ID)
- DOI or PMID where available
- URL link where available

Group references by source (PubMed, arXiv) and sort by relevance to our research.""",
                )
                
                self._state["bibliography"] = bib_result.get("response", "")
                self._current_step = 7
            
            # Write outputs to files
            output_dir = self.project_path / "literature"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Literature summary
            summary_path = output_dir / "literature_summary.md"
            with open(summary_path, "w") as f:
                f.write(f"# Literature Review: {inputs['research_topic']}\n\n")
                f.write(self._state["literature_summary"])
            
            # Methodology notes
            methods_path = output_dir / "methodology_notes.md"
            with open(methods_path, "w") as f:
                f.write("# Methodological Analysis\n\n")
                f.write(self._state["methodology_notes"])
            
            # Bibliography (now Markdown)
            bib_path = output_dir / "references.md"
            with open(bib_path, "w") as f:
                f.write(f"# References: {inputs['research_topic']}\n\n")
                f.write("*Bibliography generated by MiniLab*\n\n")
                f.write(self._state["bibliography"])
            
            self._status = WorkflowStatus.COMPLETED
            self._log_step("Literature review completed successfully")
            
            # Count actual references (look for common reference patterns)
            bib_content = self._state.get("bibliography", "")
            ref_count = max(
                bib_content.count("PMID:") + bib_content.count("arXiv:"),
                bib_content.count("- "),  # Markdown list items
                len([line for line in bib_content.split('\n') if line.strip().startswith(('1.', '2.', '3.'))])
            )
            
            return WorkflowResult(
                status=WorkflowStatus.COMPLETED,
                outputs={
                    "bibliography": self._state["bibliography"],
                    "literature_summary": self._state["literature_summary"],
                    "methodology_notes": self._state["methodology_notes"],
                    "knowledge_gaps": self._state.get("assessment", ""),
                },
                artifacts=[str(summary_path), str(methods_path), str(bib_path)],
                summary=f"Literature review complete. {ref_count} references compiled.",
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
    
    def _write_quick_outputs(self, inputs: dict[str, Any]) -> WorkflowResult:
        """Write outputs for quick mode (consolidated single file)."""
        output_dir = self.project_path / "literature"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Single consolidated literature file for quick mode
        lit_path = output_dir / "literature_summary.md"
        with open(lit_path, "w") as f:
            f.write(f"# Literature Review: {inputs['research_topic']}\n\n")
            f.write("*Quick review mode - for a comprehensive review, use a larger token budget*\n\n")
            f.write("## Summary\n\n")
            f.write(self._state.get("literature_summary", ""))
            f.write("\n\n## References\n\n")
            f.write(self._state.get("bibliography", ""))
        
        self._status = WorkflowStatus.COMPLETED
        self._log_step("Quick literature review completed")
        
        # Count references
        bib_content = self._state.get("bibliography", "")
        ref_count = max(
            bib_content.count("PMID:") + bib_content.count("arXiv:"),
            bib_content.count("- "),
            1  # At least 1 if there's any content
        )
        
        return WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            outputs={
                "bibliography": self._state.get("bibliography", ""),
                "literature_summary": self._state.get("literature_summary", ""),
                "methodology_notes": "Quick mode - methodology analysis not performed",
                "knowledge_gaps": "Quick mode - gap analysis not performed",
            },
            artifacts=[str(lit_path)],
            summary=f"Quick literature review complete. ~{ref_count} references found.",
        )
