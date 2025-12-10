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
        - methodology_notes: Relevant methods identified
        - knowledge_gaps: Identified gaps in literature
    """
    
    name = "literature_review"
    description = "Background research and literature synthesis"
    
    required_inputs = ["research_topic", "project_spec"]
    optional_inputs = ["specific_queries", "max_papers", "year_range"]
    expected_outputs = ["bibliography", "literature_summary", "methodology_notes", "knowledge_gaps"]
    
    primary_agents = ["gould"]
    supporting_agents = ["farber", "feynman"]
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[WorkflowCheckpoint] = None,
    ) -> WorkflowResult:
        """
        Execute literature review workflow.
        
        Steps:
        1. Generate search queries from topic
        2. Search PubMed for biomedical literature
        3. Search arXiv for computational methods
        4. Retrieve and analyze abstracts
        5. Synthesize findings into summary
        6. Identify methodological insights
        7. Compile bibliography
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
                "research_topic": inputs["research_topic"],
                "project_spec": inputs["project_spec"],
                "search_queries": [],
                "pubmed_results": [],
                "arxiv_results": [],
                "analyzed_papers": [],
                "bibliography": [],
            }
        
        max_papers = inputs.get("max_papers", 20)
        year_range = inputs.get("year_range", (2019, 2024))
        
        self._log_step(f"Starting literature review for: {inputs['research_topic']}")
        
        try:
            # Step 1: Generate search queries (Gould)
            if self._current_step <= 0:
                self._log_step("Step 1: Generating search queries")
                
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
                self._log_step("Step 2: Searching PubMed")
                
                pubmed_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Search PubMed for relevant papers using these queries.
Use the pubmed_search tool for each query.

Queries:
{chr(10).join(self._state['pubmed_queries'])}

Retrieve up to {max_papers // 2} papers total.
For each paper, get the abstract using pubmed_fetch.
List the results with title, authors, year, PMID, and abstract.""",
                )
                
                self._state["pubmed_results"] = pubmed_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Search arXiv (Gould uses arxiv tool)
            if self._current_step <= 2:
                self._log_step("Step 3: Searching arXiv")
                
                arxiv_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Search arXiv for relevant computational/methods papers.
Use the arxiv_search tool for each query.

Queries:
{chr(10).join(self._state['arxiv_queries'])}

Retrieve up to {max_papers // 2} papers.
List results with title, authors, year, arXiv ID, and abstract.""",
                )
                
                self._state["arxiv_results"] = arxiv_result.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Critical assessment of relevance (Farber)
            if self._current_step <= 3:
                self._log_step("Step 4: Critical assessment of papers")
                
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
                self._log_step("Step 5: Analyzing methodologies")
                
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
                self._log_step("Step 6: Synthesizing literature summary")
                
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
                self._log_step("Step 7: Compiling bibliography")
                
                bib_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Compile a formatted bibliography from the literature review.
Use the citation tools to format references properly.

Include all relevant papers from:
{self._state['pubmed_results']}
{self._state['arxiv_results']}

Format as BibTeX entries and also provide a human-readable reference list.
Save the BibTeX file using the filesystem tool.""",
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
            
            # Bibliography
            bib_path = output_dir / "references.bib"
            with open(bib_path, "w") as f:
                f.write("% Bibliography generated by MiniLab\n")
                f.write(f"% Topic: {inputs['research_topic']}\n\n")
                f.write(self._state["bibliography"])
            
            self._status = WorkflowStatus.COMPLETED
            self._log_step("Literature review completed successfully")
            
            return WorkflowResult(
                status=WorkflowStatus.COMPLETED,
                outputs={
                    "bibliography": self._state["bibliography"],
                    "literature_summary": self._state["literature_summary"],
                    "methodology_notes": self._state["methodology_notes"],
                    "knowledge_gaps": self._state.get("assessment", ""),
                },
                artifacts=[str(summary_path), str(methods_path), str(bib_path)],
                summary=f"Literature review complete. {len(self._state.get('bibliography', '').split('@'))-1} references compiled.",
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
