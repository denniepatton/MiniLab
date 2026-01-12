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
    
    Scope and depth are AGENT-DRIVEN based on project needs,
    not hardcoded thresholds. Gould assesses the topic complexity
    and decides how thorough to be.
    
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
        - methodology_notes: Relevant methods identified (if comprehensive)
        - knowledge_gaps: Identified gaps in literature (if comprehensive)
    """
    
    name = "literature_review"
    description = "Background research and literature synthesis"
    
    required_inputs = ["research_topic", "project_spec"]
    optional_inputs = ["specific_queries", "max_papers", "target_citations", "token_budget", "review_mode", "user_preferences"]
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
        
        AGENT-DRIVEN: Gould assesses the topic and decides scope dynamically.
        No hardcoded mode selection - the agent determines appropriate depth
        based on topic complexity, user preferences, and available budget.
        """
        # Validate inputs
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        # Extract context for agent decision-making
        token_budget = inputs.get("token_budget")
        user_preferences = inputs.get("user_preferences", "")
        explicit_mode = inputs.get("review_mode")  # Only if user explicitly requested
        
        # Initialize budget tracking for this workflow
        self._init_budget_tracking(token_budget)
        
        # Restore or initialize state
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = WorkflowStatus.IN_PROGRESS
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
        
        # Let Gould decide how to approach this review
        return await self._execute_agent_driven_review(inputs, explicit_mode)
    
    async def _execute_agent_driven_review(
        self, 
        inputs: dict[str, Any],
        explicit_mode: Optional[str] = None
    ) -> WorkflowResult:
        """
        Execute literature review with agent-driven scope decisions.
        
        Gould assesses the topic complexity and decides:
        - How many searches to run
        - How many papers to include
        - Whether to involve other agents (Farber for critique, Feynman for methods)
        - How deep to go on methodology analysis
        
        No hardcoded limits - agent reasoning determines scope.
        """
        from ..utils import console
        
        try:
            # Step 1: Gould assesses the topic and plans the review
            if self._current_step <= 0:
                console.info("Gould assessing topic and planning review scope...")
                
                user_prefs = self._state.get("user_preferences", "")
                budget_context = ""
                if self._state.get("token_budget"):
                    # Calculate workflow-specific budget dynamically
                    workflow_budget = int(self._state['token_budget'] * self.get_budget_allocation(self.name))
                    budget_context = f"""
BUDGET CONTEXT: This workflow has ~{workflow_budget:,} tokens allocated ({int(self.get_budget_allocation(self.name) * 100)}% of session budget).
Use your judgment to scale scope appropriately:
- Narrow, well-defined topics: fewer searches, focused results
- Broad, interdisciplinary topics: more comprehensive coverage
- Balance depth vs. breadth based on project needs"""
                
                planning_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Assess this research topic and plan an appropriate literature review.

Research Topic: {inputs['research_topic']}

Project Context:
{inputs['project_spec'][:1500]}

User Preferences: {user_prefs if user_prefs else "None specified"}
{budget_context}
{"User explicitly requested: " + explicit_mode + " mode" if explicit_mode else ""}

Your task:
1. Assess the COMPLEXITY of this topic:
   - Is it a narrow, well-defined area or a broad interdisciplinary topic?
   - How much prior work exists?
   - Are there specialized methodological papers we need?

2. PLAN your approach (output as JSON at the end):
   - num_search_queries: How many distinct search queries (use your judgment: 2-10 depending on topic breadth)
   - papers_per_query: Results per query (3-10 depending on topic specificity)
   - include_methodology_analysis: true/false (involve Feynman for technical papers? Your call based on project needs)
   - include_critical_assessment: true/false (involve Farber for quality review? Your call based on project needs)
   - search_sources: ["pubmed", "arxiv"] or just one if appropriate
   
3. Generate your search queries based on the topic.

End your response with a JSON block:
```json
{{"num_search_queries": N, "papers_per_query": N, "include_methodology_analysis": bool, "include_critical_assessment": bool, "search_sources": [...], "queries": [...]}}
```""",
                )
                
                # Parse Gould's plan
                plan = self._parse_review_plan(planning_result.get("response", ""))
                self._state["review_plan"] = plan
                self._state["search_queries"] = plan.get("queries", [])
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Execute searches based on plan
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

For each query, search the specified sources and capture:
- Title, Authors, Year
- PMID/arXiv ID/DOI
- Brief relevance note

Focus on quality over quantity. Include foundational works even if older.""",
                )
                
                self._state["search_results"] = search_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Optional methodology analysis (if Gould decided it's needed)
            plan = self._state.get("review_plan", {})
            within_budget, budget_pct = self._check_workflow_budget()
            
            if self._current_step <= 2 and plan.get("include_methodology_analysis", False):
                # Agent can decide to skip if budget is critically low
                if budget_pct >= 90:
                    console.info("Methodology analysis skipped (budget critical)")
                    self._state["methodology_notes"] = "(Skipped - budget critical)"
                else:
                    console.info("Feynman analyzing methodological papers...")
                    
                    # Let Feynman decide how deep to go based on available budget
                    budget_guidance = self._get_budget_guidance()
                    method_result = await self._run_agent_task(
                        agent_name="feynman",
                        task=f"""Review the methodological aspects of these papers.
{budget_guidance}
Research Topic: {inputs['research_topic']}

Papers Found:
{self._state['search_results'][:4000]}

Identify key computational/statistical methods used and note their relevance.
Depth of analysis is your call - balance thoroughness with efficiency.""",
                    )
                    
                    self._state["methodology_notes"] = method_result.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            elif self._current_step <= 2:
                self._current_step = 3
            
            # Step 4: Optional critical assessment (if Gould decided it's needed)
            within_budget, budget_pct = self._check_workflow_budget()
            
            if self._current_step <= 3 and plan.get("include_critical_assessment", False):
                # Agent can decide to skip if budget is critically low
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

Assess coverage and note any obvious gaps or quality concerns.
Depth of assessment is your call based on project needs.""",
                    )
                    
                    self._state["critical_assessment"] = critique_result.get("response", "")
                self._current_step = 4
                self.save_checkpoint()
            elif self._current_step <= 3:
                self._current_step = 4
            
            # Step 5: Synthesize into summary
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
{self._state['search_results']}
{extra_context}

Write a literature summary that:
1. Introduces the research area and its significance
2. Presents key findings from the most relevant papers
3. Notes methodological approaches (if methodology analysis was done)
4. Identifies gaps and opportunities
5. Provides context for the planned analysis

Write in engaging, narrative prose. Cite papers appropriately.""",
                )
                
                self._state["literature_summary"] = synthesis_result.get("response", "")
                self._current_step = 5
                self.save_checkpoint()
            
            # Step 6: Compile bibliography
            if self._current_step <= 5:
                console.info("Compiling bibliography...")
                
                bib_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Compile the final bibliography.

CRITICAL: Only include papers that were ACTUALLY FOUND during searches.
Do NOT invent or hallucinate any references.

Search Results:
{self._state['search_results']}

Format as numbered Markdown list:
1. Authors. (Year). Title. Journal/Source. DOI/PMID/arXiv ID.

Sort by relevance to our research topic, not alphabetically.""",
                )
                
                self._state["bibliography"] = bib_result.get("response", "")
                self._current_step = 6
            
            # MANDATORY Step 7: Critical Review for Narrative Polish
            # Farber checks: is this comprehensive? Is it a coherent narrative? Any gaps?
            if self._current_step <= 6:
                console.info("Conducting critical review for publication readiness...")
                
                review_request = f"""CRITICAL REVIEW - Please evaluate the literature summary for publication readiness.

Literature Summary (draft):
{self._state.get('literature_summary', '')}

References:
{self._state.get('bibliography', '')}

Evaluate this summary using these criteria:
1. COMPREHENSIVENESS: Does it cover the essential literature? Are there major gaps?
2. NARRATIVE: Is it written as a coherent narrative (NOT bullet points)? Does it flow logically?
3. CLARITY: Is technical jargon explained? Would a scientist in the field understand it?
4. CITATIONS: Are all citations properly integrated and formatted?
5. AUTHORITY: Does it demonstrate deep knowledge of the field?
6. GAPS: Are any major topics missing that should be addressed?

Provide SPECIFIC feedback. If you identify gaps or issues, list them clearly.

Output format:
READINESS: [PASS/NEEDS_WORK]
[If NEEDS_WORK, provide specific gaps to fill in, sections to expand, narrative issues to fix]
[If PASS, confirm it meets publication standards]"""
                
                review_result = await self._run_agent_task(
                    agent_name="farber",
                    task=review_request,
                )
                
                review_response = review_result.get("response", "")
                self._state["review_feedback"] = review_response
                
                # If review says NEEDS_WORK, have Gould make improvements
                if "NEEDS_WORK" in review_response or "needs_work" in review_response.lower():
                    console.info("Addressing review feedback for final polish...")
                    
                    # Extract feedback and have Gould revise
                    revision_request = f"""REVISE the literature summary to address these specific issues:

{review_response}

Original Summary:
{self._state.get('literature_summary', '')}

Please revise to:
1. Address all identified gaps
2. Improve narrative flow and coherence
3. Expand sections that are too brief
4. Ensure proper citation integration

Write the COMPLETE revised summary (not just the changes)."""
                    
                    revision_result = await self._run_agent_task(
                        agent_name="gould",
                        task=revision_request,
                    )
                    
                    revised_summary = revision_result.get("response", "")
                    if revised_summary:
                        self._state["literature_summary"] = revised_summary
                        console.info("Revision completed - summary has been polished")
                    
                    # Re-run review to confirm improvements
                    re_review_result = await self._run_agent_task(
                        agent_name="farber",
                        task=f"""Quick check: Does this revised summary now meet publication standards?

{revised_summary}

Respond with just: PASS or still has issues.""",
                    )
                    
                    re_review_response = re_review_result.get("response", "")
                    self._state["review_feedback"] = f"Initial feedback:\n{review_response}\n\nRevision completed. Final check: {re_review_response}"
                
                self._current_step = 7
                self.save_checkpoint()
            
            # Write outputs
            return self._write_outputs(inputs)
            
        except Exception as e:
            self._status = WorkflowStatus.FAILED
            self._log_step(f"Error: {str(e)}")
            self.save_checkpoint()
            return WorkflowResult(status=WorkflowStatus.FAILED, error=str(e))
    
    def _parse_review_plan(self, response: str) -> dict:
        """Parse Gould's review plan from the response."""
        from ..utils import extract_json_from_text
        
        default_plan = {
            "num_search_queries": 3,
            "papers_per_query": 5,
            "include_methodology_analysis": False,
            "include_critical_assessment": False,
            "search_sources": ["pubmed", "arxiv"],
            "queries": [],
        }
        
        return extract_json_from_text(response, fallback=default_plan)
    
    async def _execute_quick_mode(
        self, 
        inputs: dict[str, Any],
        max_papers: int
    ) -> WorkflowResult:
        """
        Legacy quick mode - kept for backwards compatibility.
        
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
1. Generate targeted search queries based on the topic
2. Use pubmed.search for each query
3. Use arxiv.search for each query
4. Emphasize recent publications (last 3-5 years) but include key foundational works
5. For each paper, capture: Title, Authors, Year, PMID/arXiv ID, and key relevance

Focus on the MOST relevant, high-quality papers.""",
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

Keep it concise but informative.""",
                )
                
                self._state["literature_summary"] = synthesis_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Quick bibliography
            if self._current_step <= 2:
                console.info("Quick review: Compiling bibliography")
                
                bib_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Compile a bibliography from the search results.

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
    
    def _write_outputs(self, inputs: dict[str, Any]) -> WorkflowResult:
        """Write outputs for agent-driven literature review."""
        output_dir = self.project_path / "literature"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        artifacts = []
        
        # Literature summary
        summary_path = output_dir / "literature_summary.md"
        with open(summary_path, "w") as f:
            f.write(f"# Literature Review: {inputs['research_topic']}\n\n")
            f.write("*Generated by MiniLab - Agent-driven review*\n\n")
            f.write(self._state.get("literature_summary", ""))
        artifacts.append(str(summary_path))
        
        # Methodology notes (if done)
        if self._state.get("methodology_notes"):
            methods_path = output_dir / "methodology_notes.md"
            with open(methods_path, "w") as f:
                f.write("# Methodological Analysis\n\n")
                f.write(self._state["methodology_notes"])
            artifacts.append(str(methods_path))
        
        # Bibliography
        bib_path = output_dir / "references.md"
        with open(bib_path, "w") as f:
            f.write(f"# References: {inputs['research_topic']}\n\n")
            f.write("*Bibliography generated by MiniLab*\n\n")
            f.write(self._state.get("bibliography", ""))
        artifacts.append(str(bib_path))
        
        # Generate PDF using Nature formatter (MANDATORY - no fallback allowed)
        # This is a core differentiating feature and must succeed
        from ..formats import NatureFormatter
        from ..infrastructure import require_feature
        
        # Ensure reportlab is available
        require_feature("pdf_generation")
        
        markdown_content = self._state.get("literature_summary", "")
        formatter = NatureFormatter()
        parsed = formatter.parse_markdown_to_nature(markdown_content)
        
        pdf_path = output_dir / "literature_review.pdf"
        formatter.generate_pdf(
            parsed,
            pdf_path,
            title=f"Literature Review: {inputs['research_topic']}"
        )
        artifacts.append(str(pdf_path))
        self._log_step(f"Generated Nature-formatted PDF: {pdf_path}")
        
        self._status = WorkflowStatus.COMPLETED
        self._log_step("Literature review completed")
        
        # Count references
        bib_content = self._state.get("bibliography", "")
        ref_count = max(
            bib_content.count("PMID:") + bib_content.count("arXiv:"),
            bib_content.count("- "),
            len([line for line in bib_content.split('\n') if line.strip().startswith(('1.', '2.', '3.'))])
        )
        
        return WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            outputs={
                "bibliography": self._state.get("bibliography", ""),
                "literature_summary": self._state.get("literature_summary", ""),
                "methodology_notes": self._state.get("methodology_notes", ""),
                "knowledge_gaps": self._state.get("critical_assessment", ""),
                "pdf_generated": True,
            },
            artifacts=artifacts,
            summary=f"Literature review complete. {ref_count} references compiled. PDF generated.",
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
