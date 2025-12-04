"""
Single Analysis Orchestrator - Conversational Multi-Agent Research Workflow

This orchestrator facilitates a natural, conversational research workflow where agents:
- Act autonomously with their specialized tools
- Communicate with each other to solve problems
- Ask questions when clarification is needed
- Never hallucinate - always verify with actual tool usage

STAGE 0: Confirm files and project naming
STAGE 1: Build project structure and summarize inputs  
STAGE 2: Plan full analysis (lit review, theory, implementation)
STAGE 3: Execute analysis (scripts, review, run)
STAGE 4: Write-up (figures assessment, legends, summary)
STAGE 5: Critical review and potential iteration
"""

from __future__ import annotations
import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List

from MiniLab.agents.base import Agent
from MiniLab.storage.transcript import TranscriptLogger


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _print_stage(stage: str, title: str):
    """Print a stage header."""
    print("\n" + "=" * 70)
    print(f"  {stage}: {title}")
    print("=" * 70 + "\n")


def _print_substage(substage: str):
    """Print a substage header."""
    print(f"\n  --- {substage} ---\n")


def _show_agent(agent_name: str, message: str, truncate: int = 0):
    """Display an agent's message cleanly. Set truncate=0 for no truncation."""
    # Clean up any tool markup for display
    clean = message
    clean = re.sub(r'DELEGATE:.*?---END', '', clean, flags=re.DOTALL)
    clean = re.sub(r'\[TOOL RESULT\].*?(?=\n\n|\Z)', '', clean, flags=re.DOTALL)
    clean = re.sub(r'<[^>]+>', '', clean)
    clean = '\n'.join(line for line in clean.split('\n') if line.strip())
    
    # Only truncate if truncate > 0
    if truncate > 0 and len(clean) > truncate:
        clean = clean[:truncate] + "\n  [...truncated for display...]"
    
    print(f"  [{agent_name}]:")
    for line in clean.split('\n'):
        print(f"    {line}")
    print()


def _get_user_input(prompt: str) -> str:
    """Get input from user with a prompt."""
    print(f"  {prompt}")
    return input("  > ").strip()


def _user_approves(response: str) -> bool:
    """Check if user response indicates approval."""
    approvals = ['yes', 'y', 'correct', 'good', 'fine', 'ok', 'proceed', 'looks good', 'approved', 'accept']
    return any(word in response.lower() for word in approvals)


async def _agent_uses_tool(agent: Agent, tool_name: str, **kwargs) -> Dict[str, Any]:
    """Have an agent use a tool and return the result."""
    if not agent.has_tool(tool_name):
        return {"success": False, "error": f"{agent.display_name} doesn't have {tool_name} tool"}
    return await agent.use_tool(tool_name, **kwargs)


async def _discover_files(agent: Agent, base_path: str) -> List[str]:
    """Recursively discover all files in a directory using agent's filesystem tool."""
    discovered = []
    # Normalize path - remove trailing slash
    base_path = base_path.rstrip('/')
    to_explore = [base_path]
    explored = set()
    
    while to_explore:
        current = to_explore.pop(0)
        if current in explored:
            continue
        explored.add(current)
        
        result = await _agent_uses_tool(agent, "filesystem", action="list", path=current)
        
        if result.get("success"):
            for item in result.get("items", []):
                if item['name'].startswith('.'):
                    continue
                item_path = f"{current}/{item['name']}"
                if item['type'] == 'file':
                    discovered.append(item_path)
                elif item['type'] == 'directory':
                    to_explore.append(item_path)
    
    return discovered


async def _agent_reads_file(agent: Agent, path: str) -> str:
    """Have agent read a file and return its content."""
    result = await _agent_uses_tool(agent, "filesystem", action="read", path=path)
    if result.get("success"):
        return result.get("content", "")
    return f"[Error reading {path}: {result.get('error', 'Unknown')}]"


async def _agent_writes_file(agent: Agent, path: str, content: str) -> bool:
    """Have agent write content to a file."""
    result = await _agent_uses_tool(agent, "filesystem", action="write", path=path, content=content)
    return result.get("success", False)


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

async def run_single_analysis(
    agents: Dict[str, Agent],
    research_question: str,
    max_tokens: int = 1_000_000,
    logger: Optional[TranscriptLogger] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute a conversational multi-agent research workflow.
    
    The workflow progresses through 6 stages with natural agent communication,
    user checkpoints, and tool-verified actions (no hallucination).
    
    Args:
        agents: Dictionary of initialized Agent instances
        research_question: User's research question/task
        max_tokens: Token budget (for tracking)
        logger: Optional transcript logger
        output_dir: Optional output directory override
    
    Returns:
        Dictionary with workflow results and metadata
    """
    
    print("\n" + "=" * 70)
    print("  MINILAB SINGLE ANALYSIS WORKFLOW")
    print("=" * 70)
    print(f"\n  Research Question:\n    {research_question}\n")
    print("=" * 70 + "\n")
    
    # Agent references
    bohr = agents["bohr"]
    farber = agents["farber"]
    gould = agents["gould"]
    feynman = agents["feynman"]
    shannon = agents["shannon"]
    greider = agents["greider"]
    bayes = agents["bayes"]
    hinton = agents["hinton"]
    dayhoff = agents["dayhoff"]
    
    # Token tracking (rough estimate)
    tokens_used = 0
    TOKEN_PER_CALL = 2000
    
    # Workflow state
    project_name = None
    files = []
    
    # =========================================================================
    # STAGE 0: CONFIRM FILES AND NAMING
    # =========================================================================
    _print_stage("STAGE 0", "Confirm Files and Project Naming")
    if logger:
        logger.log_stage_transition("Stage 0", "Files and naming")
    
    # Extract target directory from research question
    dir_match = re.search(r'ReadData/[\w/]+', research_question)
    target_dir = dir_match.group(0) if dir_match else "ReadData"
    
    print(f"  Target directory: {target_dir}\n")
    print("  Discovering files...")
    
    # Actually discover files using Bohr's filesystem tool
    files = await _discover_files(bohr, target_dir)
    tokens_used += TOKEN_PER_CALL * 2  # Rough estimate for discovery
    
    if not files:
        print(f"\n  ERROR: No files found in {target_dir}")
        return {"success": False, "error": f"No files found in {target_dir}"}
    
    print(f"  Found {len(files)} files.\n")
    
    # Now have Bohr think about a project name conversationally
    file_list_str = "\n".join(f"    - {f}" for f in files)
    
    bohr_response = await bohr.arespond(f"""I need to start a new analysis project. The user asked:

"{research_question}"

I've discovered these files:
{file_list_str}

My tasks:
1. Think of an appropriate, descriptive project name (CamelCase format, like "PluvictoResponseAnalysis" or "TumorMutationStudy")
2. Confirm I understand the scope of the data

Please suggest a project name and briefly note what kind of data this appears to be. Keep it concise.""")
    tokens_used += TOKEN_PER_CALL
    
    if logger:
        logger.log_agent_response("Bohr", "bohr", bohr_response, TOKEN_PER_CALL)
    
    # Extract project name from response
    # Look for CamelCase names in various formats
    name_patterns = [
        # "**Project Name: PluvictoCfDNAResponse**" format
        r"\*\*Project\s*Name[:\s]+([A-Z][a-zA-Z0-9]+)\*\*",
        # "Project Name: `PluvictoCfDNAResponse`" format
        r"Project\s*Name[:\s]+`([A-Z][a-zA-Z0-9]+)`",
        # "Project Name: PluvictoCfDNAResponse" format (plain)
        r"Project\s*Name[:\s]+([A-Z][a-zA-Z0-9]+)",
        # Backtick-wrapped standalone: `ProjectName`
        r"`([A-Z][a-zA-Z0-9]{6,})`",
        # Bold standalone: **ProjectName**
        r"\*\*([A-Z][a-zA-Z0-9]{6,})\*\*",
        # Generic patterns
        r"suggest(?:ed)?[:\s]+[\"'`*]*([A-Z][a-zA-Z0-9]+)[\"'`*]*",
        r"call(?:ed|ing)?(?:\s+it)?[:\s]+[\"'`*]*([A-Z][a-zA-Z0-9]+)[\"'`*]*",
        r"name[:\s]+[\"'`*]*([A-Z][a-zA-Z0-9]+)[\"'`*]*",
    ]
    
    project_name = None
    for pattern in name_patterns:
        matches = re.findall(pattern, bohr_response, re.IGNORECASE)
        for candidate in matches:
            candidate = candidate.strip('*"\'` ')
            # Must be CamelCase (2+ capitals) and reasonable length
            capitals = sum(1 for c in candidate if c.isupper())
            if capitals >= 2 and 6 < len(candidate) < 50:
                project_name = candidate
                break
        if project_name:
            break
    
    if not project_name:
        # Last resort: find any CamelCase word that looks like a project name
        camel_matches = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)\b', bohr_response)
        for candidate in camel_matches:
            if len(candidate) > 6:
                project_name = candidate
                break
    
    if not project_name:
        project_name = "UnnamedProject"
    
    # Show user and get confirmation
    _show_agent("Bohr", bohr_response)
    
    print("  Files discovered:")
    for f in files:
        print(f"    - {f}")
    print(f"\n  Suggested project name: {project_name}\n")
    
    user_input = _get_user_input("Is this correct? (approve, rename to X, or describe issues)")
    
    while not _user_approves(user_input):
        # Check if user wants to rename
        rename_match = re.search(r'rename(?:\s+to)?\s+["\']?(\w+)["\']?', user_input, re.IGNORECASE)
        if rename_match:
            project_name = rename_match.group(1)
            print(f"\n  Project renamed to: {project_name}")
        else:
            # User has concerns - have Bohr address them
            bohr_response = await bohr.arespond(f"""The user has feedback about my file discovery and project naming.

User feedback: "{user_input}"

The files I found were:
{file_list_str}

Current project name: {project_name}

Please address their concerns. If they mention missing files, I should look again. If they want a different name, suggest alternatives.""")
            tokens_used += TOKEN_PER_CALL
            
            _show_agent("Bohr", bohr_response)
            
            # Try to re-extract project name with same patterns
            for pattern in name_patterns:
                matches = re.findall(pattern, bohr_response)
                for candidate in matches:
                    candidate = candidate.strip('*"\'` ')
                    capitals = sum(1 for c in candidate if c.isupper())
                    if capitals >= 2 and 6 < len(candidate) < 50:
                        project_name = candidate
                        break
                if project_name and project_name != "UnnamedProject":
                    break
            
            print(f"  Current project name: {project_name}\n")
        
        user_input = _get_user_input("Is this correct now?")
    
    print(f"\n  Confirmed: Project '{project_name}' with {len(files)} files.\n")
    
    # =========================================================================
    # STAGE 1: BUILD PROJECT AND SUMMARIZE INPUTS
    # =========================================================================
    _print_stage("STAGE 1", "Build Project Structure and Summarize Inputs")
    if logger:
        logger.log_stage_transition("Stage 1", "Project setup and data summary")
    
    # Create project directory structure
    project_path = Path.cwd() / "Sandbox" / project_name
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "scratch").mkdir(exist_ok=True)
    (project_path / "scripts").mkdir(exist_ok=True)
    
    print(f"  Created: Sandbox/{project_name}/")
    print(f"           Sandbox/{project_name}/scratch/")
    print(f"           Sandbox/{project_name}/scripts/\n")
    
    # Have Bohr read and summarize the data files
    _print_substage("1A: Reading and Summarizing Data Files")
    
    # First, compute actual data statistics by reading files
    print("  Computing data statistics...")
    
    data_stats = {}
    for f in files:
        full_content = await _agent_reads_file(bohr, f)
        lines = full_content.strip().split('\n')
        
        if lines:
            # First line is header
            header = lines[0]
            columns = [c.strip() for c in header.split(',')]
            n_rows = len(lines) - 1  # Exclude header
            n_cols = len(columns)
            
            # Try to identify sample IDs (usually first column)
            sample_ids = set()
            for line in lines[1:min(len(lines), 100)]:  # Sample first 100 rows
                parts = line.split(',')
                if parts:
                    sample_ids.add(parts[0].strip())
            
            # Get preview (first 5 data lines)
            preview_lines = lines[:6]  # header + 5 data rows
            
            data_stats[f] = {
                'filename': f.split('/')[-1],
                'n_rows': n_rows,
                'n_cols': n_cols,
                'columns': columns,
                'sample_ids_preview': list(sample_ids)[:10],
                'preview': '\n'.join(preview_lines)
            }
    
    tokens_used += TOKEN_PER_CALL * len(files)
    
    # Build statistics summary
    stats_summary = []
    total_samples = set()
    for f, stats in data_stats.items():
        stats_summary.append(f"**{stats['filename']}**: {stats['n_rows']} rows x {stats['n_cols']} columns")
        stats_summary.append(f"  Columns: {', '.join(stats['columns'][:10])}{'...' if len(stats['columns']) > 10 else ''}")
        total_samples.update(stats['sample_ids_preview'])
    
    stats_text = '\n'.join(stats_summary)
    
    # Build preview text (condensed)
    preview_parts = []
    for f, stats in data_stats.items():
        preview_parts.append(f"=== {stats['filename']} ({stats['n_rows']} rows x {stats['n_cols']} cols) ===\n{stats['preview']}")
    preview_text = "\n\n".join(preview_parts)
    
    print(f"  Found {len(total_samples)}+ unique sample IDs across {len(files)} files\n")
    
    bohr_response = await bohr.arespond(f"""I'm starting project "{project_name}". Here are the data statistics and previews:

## Data Statistics
{stats_text}

## File Previews (first 5 rows each)
{preview_text}

Based on this, please provide:

i) **ID Format**: The sample/patient ID naming convention (format, pattern)
ii) **Sample Count**: Total unique samples across all files (I see at least {len(total_samples)} unique IDs)
iii) **Feature Summary**: What kinds of features are in each file, with brief descriptions
iv) **Questions**: Any ambiguous columns or patterns that need clarification

Be specific about numbers - this will inform our statistical power for analyses.""")
    tokens_used += TOKEN_PER_CALL
    
    if logger:
        logger.log_agent_response("Bohr", "bohr", bohr_response, TOKEN_PER_CALL)
    
    _show_agent("Bohr", bohr_response)
    
    # Write detailed data manifest
    manifest_content = f"""DATA MANIFEST for {project_name}
Generated by Bohr
================================================================================

RESEARCH QUESTION:
{research_question}

================================================================================
DATA STATISTICS SUMMARY
================================================================================
"""
    for f, stats in data_stats.items():
        manifest_content += f"""
{stats['filename']}
  Rows: {stats['n_rows']}
  Columns: {stats['n_cols']}
  Column names: {', '.join(stats['columns'])}
"""
    
    manifest_content += f"""
================================================================================
TOTAL UNIQUE SAMPLE IDs (from first 100 rows): {len(total_samples)}+
Sample ID examples: {', '.join(list(total_samples)[:20])}

================================================================================
BOHR'S ANALYSIS
================================================================================
{bohr_response}
"""
    await _agent_writes_file(bohr, f"Sandbox/{project_name}/scratch/data_manifest.txt", manifest_content)
    print(f"  Created: Sandbox/{project_name}/scratch/data_manifest.txt\n")
    
    user_input = _get_user_input("Is this interpretation of the data correct?")
    
    # Track user clarifications for the manifest
    user_clarifications = []
    
    while not _user_approves(user_input):
        user_clarifications.append(user_input)
        
        bohr_response = await bohr.arespond(f"""The user has provided clarifying feedback about the data.

User feedback: "{user_input}"

Please:
1. Briefly acknowledge what you learned from their feedback
2. Update your understanding of the data accordingly
3. If you still have questions, ask them clearly
4. If the feedback resolves your questions, say "Ready to proceed to analysis planning."

Be concise - just summarize the key points and any remaining questions.""")
        tokens_used += TOKEN_PER_CALL
        
        _show_agent("Bohr", bohr_response)
        
        # Update manifest with clarifications
        manifest_content = f"""DATA MANIFEST for {project_name}
Generated by Bohr
================================================================================

RESEARCH QUESTION:
{research_question}

================================================================================
DATA STATISTICS SUMMARY
================================================================================
"""
        for f, stats in data_stats.items():
            manifest_content += f"""
{stats['filename']}
  Rows: {stats['n_rows']}
  Columns: {stats['n_cols']}
  Column names: {', '.join(stats['columns'])}
"""
        manifest_content += f"""
================================================================================
TOTAL UNIQUE SAMPLE IDs: {len(total_samples)}+

================================================================================
USER CLARIFICATIONS
================================================================================
"""
        for i, clarification in enumerate(user_clarifications, 1):
            manifest_content += f"\n{i}. {clarification}\n"
        
        manifest_content += f"""
================================================================================
BOHR'S UPDATED UNDERSTANDING
================================================================================
{bohr_response}
"""
        await _agent_writes_file(bohr, f"Sandbox/{project_name}/scratch/data_manifest.txt", manifest_content)
        
        # Check if Bohr is ready to proceed or has more questions
        if "ready to proceed" in bohr_response.lower() or "no further questions" in bohr_response.lower():
            print("  Bohr is ready to proceed.\n")
            break
        
        user_input = _get_user_input("Any other clarifications, or ready to proceed? (type 'proceed' or provide more info)")
    
    print("\n  Data summary confirmed.\n")
    
    # Combine original analysis with any clarifications for downstream stages
    data_summary = bohr_response
    if user_clarifications:
        data_summary = f"""Original Analysis:
{bohr_response}

User Clarifications:
{chr(10).join(f'- {c}' for c in user_clarifications)}"""
    
    # =========================================================================
    # STAGE 2: PLAN FULL ANALYSIS
    # =========================================================================
    _print_stage("STAGE 2", "Plan Full Analysis")
    if logger:
        logger.log_stage_transition("Stage 2", "Analysis planning")
    
    # 2A: Initial Planning - Bohr -> Gould -> Farber -> Bohr
    _print_substage("2A: Literature Review and Feasibility Assessment")
    
    print("  Bohr communicating with Gould for literature review...")
    
    # Gould does literature review
    gould_response = await gould.arespond(f"""Bohr has asked me to conduct a literature review for a new project.

PROJECT: {project_name}
RESEARCH QUESTION: "{research_question}"

DATA SUMMARY:
{data_summary}

As the team librarian and science writer, I need to:
1. Search for relevant papers in this domain (use web_search and citation_index tools)
2. Summarize the current state of research
3. Identify common methodological approaches
4. Compile preliminary citations (with DOIs where possible)
5. Suggest 3-5 testable hypotheses based on the literature

I should provide Bohr with a comprehensive briefing that can be evaluated for feasibility.""")
    tokens_used += TOKEN_PER_CALL
    
    if logger:
        logger.log_agent_response("Gould", "gould", gould_response, TOKEN_PER_CALL)
    
    _show_agent("Gould", gould_response)
    
    # Save literature review
    await _agent_writes_file(gould, f"Sandbox/{project_name}/scratch/literature_review.md", 
                            f"# Literature Review\n\n{gould_response}")
    
    # Farber evaluates feasibility
    print("  Gould sending to Farber for feasibility review...")
    
    farber_response = await farber.arespond(f"""Gould has provided a literature review and suggested hypotheses. As the clinical critic, I need to evaluate the merit and feasibility.

PROJECT: {project_name}
RESEARCH QUESTION: "{research_question}"

DATA SUMMARY:
{data_summary}

GOULD'S LITERATURE REVIEW:
{gould_response}

As the adversarial critic and clinical expert, I should:
1. Evaluate clinical relevance of the proposed analyses
2. Assess feasibility given the available data
3. Identify potential confounders or biases
4. Challenge weak assumptions
5. Make constructive suggestions for improvement

I'll be thorough but fair in my assessment.""")
    tokens_used += TOKEN_PER_CALL
    
    if logger:
        logger.log_agent_response("Farber", "farber", farber_response, TOKEN_PER_CALL)
    
    _show_agent("Farber", farber_response)
    
    # Bohr synthesizes - allow proper discussion loop
    print("  Farber returning to Bohr for synthesis...")
    
    plan_approved = False
    iteration = 0
    max_iterations = 5  # Allow more discussion rounds
    
    while not plan_approved and iteration < max_iterations:
        bohr_synthesis = await bohr.arespond(f"""I've received input from both Gould and Farber.

GOULD'S LITERATURE REVIEW:
{gould_response}

FARBER'S FEASIBILITY ASSESSMENT:
{farber_response}

As project lead, I need to synthesize this into a coherent plan. 

If I have significant concerns or need clarifications, I should communicate them.
If the plan looks sound, I should approve it.

Please respond with EITHER:
- APPROVE: [summary of the approved plan direction and key hypotheses]
OR
- CONCERNS: [specific issues] -> [direct to GOULD and/or FARBER]
OR  
- ASK_USER: [critical questions that only the user can answer]""")
        tokens_used += TOKEN_PER_CALL
        
        if logger:
            logger.log_agent_response("Bohr", "bohr", bohr_synthesis, TOKEN_PER_CALL)
        
        _show_agent("Bohr", bohr_synthesis)
        
        if "APPROVE" in bohr_synthesis.upper() and "CONCERNS" not in bohr_synthesis.upper():
            plan_approved = True
        elif "ASK_USER" in bohr_synthesis.upper():
            # Bohr has questions for the user - pause and ask
            print("\n  Bohr has questions that require user input.\n")
            user_clarification = _get_user_input("Please provide clarification (or 'proceed' to continue anyway):")
            
            if user_clarification.lower().strip() != 'proceed':
                # Feed user's answer back to Bohr
                bohr_followup = await bohr.arespond(f"""The user has provided clarification:

"{user_clarification}"

Based on this new information, please reassess. Respond with:
- APPROVE: [updated plan] if we can proceed
- CONCERNS: [remaining issues] if more discussion needed""")
                tokens_used += TOKEN_PER_CALL
                _show_agent("Bohr", bohr_followup)
                
                if "APPROVE" in bohr_followup.upper() and "CONCERNS" not in bohr_followup.upper():
                    plan_approved = True
                    bohr_synthesis = bohr_followup
            else:
                # User wants to proceed anyway
                plan_approved = True
                
        elif "CONCERNS" in bohr_synthesis.upper():
            iteration += 1
            print(f"\n  Bohr has concerns (discussion round {iteration}/{max_iterations})...")
            
            # Route concerns to appropriate agent(s) - check for BOTH
            addressed_gould = False
            addressed_farber = False
            
            if "GOULD" in bohr_synthesis.upper():
                gould_response = await gould.arespond(f"""Bohr has directed concerns to me:

{bohr_synthesis}

Please address these concerns thoroughly.""")
                tokens_used += TOKEN_PER_CALL
                _show_agent("Gould", gould_response)
                addressed_gould = True
                
            if "FARBER" in bohr_synthesis.upper():
                farber_response = await farber.arespond(f"""Bohr has directed concerns to me:

{bohr_synthesis}

Please address these concerns thoroughly.""")
                tokens_used += TOKEN_PER_CALL
                _show_agent("Farber", farber_response)
                addressed_farber = True
            
            # If concerns weren't routed to anyone specific, ask user
            if not addressed_gould and not addressed_farber:
                print("\n  Concerns not directed to specific team members.")
                user_input = _get_user_input("How should we proceed? (provide guidance or 'proceed' to continue):")
                if user_input.lower().strip() == 'proceed':
                    plan_approved = True
        else:
            # Ambiguous response - ask user
            print("\n  Bohr's response was ambiguous.")
            user_input = _get_user_input("Should we proceed with this plan? (yes/no):")
            if user_input.lower().strip() in ['yes', 'y', 'proceed']:
                plan_approved = True
    
    # If we exited the loop without approval, ask user
    if not plan_approved:
        print(f"\n  Discussion reached {max_iterations} rounds without resolution.")
        user_decision = _get_user_input("Should we proceed with the current plan anyway? (yes/no):")
        if user_decision.lower().strip() in ['yes', 'y', 'proceed']:
            plan_approved = True
        else:
            print("\n  Analysis paused. Please resolve concerns before continuing.")
            return {"success": False, "error": "Plan not approved after discussion"}
    
    approved_plan = bohr_synthesis
    print("\n  ✓ Plan approved - proceeding to next stage.\n")
    
    # 2B: Theory Core Enhancement
    _print_substage("2B: Theory Guild Enhancement")
    
    print("  Consulting Feynman, Shannon, and Greider...")
    
    # Parallel consultation with theory guild
    feynman_task = feynman.arespond(f"""Bohr has asked for my theoretical input on this analysis plan.

PROJECT: {project_name}
RESEARCH QUESTION: "{research_question}"
DATA: {data_summary}

APPROVED PLAN:
{approved_plan}

As a theoretical physicist and creative thinker, I should:
1. Suggest novel analytical approaches that might reveal deeper patterns
2. Identify underlying principles or mechanisms worth exploring
3. Propose unconventional hypotheses
4. Consider cross-disciplinary insights from physics/information theory

Keep suggestions focused and actionable for this specific dataset.""")
    
    shannon_task = shannon.arespond(f"""Bohr has asked for my input on this analysis plan.

PROJECT: {project_name}
RESEARCH QUESTION: "{research_question}"
DATA: {data_summary}

APPROVED PLAN:
{approved_plan}

As an information theorist and causal designer, I should:
1. Evaluate the data structure and information flow
2. Suggest causal inference approaches
3. Identify potential confounding and how to address it
4. Recommend statistical methods for robustness
5. Ensure reproducibility best practices

Keep suggestions focused and actionable.""")
    
    greider_task = greider.arespond(f"""Bohr has asked for my biological input on this analysis plan.

PROJECT: {project_name}
RESEARCH QUESTION: "{research_question}"
DATA: {data_summary}

APPROVED PLAN:
{approved_plan}

As a molecular biologist and mechanistic expert, I should:
1. Evaluate biological plausibility of hypotheses
2. Suggest mechanistic angles to explore
3. Identify relevant biological pathways
4. Consider what biological validation might look like
5. Ensure interpretations are biologically sound

Keep suggestions focused and actionable.""")
    
    feynman_resp, shannon_resp, greider_resp = await asyncio.gather(
        feynman_task, shannon_task, greider_task
    )
    tokens_used += TOKEN_PER_CALL * 3
    
    if logger:
        logger.log_agent_response("Feynman", "feynman", feynman_resp, TOKEN_PER_CALL)
        logger.log_agent_response("Shannon", "shannon", shannon_resp, TOKEN_PER_CALL)
        logger.log_agent_response("Greider", "greider", greider_resp, TOKEN_PER_CALL)
    
    _show_agent("Feynman", feynman_resp)
    _show_agent("Shannon", shannon_resp)
    _show_agent("Greider", greider_resp)
    
    # Bohr synthesizes into detailed plan
    print("  Bohr synthesizing detailed plan...")
    
    detailed_plan = await bohr.arespond(f"""I've received theoretical enhancements from the Theory Guild.

APPROVED PLAN:
{approved_plan}

FEYNMAN'S SUGGESTIONS (theoretical/creative):
{feynman_resp}

SHANNON'S SUGGESTIONS (information/causality):
{shannon_resp}

GREIDER'S SUGGESTIONS (biological/mechanistic):
{greider_resp}

I need to synthesize all of this into a highly detailed, actionable analysis plan.

The plan MUST include:
1. HYPOTHESES: Specific, testable hypotheses (3-5)
2. ANALYSES: Statistical tests and methods to use
3. FIGURES: Description of 4-6 figure panels for {project_name}_figures.pdf
   - Panel a: [description]
   - Panel b: [description]
   - etc.
4. CITATIONS: Key papers to reference (at least 5, with DOIs if known)
5. EXPECTED OUTPUTS: What results we expect if hypotheses are supported/refuted

Please write this detailed plan.""")
    tokens_used += TOKEN_PER_CALL
    
    if logger:
        logger.log_agent_response("Bohr", "bohr", detailed_plan, TOKEN_PER_CALL)
    
    _show_agent("Bohr", detailed_plan)
    
    # Save detailed plan
    await _agent_writes_file(bohr, f"Sandbox/{project_name}/scratch/detailed_plan.md",
                            f"# Detailed Analysis Plan for {project_name}\n\n{detailed_plan}")
    print(f"  Created: Sandbox/{project_name}/scratch/detailed_plan.md\n")
    
    # 2C: Implementation Planning (Dayhoff)
    _print_substage("2C: Implementation Planning (Dayhoff)")
    
    print("  Dayhoff creating implementation plan...")
    
    impl_plan = await dayhoff.arespond(f"""Bohr has sent me the detailed analysis plan. I need to create a concrete implementation plan.

PROJECT: {project_name}
DATA FILES:
{file_list_str}

DETAILED PLAN:
{detailed_plan}

As the bioinformatician, I need to outline ALL scripts needed:

1. DATA LOADING (01_load_data.py)
   - Input files and expected formats
   - Validation checks
   - Output: cleaned data objects

2. DATA PREPROCESSING (02_preprocess.py)
   - Cleaning steps
   - Transformations
   - Feature engineering
   - Output: analysis-ready data

3. ANALYSIS (03_analysis.py)
   - Statistical tests to perform
   - Models to fit
   - Output: results tables/objects

4. FIGURE GENERATION (04_figures.py)
   - Panel-by-panel specifications
   - Plot types and parameters
   - Output: individual panel files

5. FIGURE ASSEMBLY (05_assemble.py)
   - Combine panels into {project_name}_figures.pdf
   - Layout: 4-6 panels labeled a-f
   - Page size: 8.5 x 11 inches

For each script, specify:
- Purpose and dependencies
- Input/output files
- Key libraries needed
- Critical parameters""")
    tokens_used += TOKEN_PER_CALL
    
    if logger:
        logger.log_agent_response("Dayhoff", "dayhoff", impl_plan, TOKEN_PER_CALL)
    
    _show_agent("Dayhoff", impl_plan)
    
    # Save implementation plan
    await _agent_writes_file(dayhoff, f"Sandbox/{project_name}/scratch/implementation_plan.md",
                            f"# Implementation Plan for {project_name}\n\n{impl_plan}")
    print(f"  Created: Sandbox/{project_name}/scratch/implementation_plan.md\n")
    
    print("  Stage 2 complete: Full analysis plan created.\n")
    
    # =========================================================================
    # STAGE 2D: CRITICAL QUESTIONS CHECKPOINT
    # =========================================================================
    _print_substage("2D: Critical Questions (User Input)")
    
    # Extract critical questions from the planning discussion
    all_plan_text = f"{gould_response}\n{farber_response}\n{feynman_resp}\n{shannon_resp}\n{greider_resp}\n{detailed_plan}"
    
    # Have Bohr summarize critical questions that need user input
    critical_q = await bohr.arespond(f"""Based on our planning discussion, identify any CRITICAL questions that the USER should answer before we proceed to implementation.

Look for questions about:
- Missing data (why are samples missing? is it random or informative?)
- Censoring (is survival data right-censored? what's the follow-up period?)
- Clinical context (what decision will this model inform?)
- Data quality (are there known batch effects? QC failures?)
- Prior treatments (what therapies preceded Pluvicto?)

PLANNING DISCUSSION:
{all_plan_text[:6000]}

If there are critical questions, list them numbered (1, 2, 3...).
If no critical questions remain, say "NO CRITICAL QUESTIONS - ready to proceed."

Be specific and concise.""")
    tokens_used += TOKEN_PER_CALL
    
    _show_agent("Bohr", critical_q)
    
    # If there are questions, pause for user input
    if "NO CRITICAL QUESTIONS" not in critical_q.upper():
        print("\n  The team has questions that may affect the analysis.\n")
        user_answers = _get_user_input("Please answer these questions (or type 'skip' to proceed without):")
        
        if user_answers.lower().strip() != 'skip':
            # Update the detailed plan with user's answers
            plan_update = await bohr.arespond(f"""The user has provided answers to our critical questions:

USER'S ANSWERS:
{user_answers}

Please briefly summarize how this affects our analysis plan. Update any assumptions or approaches based on this information.""")
            tokens_used += TOKEN_PER_CALL
            _show_agent("Bohr", plan_update)
            
            # Append to detailed plan
            detailed_plan = detailed_plan + f"\n\n## User Clarifications\n{user_answers}\n\n## Plan Updates\n{plan_update}"
    
    print()
    
    # =========================================================================
    # STAGE 3: EXECUTION
    # =========================================================================
    _print_stage("STAGE 3", "Execution")
    if logger:
        logger.log_stage_transition("Stage 3", "Script generation and execution")
    
    # 3A: Hinton generates scripts - ORCHESTRATOR WRITES FILES
    _print_substage("3A: Script Generation (Hinton)")
    if logger:
        logger.log_system_event("SUBSTAGE", "3A: Script Generation (Hinton)")
    
    # Ensure directories exist
    scripts_dir = project_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = project_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Define expected scripts
    expected_scripts = [
        ("01_load_data.py", "Load all data files, validate patient IDs, compute basic statistics, characterize outcomes and missingness"),
        ("02_preprocess.py", "Clean, normalize, handle missing values, create composite scores, prepare analysis-ready feature matrix"),
        ("03_analysis.py", "Run statistical analyses: survival modeling, predictive models with cross-validation, feature importance"),
        ("04_figures.py", "Generate individual figure panels as PNG files (4-6 panels covering key results)"),
        ("05_assemble.py", "Combine PNG panels into final PDF with proper layout and labels"),
    ]
    
    print(f"  Creating scripts in Sandbox/{project_name}/scripts/\n")
    
    def extract_code_from_response(response: str) -> str:
        """Extract Python code from agent response - try multiple strategies."""
        import re
        
        # Strategy 1: Find ```python ... ``` blocks
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # Take the longest match (likely the full script)
            code = max(matches, key=len).strip()
            if len(code) > 50:
                return code
        
        # Strategy 2: Find generic ``` ... ``` blocks  
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            for code in matches:
                code = code.strip()
                # Check if it looks like Python code
                if ('import ' in code or 'def ' in code or 'class ' in code or 
                    'print(' in code or 'pd.read' in code):
                    if len(code) > 50:
                        return code
        
        # Strategy 3: Look for shebang or docstring start
        lines = response.split('\n')
        code_start = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('#!/') or 
                stripped.startswith('"""') or
                stripped.startswith("'''") or
                stripped.startswith('import ') or
                stripped.startswith('from ')):
                code_start = i
                break
        
        if code_start >= 0:
            # Take everything from code start to end, or until we hit markdown
            code_lines = []
            for line in lines[code_start:]:
                # Stop if we hit obvious markdown/explanation
                if line.strip().startswith('##') or line.strip().startswith('**'):
                    break
                code_lines.append(line)
            code = '\n'.join(code_lines).strip()
            if len(code) > 50:
                return code
        
        # Strategy 4: If response is mostly code-like, take the whole thing
        # Count indicators of code vs prose
        code_indicators = sum([
            response.count('import '),
            response.count('def '),
            response.count('print('),
            response.count('pd.'),
            response.count('np.'),
            response.count('= '),
            response.count('if __name__'),
        ])
        prose_indicators = sum([
            response.count('. '),  # Sentences
            response.count('Here '),
            response.count('This '),
            response.count('The '),
            response.count('I '),
        ])
        
        if code_indicators > prose_indicators * 2:
            # Probably raw code - clean it up
            clean = response.strip()
            # Remove any leading explanation lines
            lines = clean.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ', '#!/', '"""', "'''")):
                    return '\n'.join(lines[i:]).strip()
            return clean
        
        return ""
    
    # Script-specific token limits - VERY HIGH for complex comprehensive scripts
    # Increased again to prevent truncation (scripts need ~400+ lines)
    script_token_limits = {
        "01_load_data.py": 24000,
        "02_preprocess.py": 24000,
        "03_analysis.py": 24000,
        "04_figures.py": 24000,
        "05_assemble.py": 12000,
    }
    DEFAULT_SCRIPT_TOKENS = 24000
    
    def validate_python_syntax(code: str) -> tuple[bool, str]:
        """Check Python syntax without executing. Returns (is_valid, error_message)."""
        import ast
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
    
    def get_lines_around_error(code: str, line_num: int, context: int = 10) -> str:
        """Get lines around an error for context."""
        lines = code.split('\n')
        start = max(0, line_num - context - 1)
        end = min(len(lines), line_num + context)
        
        result = []
        for i in range(start, end):
            marker = ">>> " if i == line_num - 1 else "    "
            result.append(f"{i+1:4d}{marker}{lines[i]}")
        return '\n'.join(result)
    
    def detect_truncation(code: str) -> tuple[bool, str]:
        """
        Detect if code appears to be truncated mid-generation.
        Returns (is_truncated, reason).
        
        Signs of truncation:
        - Ends mid-string literal
        - Ends with unclosed bracket
        - Last function definition has no body
        - File ends mid-statement
        - Indentation drops unexpectedly at end
        - No main block exists
        """
        lines = code.strip().split('\n')
        if not lines:
            return True, "Empty code"
        
        last_line = lines[-1].strip()
        last_line_raw = lines[-1]  # Keep original indentation
        
        # Check for unclosed string literals in last few lines
        last_chunk = '\n'.join(lines[-5:])
        
        # Simple check: does the last line look truncated?
        if last_line.endswith((':',)) and not any(last_line.startswith(kw) for kw in ['if', 'else', 'elif', 'for', 'while', 'def', 'class', 'try', 'except', 'with', 'finally']):
            # Ends with : but not a block statement
            return True, "Ends with colon but incomplete statement"
        
        # Check if last line is incomplete
        if last_line and last_line[-1] in '([{,\\':
            return True, f"Ends with open bracket or continuation: {last_line[-20:]}"
        
        # Check for sudden indent drop - if we're deep in a block and last line is unindented
        # Look at indentation pattern in last 10 lines
        last_ten = lines[-10:] if len(lines) >= 10 else lines
        indents = []
        for line in last_ten:
            if line.strip():  # Non-empty line
                indent = len(line) - len(line.lstrip())
                indents.append(indent)
        
        if len(indents) >= 3:
            # Check if last line's indent is way less than previous lines
            avg_indent = sum(indents[:-1]) / len(indents[:-1])
            if indents[-1] == 0 and avg_indent > 8:
                return True, f"Sudden indent drop: was ~{avg_indent:.0f}, now 0 at end"
        
        # Check if file is missing if __name__ == "__main__"
        has_main = 'if __name__' in code or "if __name__" in code
        if not has_main and len(lines) > 50:
            return True, "No if __name__ == '__main__' block found"
        
        # Check for function definition without body
        for i in range(len(lines) - 1, max(0, len(lines) - 10), -1):
            line = lines[i].strip()
            if line.startswith('def ') and line.endswith(':'):
                # Check if next lines exist and have body
                if i == len(lines) - 1:
                    return True, f"Function definition without body: {line[:50]}"
                next_lines = [l for l in lines[i+1:] if l.strip()]
                if not next_lines:
                    return True, f"Function definition without body: {line[:50]}"
                # Check if next non-empty line is indented
                first_body_line = lines[i+1] if i+1 < len(lines) else ""
                if first_body_line.strip() and not first_body_line.startswith(' ') and not first_body_line.startswith('\t'):
                    return True, f"Function definition without proper body: {line[:50]}"
        
        # Check for unclosed brackets at end
        open_brackets = 0
        for char in last_chunk:
            if char in '([{':
                open_brackets += 1
            elif char in ')]}':
                open_brackets -= 1
        if open_brackets > 2:  # Some tolerance for multi-line statements
            return True, f"Unclosed brackets detected ({open_brackets} open)"
        
        return False, ""
    
    async def complete_truncated_script(script_path: Path, truncation_reason: str) -> bool:
        """
        Complete a script that was truncated during generation.
        Returns True if successfully completed.
        """
        nonlocal tokens_used
        
        with open(script_path, 'r') as f:
            code = f.read()
        
        # Get the last ~100 lines for context
        lines = code.split('\n')
        last_lines = '\n'.join(lines[-100:]) if len(lines) > 100 else code
        
        print(f"      🔧 Detected truncation: {truncation_reason}")
        print(f"      Asking Hinton to complete the script...")
        
        completion_prompt = f"""The following Python script was TRUNCATED during generation. Please provide the COMPLETION.

SCRIPT: {script_path.name}
TRUNCATION DETECTED: {truncation_reason}

=== LAST 100 LINES OF SCRIPT (where truncation occurred) ===
{last_lines}
=== END OF CURRENT CODE ===

INSTRUCTIONS:
1. Analyze where the script was cut off
2. Provide ONLY the code that completes it
3. Do NOT repeat any code already present
4. Ensure all functions are complete
5. Ensure the main block is present and complete
6. Match the indentation of the existing code

If the script ends mid-function, complete that function.
If the script is missing the if __name__ == "__main__" block, add it.
If the script is missing plt.savefig() calls, add them.

Provide ONLY the completion code in a ```python``` block."""

        completion_response = await hinton.arespond(completion_prompt, max_tokens=8000)
        tokens_used += TOKEN_PER_CALL
        
        # Extract the completion
        completion_code = extract_code_from_response(completion_response)
        
        if not completion_code or len(completion_code) < 20:
            print(f"      ⚠ No valid completion received")
            return False
        
        # Append the completion to the existing code
        completed_code = code.rstrip() + '\n' + completion_code
        
        # Validate the completed code
        is_valid, error_msg = validate_python_syntax(completed_code)
        
        if is_valid:
            with open(script_path, 'w') as f:
                f.write(completed_code)
            print(f"      ✓ Script completed successfully")
            return True
        else:
            # Try one more time with the error context
            print(f"      Completion had syntax error: {error_msg}, trying again...")
            retry_prompt = f"""The completion you provided had a syntax error. Please fix.

CURRENT SCRIPT END:
{last_lines[-50:]}

YOUR COMPLETION:
{completion_code[:500]}

SYNTAX ERROR: {error_msg}

Provide a CORRECTED completion in ```python``` block."""
            
            retry_response = await hinton.arespond(retry_prompt, max_tokens=8000)
            tokens_used += TOKEN_PER_CALL
            
            retry_code = extract_code_from_response(retry_response)
            if retry_code:
                completed_code = code.rstrip() + '\n' + retry_code
                is_valid, _ = validate_python_syntax(completed_code)
                if is_valid:
                    with open(script_path, 'w') as f:
                        f.write(completed_code)
                    print(f"      ✓ Script completed on retry")
                    return True
        
        return False
    
    def apply_edit_instruction(code: str, instruction: str) -> str:
        """
        Apply an edit instruction to code.
        
        Supported formats:
        - REPLACE_LINES start-end: new_code
        - INSERT_AFTER line: new_code  
        - DELETE_LINES start-end
        - REPLACE_STRING: old >>> new
        """
        lines = code.split('\n')
        
        # Try to parse REPLACE_LINES
        if instruction.startswith('REPLACE_LINES'):
            try:
                header, new_code = instruction.split(':', 1)
                range_part = header.replace('REPLACE_LINES', '').strip()
                if '-' in range_part:
                    start, end = map(int, range_part.split('-'))
                else:
                    start = end = int(range_part)
                new_lines = new_code.strip().split('\n')
                lines = lines[:start-1] + new_lines + lines[end:]
                return '\n'.join(lines)
            except:
                pass
        
        # Try INSERT_AFTER
        if instruction.startswith('INSERT_AFTER'):
            try:
                header, new_code = instruction.split(':', 1)
                line_num = int(header.replace('INSERT_AFTER', '').strip())
                new_lines = new_code.strip().split('\n')
                lines = lines[:line_num] + new_lines + lines[line_num:]
                return '\n'.join(lines)
            except:
                pass
        
        # Try DELETE_LINES
        if instruction.startswith('DELETE_LINES'):
            try:
                range_part = instruction.replace('DELETE_LINES', '').strip()
                if '-' in range_part:
                    start, end = map(int, range_part.split('-'))
                else:
                    start = end = int(range_part)
                lines = lines[:start-1] + lines[end:]
                return '\n'.join(lines)
            except:
                pass
        
        # Try REPLACE_STRING
        if instruction.startswith('REPLACE_STRING:'):
            try:
                content = instruction.replace('REPLACE_STRING:', '').strip()
                if '>>>' in content:
                    old, new = content.split('>>>', 1)
                    return code.replace(old.strip(), new.strip())
            except:
                pass
        
        return code  # Return unchanged if couldn't parse
    
    async def interactive_fix_script(script_path: Path, error_msg: str, max_iterations: int = 5) -> bool:
        """
        Interactively fix a script by having Hinton make targeted edits.
        Returns True if script was fixed successfully.
        """
        nonlocal tokens_used
        
        for iteration in range(max_iterations):
            # Read current code
            with open(script_path, 'r') as f:
                code = f.read()
            
            # Get line number from error if possible
            line_num = None
            if 'Line' in error_msg or 'line' in error_msg:
                import re
                match = re.search(r'[Ll]ine\s*(\d+)', error_msg)
                if match:
                    line_num = int(match.group(1))
            
            # Build context for Hinton
            if line_num:
                context = get_lines_around_error(code, line_num, context=15)
            else:
                # Show last 30 lines if no line number
                context = '\n'.join(f"{i+1:4d}    {line}" for i, line in enumerate(code.split('\n')[-30:]))
            
            fix_prompt = f"""Fix this Python script error. Make TARGETED edits only.

SCRIPT: {script_path.name}
ERROR: {error_msg}

CODE CONTEXT (line numbers shown):
{context}

RESPOND WITH ONE OF THESE EDIT COMMANDS:

1. To replace specific lines:
   REPLACE_LINES start-end:
   <new code here>

2. To insert after a line:
   INSERT_AFTER line_number:
   <new code here>

3. To delete lines:
   DELETE_LINES start-end

4. To replace a string:
   REPLACE_STRING: old_text >>> new_text

Example for fixing unterminated string on line 50:
REPLACE_LINES 50-50:
    print("This is the fixed string")

Provide ONLY the edit command, nothing else."""

            fix_response = await hinton.arespond(fix_prompt, max_tokens=2000)
            tokens_used += TOKEN_PER_CALL
            
            # Try to apply the edit
            new_code = apply_edit_instruction(code, fix_response.strip())
            
            if new_code == code:
                # Edit didn't work, try extracting code block
                if '```' in fix_response:
                    extracted = extract_code_from_response(fix_response)
                    if extracted and len(extracted) > len(code) * 0.5:
                        new_code = extracted
                
                # Still no change - use simpler direct prompt
                if new_code == code:
                    simple_prompt = f"""Fix this code that produced an error:

Code:
```python
{code}
```

Error: {error_msg}

Provide only the fixed code in a ```python``` block with no explanation."""
                    
                    simple_response = await hinton.arespond(simple_prompt, max_tokens=16000)
                    tokens_used += TOKEN_PER_CALL
                    extracted = extract_code_from_response(simple_response)
                    if extracted and len(extracted) > 100:
                        new_code = extracted
            
            # Write the updated code
            with open(script_path, 'w') as f:
                f.write(new_code)
            
            # Check syntax
            is_valid, new_error = validate_python_syntax(new_code)
            
            if is_valid:
                print(f"        ✓ Fixed after {iteration + 1} edit(s)")
                return True
            
            # Update error for next iteration
            error_msg = new_error
            print(f"        Edit {iteration + 1}: Still has error - {new_error[:50]}...")
        
        return False
    
    async def run_script_with_fixes(script_path: Path, max_fix_attempts: int = 5) -> tuple[bool, str]:
        """
        Run a script and interactively fix runtime errors.
        Returns (success, output/error).
        """
        nonlocal tokens_used
        import subprocess
        
        for attempt in range(max_fix_attempts):
            # Read current code
            with open(script_path, 'r') as f:
                code = f.read()
            
            # Check for truncation FIRST (before syntax)
            is_truncated, truncation_reason = detect_truncation(code)
            if is_truncated:
                print(f"      Truncation detected: {truncation_reason}")
                if await complete_truncated_script(script_path, truncation_reason):
                    # Re-read the completed code
                    with open(script_path, 'r') as f:
                        code = f.read()
                else:
                    return False, f"Could not complete truncated script: {truncation_reason}"
            
            # Now check syntax
            is_valid, syntax_error = validate_python_syntax(code)
            if not is_valid:
                print(f"      Syntax error detected, fixing...")
                if await interactive_fix_script(script_path, syntax_error):
                    continue  # Try running again
                else:
                    return False, f"Could not fix syntax error: {syntax_error}"
            
            # Run the script
            try:
                result = subprocess.run(
                    ["micromamba", "run", "-n", "minilab", "python", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(Path.cwd())
                )
                
                if result.returncode == 0:
                    return True, result.stdout
                
                # Runtime error - try to fix
                error_output = result.stderr or result.stdout
                print(f"      Runtime error, attempting fix {attempt + 1}/{max_fix_attempts}...")
                
                # Have Hinton analyze and fix the runtime error
                with open(script_path, 'r') as f:
                    current_code = f.read()
                
                # Extract line number from traceback if possible
                import re
                line_match = re.search(r'line (\d+)', error_output)
                line_num = int(line_match.group(1)) if line_match else None
                
                if line_num:
                    context = get_lines_around_error(current_code, line_num, context=10)
                else:
                    context = current_code[-2000:]
                
                # Use simple, direct prompt format
                fix_prompt = f"""Fix this code that produced an error:

Code:
```python
{current_code}
```

Error:
{error_output[:2000]}

Provide only the fixed code in a ```python``` block with no explanation."""

                fix_response = await hinton.arespond(fix_prompt, max_tokens=24000)
                tokens_used += TOKEN_PER_CALL
                
                # Extract the fixed code
                new_code = extract_code_from_response(fix_response)
                
                if not new_code or len(new_code) < 100:
                    # Fallback: try edit command approach
                    edit_prompt = f"""Fix this RUNTIME error. Respond with an edit command:

SCRIPT: {script_path.name}
ERROR: {error_output[:1000]}

RELEVANT CODE:
{context}

Use: REPLACE_LINES start-end: <new code>
Or: REPLACE_STRING: old >>> new"""
                    
                    edit_response = await hinton.arespond(edit_prompt, max_tokens=4000)
                    tokens_used += TOKEN_PER_CALL
                    new_code = apply_edit_instruction(current_code, edit_response.strip())
                    if new_code == current_code:
                        extracted = extract_code_from_response(edit_response)
                        if extracted:
                            new_code = extracted
                        else:
                            new_code = current_code  # No change
                
                with open(script_path, 'w') as f:
                    f.write(new_code)
                
            except subprocess.TimeoutExpired:
                return False, "Script timed out after 5 minutes"
            except Exception as e:
                return False, f"Execution error: {str(e)}"
        
        return False, "Could not fix after maximum attempts"
    
    # Script completeness markers - ensure scripts end properly
    def is_script_complete(code: str, script_name: str) -> bool:
        """Check if a script appears complete (not truncated)."""
        if not code or len(code) < 200:
            return False
        # Check for obvious truncation signs
        lines = code.strip().split('\n')
        last_line = lines[-1].strip() if lines else ''
        # Truncated if ends mid-statement
        truncated_endings = [':', ',', '(', '[', '{', '\\', 'def ', 'class ', 'if ', 'for ', 'while ']
        if any(last_line.endswith(e) for e in truncated_endings):
            return False
        # For figure scripts, must have savefig calls
        if 'figures' in script_name.lower():
            if 'savefig' not in code and 'save_fig' not in code:
                return False
        return True
    
    async def generate_script_iteratively(script_name: str, script_purpose: str, max_attempts: int = 3) -> tuple:
        """Generate a script with iterative refinement until syntax is valid.
        Returns (code, tokens_used)."""
        nonlocal tokens_used
        local_tokens = 0
        token_limit = script_token_limits.get(script_name, DEFAULT_SCRIPT_TOKENS)
        code = ""
        hinton_response = ""
        
        for attempt in range(max_attempts):
            if attempt == 0:
                # Initial generation - comprehensive prompt for complex scripts
                prompt = f"""Write a COMPLETE, COMPREHENSIVE Python script: {script_name}

PURPOSE: {script_purpose}
PROJECT: {project_name}

DATA FILES (in ReadData/Pluvicto/):
{file_list_str}

KEY DETAILS FROM ANALYSIS PLAN:
{detailed_plan[:3000]}

REQUIREMENTS:
- Read data from: ReadData/Pluvicto/
- Save outputs to: Sandbox/{project_name}/outputs/
- Set random seed: np.random.seed(42) 
- Include docstrings and progress print statements
- Use try/except for robust error handling
- Libraries: pandas, numpy, matplotlib, seaborn, scipy, sklearn, lifelines (if survival)
- MUST include `if __name__ == "__main__":` block
- For figure scripts: MUST include plt.savefig() for EACH panel

CRITICAL SYNTAX RULES:
- Close ALL strings (every quote has a matching quote)
- Match ALL brackets: (), [], {{}}
- Complete ALL function definitions
- No trailing colons without code blocks

Output ONLY the complete Python code in ```python``` block."""
                hinton_response = await hinton.arespond(prompt, max_tokens=token_limit)
                local_tokens += TOKEN_PER_CALL
            
            code = extract_code_from_response(hinton_response)
            
            if not code or len(code) < 200:
                print(f"      Attempt {attempt + 1}: Code extraction failed, retrying...")
                hinton_response = await hinton.arespond(
                    f"Generate {script_name} as a Python script in ```python``` blocks. Purpose: {script_purpose}",
                    max_tokens=token_limit
                )
                local_tokens += TOKEN_PER_CALL
                code = extract_code_from_response(hinton_response)
                if not code:
                    continue
            
            # Check for truncation FIRST
            is_truncated, trunc_reason = detect_truncation(code)
            if is_truncated:
                print(f"      Script appears truncated: {trunc_reason}")
                print(f"      Requesting completion...")
                
                # Write to temp for completion
                temp_path = scripts_dir / f".temp_{script_name}"
                with open(temp_path, 'w') as f:
                    f.write(code)
                
                if await complete_truncated_script(temp_path, trunc_reason):
                    with open(temp_path, 'r') as f:
                        code = f.read()
                    temp_path.unlink() if temp_path.exists() else None
                else:
                    temp_path.unlink() if temp_path.exists() else None
                    print(f"      Completion failed, will try to fix syntax...")
            
            # Validate syntax
            is_valid, error_msg = validate_python_syntax(code)
            
            if is_valid:
                return code, local_tokens
            
            # Syntax error - use interactive fixing
            print(f"      Syntax error: {error_msg}")
            print(f"      Attempting interactive fix...")
            
            # Write to temp location for interactive fixing
            temp_path = scripts_dir / f".temp_{script_name}"
            with open(temp_path, 'w') as f:
                f.write(code)
            
            if await interactive_fix_script(temp_path, error_msg, max_iterations=3):
                with open(temp_path, 'r') as f:
                    code = f.read()
                temp_path.unlink()  # Clean up temp file
                return code, local_tokens
            
            temp_path.unlink() if temp_path.exists() else None
            
            # If interactive fix failed, try regeneration
            print(f"      Interactive fix failed, regenerating...")
            hinton_response = await hinton.arespond(
                f"""The previous script had syntax errors. Generate a NEW, WORKING version of {script_name}.

Purpose: {script_purpose}
Project: {project_name}

CRITICAL: Ensure ALL strings are closed, ALL brackets matched, ALL functions complete.

Output ONLY Python code in ```python``` block.""",
                max_tokens=token_limit
            )
            local_tokens += TOKEN_PER_CALL
        
        # Return whatever we have after max attempts
        return code if code else "", local_tokens
    
    scripts_generated = []
    scripts_failed = []
    
    for script_name, script_purpose in expected_scripts:
        script_path = scripts_dir / script_name
        print(f"  Generating {script_name}...")
        if logger:
            logger.log_system_event("SCRIPT_GEN", f"Starting generation of {script_name}", {"purpose": script_purpose[:100]})
        
        try:
            # Use iterative generation with syntax validation
            code, script_tokens = await generate_script_iteratively(script_name, script_purpose)
            tokens_used += script_tokens
            
            if code:
                # Final syntax check
                is_valid, error_msg = validate_python_syntax(code)
                
                with open(script_path, 'w') as f:
                    f.write(code)
                file_size = script_path.stat().st_size
                
                if is_valid:
                    print(f"    ✓ Created {script_name} ({file_size} bytes) [syntax valid]")
                    scripts_generated.append(script_name)
                    if logger:
                        logger.log_system_event("SCRIPT_GEN", f"✓ Created {script_name}", {"bytes": file_size, "syntax": "valid"})
                else:
                    print(f"    ⚠ Created {script_name} ({file_size} bytes) [WARNING: {error_msg}]")
                    scripts_generated.append(script_name)
                    if logger:
                        logger.log_system_event("SCRIPT_GEN", f"⚠ Created {script_name} with syntax issues", {"bytes": file_size, "error": error_msg})
            else:
                print(f"    ✗ Failed to generate {script_name}")
                scripts_failed.append(script_name)
                if logger:
                    logger.log_system_event("SCRIPT_GEN", f"✗ FAILED to generate {script_name}", {"error": "No code produced"})
        except Exception as e:
            print(f"    ✗ Exception generating {script_name}: {e}")
            scripts_failed.append(script_name)
            if logger:
                logger.log_system_event("SCRIPT_GEN", f"✗ EXCEPTION generating {script_name}", {"error": str(e)})
    
    # Summary
    if logger:
        logger.log_system_event("SCRIPT_GEN_SUMMARY", f"Generated {len(scripts_generated)}/{len(expected_scripts)} scripts", 
                               {"generated": scripts_generated, "failed": scripts_failed})
    
    # Verify all scripts exist and have valid syntax
    print("\n  Verifying scripts...")
    if logger:
        logger.log_system_event("VERIFICATION", "Verifying all scripts exist and have valid syntax")
    
    all_valid = True
    verification_results = {}
    for script_name, _ in expected_scripts:
        script_path = scripts_dir / script_name
        if script_path.exists():
            with open(script_path, 'r') as f:
                content = f.read()
            is_valid, error_msg = validate_python_syntax(content)
            if is_valid:
                print(f"    ✓ {script_name} [syntax OK]")
                verification_results[script_name] = "OK"
            else:
                print(f"    ⚠ {script_name} [syntax error: {error_msg}]")
                verification_results[script_name] = f"ERROR: {error_msg}"
                all_valid = False
        else:
            print(f"    ✗ {script_name} - MISSING")
            verification_results[script_name] = "MISSING"
            all_valid = False
    
    if logger:
        logger.log_system_event("VERIFICATION_RESULT", f"Scripts verified: {sum(1 for v in verification_results.values() if v == 'OK')}/{len(expected_scripts)}", verification_results)
    
    if not all_valid:
        print("\n  ⚠ Some scripts have issues. Will attempt fixes during execution...")
    
    print()
    
    # 3B: Bayes code review - read scripts and get feedback
    _print_substage("3B: Code Review (Bayes)")
    if logger:
        logger.log_system_event("SUBSTAGE", "3B: Code Review (Bayes)")
    
    print("  Bayes reviewing scripts...")
    
    # Collect actual script contents for review
    script_contents = {}
    for script_name, _ in expected_scripts:
        script_path = scripts_dir / script_name
        if script_path.exists():
            with open(script_path, 'r') as f:
                content = f.read()
            script_contents[script_name] = content
    
    if not script_contents:
        print("  ⚠ No scripts to review!")
        if logger:
            logger.log_system_event("WARNING", "No scripts available for Bayes review - all scripts failed to generate")
    else:
        if logger:
            logger.log_system_event("CODE_REVIEW", f"Bayes reviewing {len(script_contents)} scripts", {"scripts": list(script_contents.keys())})
        
        # Show first 1500 chars of each for review
        scripts_summary = "\n\n".join([
            f"=== {name} ===\n{content[:1500]}..." 
            for name, content in script_contents.items()
        ])
        
        bayes_review = await bayes.arespond(f"""Review these analysis scripts for {project_name}.

{scripts_summary}

Check for:
1. CORRECTNESS - syntax, file paths (ReadData/Pluvicto/ for input, Sandbox/{project_name}/outputs/ for output)
2. STATISTICAL VALIDITY - appropriate tests, multiple testing corrections where needed
3. REPRODUCIBILITY - random seeds set, parameters documented
4. COMPLETENESS - scripts save outputs needed by downstream scripts

Respond with:
- APPROVE: [brief summary] if scripts look good
- ISSUES: [numbered list of specific problems with script names]""")
        tokens_used += TOKEN_PER_CALL
        
        _show_agent("Bayes", bayes_review)
        
        # If issues found, have Hinton generate fixed versions
        if "ISSUES" in bayes_review.upper():
            print("\n  Addressing code review issues...")
            
            for script_name, content in script_contents.items():
                # Check if this script has issues mentioned
                if script_name.lower() in bayes_review.lower():
                    print(f"    Fixing {script_name}...")
                    
                    fix_response = await hinton.arespond(f"""Fix this script based on Bayes's review:

ISSUES FOUND:
{bayes_review}

CURRENT SCRIPT ({script_name}):
```python
{content}
```

Return the COMPLETE FIXED script in ```python ``` code blocks.
Address all issues mentioned for this script.""")
                    tokens_used += TOKEN_PER_CALL
                    
                    fixed_code = extract_code_from_response(fix_response)
                    if fixed_code and len(fixed_code) > 100:
                        script_path = scripts_dir / script_name
                        with open(script_path, 'w') as f:
                            f.write(fixed_code)
                        print(f"      ✓ Fixed {script_name}")
                    else:
                        print(f"      ⚠ Could not extract fixed code for {script_name}")
    
    print("  Code review complete.\n")
    if logger:
        logger.log_agent_response("Bayes", "bayes", bayes_review, TOKEN_PER_CALL)
    
    # 3C: Execute scripts using INTERACTIVE FIXING
    _print_substage("3C: Script Execution (Interactive)")
    if logger:
        logger.log_system_event("SUBSTAGE", "3C: Script Execution with Interactive Fixing")
    
    print("  Executing scripts with interactive error fixing...\n")
    
    execution_results = {}
    
    for script_name, _ in expected_scripts:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            print(f"    ⚠ Skipping {script_name} - file not found")
            execution_results[script_name] = "SKIPPED - file not found"
            if logger:
                logger.log_system_event("EXECUTION", f"⚠ Skipping {script_name} - file not found")
            continue
        
        print(f"    Running {script_name}...")
        if logger:
            logger.log_system_event("EXECUTION", f"Running {script_name}")
        
        # Use the new interactive fixing system
        success, output = await run_script_with_fixes(script_path, max_fix_attempts=5)
        
        if success:
            print(f"      ✓ Completed successfully")
            execution_results[script_name] = "SUCCESS"
            if logger:
                logger.log_system_event("EXECUTION", f"✓ {script_name} completed successfully")
            if output:
                # Show last few lines of output
                lines = output.strip().split('\n')
                if len(lines) > 5:
                    print(f"      ... (showing last 5 lines)")
                for line in lines[-5:]:
                    print(f"        {line[:100]}")
        else:
            print(f"      ✗ Script failed: {output[:200]}")
            execution_results[script_name] = f"FAILED: {output[:100]}"
            if logger:
                logger.log_system_event("EXECUTION", f"✗ {script_name} FAILED", {"error": output[:500]})
    
    # Execution summary
    if logger:
        successful = sum(1 for v in execution_results.values() if "SUCCESS" in v)
        logger.log_system_event("EXECUTION_SUMMARY", f"Scripts executed: {successful}/{len(expected_scripts)} successful", execution_results)
    
    print()
    
    # =========================================================================
    # VERIFICATION GATE: Check outputs before proceeding to Stage 4
    # =========================================================================
    print("  Verifying outputs...")
    if logger:
        logger.log_system_event("VERIFICATION_GATE", "Checking Stage 3 outputs before proceeding")
    
    # Check what exists in outputs directory
    outputs_dir = project_path / "outputs"
    outputs_exist = outputs_dir.exists()
    output_files = list(outputs_dir.glob("*")) if outputs_exist else []
    
    # Check for PNG/PDF figures
    png_files = list(outputs_dir.glob("*.png")) if outputs_exist else []
    png_files.extend(list(project_path.glob("*.png")))
    
    figures_pdf = project_path / f"{project_name}_figures.pdf"
    pdf_exists = figures_pdf.exists()
    
    # Check for CSV/data outputs
    csv_files = list(outputs_dir.glob("*.csv")) if outputs_exist else []
    json_files = list(outputs_dir.glob("*.json")) if outputs_exist else []
    
    output_summary = {
        "outputs_dir_exists": outputs_exist,
        "total_output_files": len(output_files),
        "png_files": len(png_files),
        "pdf_exists": pdf_exists,
        "csv_files": len(csv_files),
        "json_files": len(json_files),
    }
    
    print(f"    Outputs directory: {'exists' if outputs_exist else 'MISSING'}")
    print(f"    PNG figures: {len(png_files)}")
    print(f"    Final PDF: {'exists' if pdf_exists else 'NOT FOUND'}")
    print(f"    CSV data files: {len(csv_files)}")
    
    if logger:
        logger.log_system_event("OUTPUT_CHECK", "Stage 3 output verification", output_summary)
    
    # Determine if we have enough to proceed
    has_figures = pdf_exists or len(png_files) > 0
    has_data = len(csv_files) > 0 or len(json_files) > 0
    
    if not has_figures:
        print("\n  ⚠️ WARNING: No figures were generated!")
        print("  Stage 4 will attempt to review but may have limited content.")
        if logger:
            logger.log_system_event("WARNING", "No figures generated - Stage 4 may have limited content")
    
    if not has_data:
        print("  ⚠️ WARNING: No data output files found!")
        if logger:
            logger.log_system_event("WARNING", "No data output files (CSV/JSON) found")
    
    if pdf_exists:
        print(f"\n  ✓ SUCCESS: {project_name}_figures.pdf created.\n")
    else:
        print(f"\n  Note: {project_name}_figures.pdf not found - check manually.\n")
    
    # =========================================================================
    # STAGE 4: WRITE-UP
    # =========================================================================
    _print_stage("STAGE 4", "Write-up")
    if logger:
        logger.log_stage_transition("Stage 4", "Documentation")
    
    # 4A: Bohr reviews figures WITH VISION
    _print_substage("4A: Figure Review (Bohr)")
    if logger:
        logger.log_system_event("SUBSTAGE", "4A: Figure Review (Bohr)")
    
    print("  Bohr reviewing generated figures...")
    
    # Read figure generation script for context
    fig_script = await _agent_reads_file(bohr, f"Sandbox/{project_name}/scripts/04_figures.py")
    
    # Try to use vision if PDF exists
    if figures_pdf.exists():
        print(f"  📄 Bohr viewing {project_name}_figures.pdf with vision...")
        if logger:
            logger.log_system_event("VISION", f"Bohr viewing {figures_pdf}")
        
        bohr_fig_review = await bohr.arespond_with_vision(
            user_message=f"""I'm reviewing the generated figures for {project_name}.

I can now SEE the actual figure PDF. Please examine each panel carefully.

Here's the figure generation script for context:
{fig_script[:1500]}

For each panel/figure I can see, please describe:
1. What type of visualization it is (scatter plot, KM curve, heatmap, forest plot, etc.)
2. What data/variables are being shown
3. Are axes labeled clearly? Is there a legend?
4. Are the colors/formatting appropriate?
5. Any statistical annotations (p-values, confidence intervals)?
6. Any issues that need fixing?

Respond with:
- LOOKS_GOOD: [detailed description of each panel and what it shows]
- NEEDS_FIXES: [specific issues with panel numbers and what to fix]""",
            pdf_path=str(figures_pdf),
            max_tokens=4000
        )
    elif png_files:
        # Fall back to viewing individual PNGs
        print(f"  🖼️ Bohr viewing {len(png_files)} PNG file(s)...")
        if logger:
            logger.log_system_event("VISION", f"Bohr viewing {len(png_files)} PNG files")
        
        bohr_fig_review = await bohr.arespond_with_vision(
            user_message=f"""I'm reviewing the generated figure panels for {project_name}.

I can see the individual PNG panel images. Please examine each one carefully.

For each panel I can see, describe:
1. What type of visualization it is
2. What data is shown
3. Quality of labels, legends, formatting
4. Any issues to fix

Respond with LOOKS_GOOD: or NEEDS_FIXES:""",
            image_paths=[str(p) for p in png_files[:6]],  # Max 6 images
            max_tokens=4000
        )
    else:
        # No images available - use text-only review
        print("  ⚠ No figures found for visual review. Using script-based review...")
        
        bohr_fig_review = await bohr.arespond(f"""I need to review the figures for {project_name}, but no figure files were found.

The figure script is:
{fig_script[:2500]}

Looking at this script:
1. Does it appear complete? Does it have savefig() calls?
2. What panels should it generate?
3. Are there obvious bugs or issues?

Respond with:
- SCRIPT_OK: [description of what panels should be generated]
- NEEDS_FIXES: [specific issues with the script]""")
    
    tokens_used += TOKEN_PER_CALL
    
    if logger:
        logger.log_agent_response("Bohr", "bohr", bohr_fig_review, TOKEN_PER_CALL)
    
    _show_agent("Bohr", bohr_fig_review)
    
    # Handle figure fixes if needed - with orchestrator executing the fixes
    fix_iter = 0
    while "NEEDS_FIXES" in bohr_fig_review.upper() and fix_iter < 3:
        fix_iter += 1
        print(f"\n  Hinton fixing figures (iteration {fix_iter})...")
        
        # Get the current figure script
        fig_script_path = scripts_dir / "04_figures.py"
        current_script = ""
        if fig_script_path.exists():
            with open(fig_script_path, 'r') as f:
                current_script = f.read()
        
        # Have Hinton fix the script
        hinton_fix = await hinton.arespond(f"""Bohr reviewed the figures and found issues:

{bohr_fig_review}

CURRENT SCRIPT:
```python
{current_script[:4000]}
```

Please provide the COMPLETE FIXED script. Address all issues mentioned.
Output ONLY the Python code in ```python ``` blocks.""", max_tokens=8000)
        tokens_used += TOKEN_PER_CALL
        
        # Extract and save the fixed script
        fixed_code = extract_code_from_response(hinton_fix)
        if fixed_code and len(fixed_code) > 200:
            with open(fig_script_path, 'w') as f:
                f.write(fixed_code)
            print(f"    ✓ Updated 04_figures.py ({len(fixed_code)} bytes)")
            
            # Re-run the figure generation script
            print(f"    Running updated figure script...")
            try:
                result = subprocess.run(
                    ["micromamba", "run", "-n", "minilab", "python", str(fig_script_path)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(Path.cwd())
                )
                if result.returncode == 0:
                    print(f"    ✓ Figure script completed")
                else:
                    print(f"    ⚠ Figure script failed: {result.stderr[:300]}")
            except Exception as e:
                print(f"    ⚠ Execution error: {e}")
        else:
            print(f"    ⚠ Could not extract fixed code")
        
        # Re-review with vision
        if figures_pdf.exists():
            bohr_fig_review = await bohr.arespond_with_vision(
                user_message="Re-review the figures after Hinton's fixes. Look carefully at each panel. Use LOOKS_GOOD: or NEEDS_FIXES:",
                pdf_path=str(figures_pdf),
                max_tokens=3000
            )
        else:
            bohr_fig_review = await bohr.arespond("Re-review the figure script. Use LOOKS_GOOD: or NEEDS_FIXES:")
        tokens_used += TOKEN_PER_CALL
        
        _show_agent("Bohr", bohr_fig_review)
    
    fig_description = bohr_fig_review
    if "LOOKS_GOOD" in bohr_fig_review.upper():
        fig_description = bohr_fig_review.split("LOOKS_GOOD", 1)[1] if "LOOKS_GOOD" in bohr_fig_review.upper() else bohr_fig_review
    
    print("  Figures approved.\n")
    
    # 4B: Gould writes documents
    _print_substage("4B: Document Generation (Gould)")
    
    print("  Gould creating legends and summary documents...")
    
    gould_legends = await gould.arespond(f"""I need to write figure legends for {project_name}.

FIGURE DESCRIPTION FROM BOHR:
{fig_description}

DETAILED PLAN:
{detailed_plan}

I need to create {project_name}_legends.md with journal-quality figure legends:

## Figure Legends

**Figure 1. [Overall title]**

**(a)** [Detailed legend for panel a. Include what is shown, sample size (n=X), statistical test used, significance levels (* p<0.05, ** p<0.01, etc.), and key takeaway.]

**(b)** [Detailed legend for panel b...]

**(c)** [...]

Continue for all panels (d, e, f as applicable).

Follow Nature/Science legend format - be specific about methods, sample sizes, and statistical tests.""")
    tokens_used += TOKEN_PER_CALL
    
    if logger:
        logger.log_agent_response("Gould", "gould", gould_legends, TOKEN_PER_CALL)
    
    _show_agent("Gould", gould_legends)
    
    # Write legends file
    await _agent_writes_file(gould, f"Sandbox/{project_name}/{project_name}_legends.md", gould_legends)
    print(f"  Created: Sandbox/{project_name}/{project_name}_legends.md")
    
    # Now write summary
    gould_summary = await gould.arespond(f"""Now I need to write the summary document for {project_name}.

RESEARCH QUESTION: "{research_question}"
FIGURE DESCRIPTION: {fig_description}
DETAILED PLAN: {detailed_plan}
LITERATURE REVIEW: {gould_response}

Create {project_name}_summary.md with THREE sections:

# {project_name} Summary

## Discussion

[2-3 paragraphs that:
- Reference EVERY figure panel by letter (a, b, c, etc.)
- State the hypotheses that were tested
- Discuss whether hypotheses were supported/refuted
- Discuss potential mechanisms
- Include AT LEAST 5 citations with DOIs]

## Methods

[Detailed methods section covering:
- Data sources and preprocessing
- Statistical tests used with parameters
- Software and libraries
- Reference to scripts (e.g., "see 03_analysis.py")
- Reproducibility information (random seeds, etc.)]

## Citations

[List of at least 5 citations in Nature format:
Author1, Author2, et al. Title. Journal Volume, Pages (Year). DOI: https://doi.org/...]

Make sure all citations are real papers with actual DOIs.""")
    tokens_used += TOKEN_PER_CALL
    
    if logger:
        logger.log_agent_response("Gould", "gould", gould_summary, TOKEN_PER_CALL)
    
    _show_agent("Gould", gould_summary)
    
    # Write summary file
    await _agent_writes_file(gould, f"Sandbox/{project_name}/{project_name}_summary.md", gould_summary)
    print(f"  Created: Sandbox/{project_name}/{project_name}_summary.md\n")
    
    print("  Stage 4 complete: Documents created.\n")
    
    # =========================================================================
    # STAGE 5: CRITICAL REVIEW
    # =========================================================================
    _print_stage("STAGE 5", "Critical Review")
    if logger:
        logger.log_stage_transition("Stage 5", "Final review")
    
    print("  Farber conducting critical review of all outputs...")
    
    # Read the documents directly (orchestrator has filesystem access)
    legends_path = project_path / f"{project_name}_legends.md"
    summary_path = project_path / f"{project_name}_summary.md"
    
    legends_content = "[Legends file not found]"
    if legends_path.exists():
        with open(legends_path, 'r') as f:
            legends_content = f.read()
    
    summary_content = "[Summary file not found]"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary_content = f.read()
    
    # Also read the analysis script to verify methods
    analysis_script_content = "[Analysis script not found]"
    analysis_script_path = project_path / "scripts" / "03_analysis.py"
    if analysis_script_path.exists():
        with open(analysis_script_path, 'r') as f:
            analysis_script_content = f.read()[:3000]  # First 3000 chars
    
    # Read figure generation script
    figures_script_content = "[Figures script not found]"
    figures_script_path = project_path / "scripts" / "04_figures.py"
    if figures_script_path.exists():
        with open(figures_script_path, 'r') as f:
            figures_script_content = f.read()[:3000]
    
    farber_final_review = await farber.arespond(f"""As the critical reviewer, I need to evaluate all final outputs for {project_name}.

RESEARCH QUESTION: "{research_question}"

=== FIGURE LEGENDS ===
{legends_content}

=== SUMMARY DOCUMENT ===
{summary_content}

=== ANALYSIS SCRIPT (excerpt) ===
{analysis_script_content}

=== FIGURE GENERATION SCRIPT (excerpt) ===
{figures_script_content}

Please evaluate:

1. SOURCE VALIDITY
   - Are the citations real and relevant?
   - Are DOIs plausible (format correct)?
   - Is the literature appropriately comprehensive?

2. CONCLUSION ACCURACY  
   - Do the stated results support the conclusions?
   - Are claims appropriately hedged?
   - Are alternative explanations considered?

3. METHOD CORRECTNESS
   - Are statistical tests appropriate for the data?
   - Are assumptions stated and checked?
   - Is the analysis reproducible?

4. PRESENTATION QUALITY
   - Are figures clear and well-labeled (based on script)?
   - Are legends complete and informative?
   - Is writing clear and professional?

5. CLINICAL/SCIENTIFIC RELEVANCE
   - Does this address clinically meaningful questions about Pluvicto response?
   - Are implications reasonable?

Respond with:
- ACCEPTABLE: [strengths and any minor suggestions]
- UNACCEPTABLE: [specific issues that must be addressed]""")
    tokens_used += TOKEN_PER_CALL
    
    if logger:
        logger.log_agent_response("Farber", "farber", farber_final_review, TOKEN_PER_CALL)
    
    _show_agent("Farber", farber_final_review)
    
    # =========================================================================
    # USER FINAL DECISION
    # =========================================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n  Project: {project_name}")
    print(f"  Location: Sandbox/{project_name}/")
    print(f"\n  Outputs:")
    print(f"    - {project_name}_figures.pdf")
    print(f"    - {project_name}_legends.md")
    print(f"    - {project_name}_summary.md")
    print(f"    - scratch/ (intermediate files)")
    print(f"    - scripts/ (analysis scripts)")
    print("=" * 70 + "\n")
    
    user_input = _get_user_input("Accept results or request revision? (accept/revise)")
    
    iteration_round = 0
    while "revise" in user_input.lower() and iteration_round < 3:
        iteration_round += 1
        print(f"\n  --- ITERATION {iteration_round} ---\n")
        
        feedback = _get_user_input("Please describe what should be revised:")
        
        # Combine user and Farber feedback
        farber_issues = ""
        if "UNACCEPTABLE" in farber_final_review.upper():
            farber_issues = farber_final_review.split("UNACCEPTABLE", 1)[1]
        
        combined_feedback = f"USER FEEDBACK:\n{feedback}\n\nFARBER'S CONCERNS:\n{farber_issues}"
        
        print("\n  Re-entering Stage 2 with feedback...")
        
        # Bohr integrates feedback
        bohr_update = await bohr.arespond(f"""We need to revise the analysis based on feedback.

COMBINED FEEDBACK:
{combined_feedback}

EXISTING WORK:
- Scripts: Sandbox/{project_name}/scripts/
- Detailed plan: Sandbox/{project_name}/scratch/detailed_plan.md

I need to:
1. Synthesize the feedback into specific changes
2. Update the detailed plan
3. Coordinate with the team to implement revisions

What specific changes should we make?""")
        tokens_used += TOKEN_PER_CALL
        
        _show_agent("Bohr", bohr_update)
        
        # Quick iteration through stages
        print("  Updating implementation...")
        await dayhoff.arespond(f"Bohr's revision plan:\n{bohr_update}\n\nUpdate implementation plan accordingly.")
        tokens_used += TOKEN_PER_CALL
        
        print("  Updating scripts...")
        await hinton.arespond(f"Updated plan:\n{bohr_update}\n\nRevise scripts and re-run analysis.")
        tokens_used += TOKEN_PER_CALL
        
        print("  Updating documents...")
        await gould.arespond(f"Updated plan:\n{bohr_update}\n\nRevise legends and summary documents.")
        tokens_used += TOKEN_PER_CALL
        
        print("  Re-reviewing...")
        farber_final_review = await farber.arespond(f"""Please re-review all outputs after revision.

Feedback that was addressed:
{combined_feedback}

Check if the issues have been resolved.""")
        tokens_used += TOKEN_PER_CALL
        
        _show_agent("Farber", farber_final_review)
        
        user_input = _get_user_input("Accept results or request further revision? (accept/revise)")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("  WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\n  Project: {project_name}")
    print(f"  Location: Sandbox/{project_name}/")
    print(f"  Files analyzed: {len(files)}")
    print(f"  Estimated tokens used: ~{tokens_used:,}")
    print("=" * 70 + "\n")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path.cwd() / "Outputs"
    output_path = output_dir / project_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    return {
        "success": True,
        "project_name": project_name,
        "project_path": str(project_path),
        "output_dir": str(output_path),
        "output_path": str(output_path),
        "research_question": research_question,
        "files_analyzed": len(files),
        "files": files,
        "tokens_used": tokens_used,
        "token_count": tokens_used,
    }
