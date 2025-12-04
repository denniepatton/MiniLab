"""
Single Analysis Orchestrator v2 - Complete Rewrite per Specification

This orchestrator implements the full MiniLab Single Analysis workflow with:
- 7 stages (0-6) with proper agent routing
- Exploratory (Stage 3) vs Complete (Stage 4) execution paths
- User checkpoints at each stage
- Version tracking for documents
- Parallel agent consultation where beneficial
- Cross-agent consultation capability

STAGE 0: Confirm files and project naming (Bohr ↔ User)
STAGE 1: Build project structure and summarize inputs (Bohr → User)
STAGE 2: Plan full analysis
    2A: Synthesis Core (Bohr → Gould → Farber → Bohr)
    2B: Theory Core (Bohr → [Feynman, Shannon, Greider] → Bohr)
    2C: Implementation Core (Bohr → Dayhoff → EXECUTIONPLAN)
STAGE 3: Exploratory Execution (if needed)
    (Dayhoff → Hinton → Bayes → Dayhoff → loop or → Stage 2C)
STAGE 4: Complete Execution
    (Dayhoff → Hinton → Bayes → Dayhoff → loop or → Stage 5)
STAGE 5: Write-up (Bohr → Gould)
STAGE 6: Review (Farber → User → possibly iterate from Stage 2)

PRIMARY OUTPUTS (required):
- XXX_figures.pdf: 4-6 panel Nature-style figure (8.5x11")
- XXX_legends.pdf/.md/.docx: Nature-style figure legends
- XXX_summary.pdf/.md/.docx: Mini-paper (Intro, Discussion, Methods, References)
"""

from __future__ import annotations
import asyncio
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

from MiniLab.agents.base import Agent
from MiniLab.storage.transcript import TranscriptLogger


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

TOKEN_PER_CALL = 2000  # Rough estimate for token tracking
SCRIPT_TOKEN_LIMIT = 24000  # Max tokens for script generation
FIX_TOKEN_LIMIT = 24000  # Max tokens for code fixes

# Primary output files
PRIMARY_OUTPUTS = [
    "{project}_figures.pdf",
    "{project}_legends.md",  # Can also be .pdf or .docx
    "{project}_summary.md",  # Can also be .pdf or .docx
]


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
    clean = message
    clean = re.sub(r'DELEGATE:.*?---END', '', clean, flags=re.DOTALL)
    clean = re.sub(r'\[TOOL RESULT\].*?(?=\n\n|\Z)', '', clean, flags=re.DOTALL)
    clean = re.sub(r'<[^>]+>', '', clean)
    clean = '\n'.join(line for line in clean.split('\n') if line.strip())
    
    if truncate > 0 and len(clean) > truncate:
        clean = clean[:truncate] + "\n  [...truncated for display...]"
    
    print(f"  [{agent_name}]:")
    for line in clean.split('\n'):
        print(f"    {line}")
    print()


def _get_user_input(prompt: str) -> str:
    """Get input from user with a prompt."""
    print(f"\n  {prompt}")
    return input("  > ").strip()


def _user_approves(response: str) -> bool:
    """Check if user response indicates approval."""
    approvals = ['yes', 'y', 'correct', 'good', 'fine', 'ok', 'proceed', 
                 'looks good', 'approved', 'accept', 'continue', 'go ahead',
                 'sounds good', 'perfect', 'great', 'agreed']
    return any(word in response.lower() for word in approvals)


def _user_rejects(response: str) -> bool:
    """Check if user response indicates rejection/issues."""
    rejections = ['no', 'wrong', 'incorrect', 'bad', 'issue', 'problem',
                  'missing', 'fix', 'change', 'redo', 'again', 'not right']
    return any(word in response.lower() for word in rejections)


async def _ask_user_permission(action_description: str) -> bool:
    """
    Ask user for permission before performing an action (e.g., package install).
    
    Args:
        action_description: Description of the action requiring permission
        
    Returns:
        True if user approves, False otherwise
    """
    print(f"\n  ⚠️  PERMISSION REQUIRED")
    print(f"  An agent wants to: {action_description}")
    response = _get_user_input("Do you approve? (yes/no)")
    return _user_approves(response)


async def _discover_files(agent: Agent, base_path: str) -> List[str]:
    """Recursively discover all files in a directory using agent's filesystem tool."""
    discovered = []
    base_path = base_path.rstrip('/')
    to_explore = [base_path]
    explored = set()
    
    while to_explore:
        current = to_explore.pop(0)
        if current in explored:
            continue
        explored.add(current)
        
        result = await agent.use_tool("filesystem", action="list", path=current)
        
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


async def _agent_reads_file(agent: Agent, path: str, lines: int = None) -> str:
    """Have agent read a file and return its content."""
    if lines:
        result = await agent.use_tool("filesystem", action="head", path=path, lines=lines)
    else:
        result = await agent.use_tool("filesystem", action="read", path=path)
    
    if result.get("success"):
        return result.get("content", "")
    return f"[Error reading {path}: {result.get('error', 'Unknown')}]"


async def _agent_writes_file(agent: Agent, path: str, content: str) -> bool:
    """Have agent write content to a file."""
    result = await agent.use_tool("filesystem", action="write", path=path, content=content)
    return result.get("success", False)


def _extract_project_name(text: str) -> Optional[str]:
    """Extract CamelCase project name from text."""
    patterns = [
        r"\*\*Project\s*Name[:\s]+([A-Z][a-zA-Z0-9]+)\*\*",
        r"Project\s*Name[:\s]+`([A-Z][a-zA-Z0-9]+)`",
        r"Project\s*Name[:\s]+([A-Z][a-zA-Z0-9]+)",
        r"`([A-Z][a-zA-Z0-9]{6,})`",
        r"\*\*([A-Z][a-zA-Z0-9]{6,})\*\*",
        r"suggest(?:ed)?[:\s]+[\"'`*]*([A-Z][a-zA-Z0-9]+)[\"'`*]*",
        r"name[:\s]+[\"'`*]*([A-Z][a-zA-Z0-9]+)[\"'`*]*",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for candidate in matches:
            candidate = candidate.strip('*"\'` ')
            capitals = sum(1 for c in candidate if c.isupper())
            if capitals >= 2 and 6 < len(candidate) < 50:
                return candidate
    
    # Fallback: find any CamelCase word
    camel_matches = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)\b', text)
    for candidate in camel_matches:
        if len(candidate) > 6:
            return candidate
    
    return None


def _get_version_path(base_path: Path, filename: str, version: int) -> Path:
    """Get versioned file path (e.g., WORKINGPLAN_v2.md)."""
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    return base_path / f"{stem}_v{version}{suffix}"


def _get_latest_version(base_path: Path, filename_pattern: str) -> Tuple[int, Optional[Path]]:
    """Find the latest version of a file matching pattern."""
    stem = Path(filename_pattern).stem
    suffix = Path(filename_pattern).suffix
    
    versions = []
    for f in base_path.glob(f"{stem}_v*{suffix}"):
        match = re.search(r'_v(\d+)', f.stem)
        if match:
            versions.append((int(match.group(1)), f))
    
    if versions:
        versions.sort(reverse=True)
        return versions[0]
    
    # Check for unversioned file
    unversioned = base_path / filename_pattern
    if unversioned.exists():
        return (0, unversioned)
    
    return (0, None)


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """Check Python syntax without executing. Returns (is_valid, error_message)."""
    import ast
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def extract_code_from_response(response: str) -> str:
    """Extract Python code from a response that may contain markdown code blocks."""
    # Try to find ```python ... ``` blocks
    pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Try to find ``` ... ``` blocks (any language)
    pattern = r'```\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # If no code blocks, return the whole response if it looks like code
    if 'import ' in response or 'def ' in response or 'class ' in response:
        return response.strip()
    
    return ""


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

async def run_single_analysis(
    agents: Dict[str, Agent],
    research_question: str,
    max_tokens: int = 2_000_000,
    logger: Optional[TranscriptLogger] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute the complete MiniLab Single Analysis workflow.
    
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
    print("  MINILAB SINGLE ANALYSIS WORKFLOW v2")
    print("=" * 70)
    print(f"\n  Research Question:\n    {research_question}\n")
    print("=" * 70 + "\n")
    
    # Agent references - all 9 agents
    bohr = agents["bohr"]
    farber = agents["farber"]
    gould = agents["gould"]
    feynman = agents["feynman"]
    shannon = agents["shannon"]
    greider = agents["greider"]
    bayes = agents["bayes"]
    hinton = agents["hinton"]
    dayhoff = agents["dayhoff"]
    
    # Set up cross-agent consultation for all agents
    for agent in agents.values():
        agent.set_colleagues(agents)
    
    # Token tracking
    tokens_used = 0
    
    # Workflow state
    project_name = None
    files = []
    manifest = ""
    working_plan = ""
    working_plan_version = 0
    citations = []
    execution_plan = ""
    execution_plan_type = None  # "exploratory" or "complete"
    
    # =========================================================================
    # STAGE 0: CONFIRM FILES AND PROJECT NAMING
    # =========================================================================
    _print_stage("STAGE 0", "Confirm Files and Project Naming")
    if logger:
        logger.log_stage_transition("Stage 0", "Files and naming")
    
    # Extract target directory from research question
    dir_match = re.search(r'ReadData/[\w/]+', research_question)
    target_dir = dir_match.group(0) if dir_match else "ReadData"
    
    print(f"  Target directory: {target_dir}\n")
    print("  Bohr discovering files...")
    
    # Discover files
    files = await _discover_files(bohr, target_dir)
    tokens_used += TOKEN_PER_CALL * 2
    
    if not files:
        print(f"\n  ERROR: No files found in {target_dir}")
        return {"success": False, "error": f"No files found in {target_dir}"}
    
    print(f"  Found {len(files)} files.\n")
    
    # Have Bohr propose a project name
    file_list_str = "\n".join(f"    - {f}" for f in files)
    
    bohr_response = await bohr.arespond(f"""I need to start a new analysis project. The user asked:

"{research_question}"

I've discovered these files:
{file_list_str}

Please:
1. Suggest a descriptive project name in CamelCase format (e.g., "PluvictoResponseAnalysis")
2. Briefly describe what kind of data this appears to be

Keep it concise - just the project name suggestion and a 1-2 sentence data description.""")
    tokens_used += TOKEN_PER_CALL
    
    if logger:
        logger.log_agent_response("Bohr", "bohr", bohr_response, TOKEN_PER_CALL)
    
    project_name = _extract_project_name(bohr_response) or "UnnamedProject"
    
    # Show user and get confirmation
    _show_agent("Bohr", bohr_response)
    
    print("  Files discovered:")
    for f in files[:20]:  # Show first 20
        print(f"    - {f}")
    if len(files) > 20:
        print(f"    ... and {len(files) - 20} more files")
    print(f"\n  Suggested project name: {project_name}\n")
    
    # Stage 0 user checkpoint
    user_input = _get_user_input(
        "Is this correct? (yes to proceed, 'rename to X', or describe issues)"
    )
    
    while not _user_approves(user_input):
        # Check for rename request
        rename_match = re.search(r'rename(?:\s+to)?\s+["\']?(\w+)["\']?', user_input, re.IGNORECASE)
        if rename_match:
            project_name = rename_match.group(1)
            print(f"\n  Project renamed to: {project_name}")
            user_input = _get_user_input("Continue with this name?")
            continue
        
        # User has concerns - have Bohr address
        bohr_response = await bohr.arespond(f"""The user has feedback about my file discovery and project naming.

User feedback: "{user_input}"

Files found: {file_list_str[:2000]}
Current project name: {project_name}

Please address their concerns. Be specific about what I should look for or change.""")
        tokens_used += TOKEN_PER_CALL
        
        _show_agent("Bohr", bohr_response)
        
        # Try to re-extract project name
        new_name = _extract_project_name(bohr_response)
        if new_name:
            project_name = new_name
        
        print(f"  Current project name: {project_name}\n")
        user_input = _get_user_input("Is this correct now?")
    
    print(f"\n  ✓ Confirmed: Project '{project_name}' with {len(files)} files.\n")
    
    # =========================================================================
    # STAGE 1: BUILD PROJECT STRUCTURE AND SUMMARIZE INPUTS
    # =========================================================================
    _print_stage("STAGE 1", "Build Project Structure and Summarize Inputs")
    if logger:
        logger.log_stage_transition("Stage 1", "Project setup and data summary")
    
    # Create project directory structure
    project_path = Path.cwd() / "Sandbox" / project_name
    project_path.mkdir(parents=True, exist_ok=True)
    scratch_dir = project_path / "scratch"
    scripts_dir = project_path / "scripts"
    outputs_dir = project_path / "outputs"  # For final analysis outputs
    scratch_dir.mkdir(exist_ok=True)
    scripts_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    
    print(f"  Created project structure:")
    print(f"    Sandbox/{project_name}/")
    print(f"    Sandbox/{project_name}/scratch/")
    print(f"    Sandbox/{project_name}/scripts/")
    print(f"    Sandbox/{project_name}/outputs/")
    print(f"\n  Primary outputs to generate:")
    print(f"    - {project_name}_figures.pdf")
    print(f"    - {project_name}_legends.md")
    print(f"    - {project_name}_summary.md\n")
    
    # 1A: Bohr reads and summarizes data files
    _print_substage("1A: Reading and Summarizing Data Files")
    
    print("  Bohr analyzing data files...")
    
    # Read samples from each file
    file_summaries = []
    for f in files[:10]:  # Limit to first 10 files for speed
        content = await _agent_reads_file(bohr, f, lines=30)
        file_summaries.append(f"=== {f} ===\n{content[:1500]}")
    
    # Have Bohr create the manifest
    manifest_prompt = f"""I'm creating a data manifest for project {project_name}.

Here are samples from the data files:

{chr(10).join(file_summaries)}

Please create a data manifest that includes:
1. Overall description: "These files contain..."
2. Sample/patient ID format and naming convention
3. Total unique samples/patients (estimate if needed)
4. Types of features in the data with brief descriptions
5. Any potential issues or ambiguities you notice

Format this as a clear, structured text document."""

    manifest_response = await bohr.arespond(manifest_prompt, max_tokens=4000)
    tokens_used += TOKEN_PER_CALL
    
    # Save manifest
    manifest_path = scratch_dir / "data_manifest.txt"
    manifest = f"""# Data Manifest for {project_name}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Files: {len(files)}

## File Paths:
{chr(10).join(files)}

## Data Summary:
{manifest_response}
"""
    await _agent_writes_file(bohr, f"Sandbox/{project_name}/scratch/data_manifest.txt", manifest)
    
    _show_agent("Bohr", manifest_response, truncate=2000)
    
    # Stage 1 user checkpoint
    print("\n  Summary complete. Bohr has questions or observations about the data.\n")
    user_input = _get_user_input(
        "Review the data interpretation above. Any corrections or clarifications needed? (yes to proceed)"
    )
    
    while not _user_approves(user_input):
        # User has corrections
        bohr_response = await bohr.arespond(f"""The user has feedback about my data interpretation.

User feedback: "{user_input}"

Current manifest:
{manifest_response[:2000]}

Please re-assess and update my understanding based on their feedback.""")
        tokens_used += TOKEN_PER_CALL
        
        _show_agent("Bohr", bohr_response)
        
        # Update manifest
        manifest = manifest.replace(manifest_response, bohr_response)
        manifest_response = bohr_response
        await _agent_writes_file(bohr, f"Sandbox/{project_name}/scratch/data_manifest.txt", manifest)
        
        user_input = _get_user_input("Is this interpretation correct now?")
    
    print("\n  ✓ Data manifest confirmed and saved.\n")
    
    # =========================================================================
    # STAGE 2: PLAN FULL ANALYSIS
    # =========================================================================
    _print_stage("STAGE 2", "Plan Full Analysis")
    if logger:
        logger.log_stage_transition("Stage 2", "Analysis planning")
    
    # Track if we need exploration
    needs_exploration = False
    
    # -------------------------------------------------------------------------
    # STAGE 2A: Initial Planning - Synthesis Core
    # (Bohr → Gould → Farber → Bohr)
    # -------------------------------------------------------------------------
    _print_substage("2A: Synthesis Core (Bohr → Gould → Farber → Bohr)")
    
    synthesis_iterations = 0
    max_synthesis_iterations = 3
    user_satisfied = False
    
    while not user_satisfied and synthesis_iterations < max_synthesis_iterations:
        synthesis_iterations += 1
        print(f"\n  Synthesis Core iteration {synthesis_iterations}...")
        
        # Bohr → Gould: Send problem and manifest
        print("  Bohr briefing Gould on the project...")
        bohr_to_gould = await bohr.arespond(f"""I'm briefing Gould (librarian/science writer) on our project.

PROJECT: {project_name}
RESEARCH QUESTION: {research_question}

DATA MANIFEST:
{manifest_response[:3000]}

Please provide Gould with:
1. The core research question and hypotheses
2. Key data elements to consider
3. Initial directions for literature review

Be concise but comprehensive.""", max_tokens=2000)
        tokens_used += TOKEN_PER_CALL
        
        # Gould: Literature review and citations
        print("  Gould conducting literature review...")
        gould_response = await gould.arespond(f"""Bohr has briefed me on this project:

{bohr_to_gould}

Please:
1. Conduct a comprehensive literature review for this research area
2. Assemble a preliminary CITATIONS list with at least 5 key sources
3. Summarize typical hypotheses, analyses, and questions in this domain
4. Note any methodological considerations from the literature

Format with clear sections for CITATIONS and SUMMARY.""", max_tokens=4000)
        tokens_used += TOKEN_PER_CALL
        
        _show_agent("Gould", gould_response, truncate=1500)
        
        # Extract citations from Gould's response
        # Store for later use in Stage 5
        citations_text = ""
        if "CITATIONS" in gould_response.upper():
            # Try to extract citations section
            cit_match = re.search(r'(?:CITATIONS|REFERENCES)[:\s]*\n(.*?)(?:\n\n[A-Z]|\Z)', 
                                 gould_response, re.DOTALL | re.IGNORECASE)
            if cit_match:
                citations_text = cit_match.group(1).strip()
        if not citations_text:
            citations_text = gould_response  # Fallback to full response
        
        # Gould → Farber: Send plan and citations
        print("  Farber reviewing plan and citations...")
        farber_response = await farber.arespond(f"""Gould has assembled the literature review for {project_name}:

{gould_response[:3000]}

As the critical reviewer, please:
1. Evaluate the merit and feasibility of the emerging plan
2. Check if the citations are appropriate and sufficient
3. Identify any gaps, concerns, or suggestions
4. Provide constructive criticism

Be rigorous but constructive.""", max_tokens=3000)
        tokens_used += TOKEN_PER_CALL
        
        _show_agent("Farber", farber_response, truncate=1500)
        
        # Farber → Bohr: Synthesize into WORKINGPLAN
        print("  Bohr synthesizing input into WORKINGPLAN...")
        
        prev_plan_context = ""
        if working_plan:
            prev_plan_context = f"\nPREVIOUS WORKING PLAN (v{working_plan_version}):\n{working_plan[:2000]}\n"
        
        bohr_synthesis = await bohr.arespond(f"""I'm synthesizing input from Gould and Farber into a WORKINGPLAN.

GOULD'S LITERATURE REVIEW:
{gould_response[:2500]}

FARBER'S CRITIQUE:
{farber_response[:2000]}
{prev_plan_context}
Please create a comprehensive WORKINGPLAN that:
1. States the clear research question and hypotheses
2. Outlines the analysis approach based on literature
3. Addresses Farber's concerns
4. Lists specific analyses to perform
5. Notes any exploratory analyses needed first
6. Describes the expected figure panels (4-6 for final figure)

Mark any sections that need EXPLORATORY ANALYSIS before we can proceed to final analysis.""", max_tokens=4000)
        tokens_used += TOKEN_PER_CALL
        
        working_plan = bohr_synthesis
        working_plan_version += 1
        
        # Save versioned working plan
        plan_path = _get_version_path(scratch_dir, "WORKINGPLAN.md", working_plan_version)
        await _agent_writes_file(bohr, str(plan_path.relative_to(Path.cwd())), working_plan)
        
        _show_agent("Bohr", bohr_synthesis, truncate=2000)
        
        # Check for critical questions
        has_critical_questions = "?" in bohr_synthesis and ("critical" in bohr_synthesis.lower() or "question" in bohr_synthesis.lower())
        
        # Brief summary for user
        print(f"\n  WORKINGPLAN v{working_plan_version} created.\n")
        
        user_input = _get_user_input(
            "Review the working plan above. Approve to continue, or provide feedback/answer questions:"
        )
        
        if _user_approves(user_input):
            user_satisfied = True
        else:
            # User has feedback - incorporate it
            print("  Incorporating user feedback...")
            working_plan = f"{working_plan}\n\n## USER FEEDBACK (iteration {synthesis_iterations}):\n{user_input}"
    
    # Check if exploratory analysis is needed
    if "exploratory" in working_plan.lower() or "explore" in working_plan.lower():
        needs_exploration = True
        print("  Note: Exploratory analyses identified in the plan.\n")
    
    # -------------------------------------------------------------------------
    # STAGE 2B: Theory Core Refinement
    # (Bohr → [Feynman, Shannon, Greider parallel] → Bohr)
    # -------------------------------------------------------------------------
    _print_substage("2B: Theory Core (Bohr → Feynman/Shannon/Greider → Bohr)")
    
    print("  Consulting theorists in parallel...")
    
    theory_prompt = f"""Bohr is sharing the WORKINGPLAN for {project_name} with you:

{working_plan[:4000]}

As a theorist, please:
1. Evaluate the plan from your expertise perspective
2. Suggest ways to analyze the data that may elucidate deeper insights
3. Identify relevant mechanisms or theoretical considerations
4. Note any methodological improvements

Be concise but insightful."""

    # Parallel consultation with all three theorists
    feynman_task = feynman.arespond(theory_prompt, max_tokens=2500)
    shannon_task = shannon.arespond(theory_prompt, max_tokens=2500)
    greider_task = greider.arespond(theory_prompt, max_tokens=2500)
    
    feynman_response, shannon_response, greider_response = await asyncio.gather(
        feynman_task, shannon_task, greider_task
    )
    tokens_used += TOKEN_PER_CALL * 3
    
    print("\n  Theorist input received:")
    _show_agent("Feynman", feynman_response, truncate=800)
    _show_agent("Shannon", shannon_response, truncate=800)
    _show_agent("Greider", greider_response, truncate=800)
    
    # Bohr synthesizes theorist input
    print("  Bohr synthesizing theorist suggestions...")
    
    bohr_theory_synthesis = await bohr.arespond(f"""I'm incorporating theorist feedback into the WORKINGPLAN.

FEYNMAN (physics/creative):
{feynman_response[:1500]}

SHANNON (information theory/causality):
{shannon_response[:1500]}

GREIDER (molecular biology/mechanisms):
{greider_response[:1500]}

CURRENT WORKINGPLAN:
{working_plan[:3000]}

Please update the WORKINGPLAN to incorporate valuable suggestions while maintaining feasibility.
The plan should now include:
1. Any exploratory analyses needed first (if applicable)
2. Clear path to the 3 PRIMARY OUTPUTS:
   - {project_name}_figures.pdf (4-6 panels)
   - {project_name}_legends.md
   - {project_name}_summary.md
3. Specific tests, visualizations, and expected results
4. Citations to include""", max_tokens=5000)
    tokens_used += TOKEN_PER_CALL
    
    working_plan = bohr_theory_synthesis
    working_plan_version += 1
    
    # Save updated plan
    plan_path = _get_version_path(scratch_dir, "WORKINGPLAN.md", working_plan_version)
    await _agent_writes_file(bohr, str(plan_path.relative_to(Path.cwd())), working_plan)
    
    _show_agent("Bohr", bohr_theory_synthesis, truncate=1500)
    print(f"\n  WORKINGPLAN v{working_plan_version} saved.\n")
    
    # -------------------------------------------------------------------------
    # STAGE 2C: Implementation Strategy - Implementation Core
    # (Bohr → Dayhoff → EXECUTIONPLAN)
    # -------------------------------------------------------------------------
    _print_substage("2C: Implementation Core (Bohr → Dayhoff)")
    
    print("  Dayhoff creating execution plan...")
    
    # Determine if exploratory or complete
    if needs_exploration:
        execution_plan_type = "exploratory"
        dayhoff_prompt = f"""Bohr is sharing the WORKINGPLAN for {project_name}:

{working_plan}

The plan identifies EXPLORATORY ANALYSES that need to be done first.

Please create an EXECUTIONPLAN-EXPLORATORY that:
1. Lists specific exploratory scripts to create (in order)
2. Describes what each script should do (data loading, cleaning, PCAs, distributions, etc.)
3. Specifies what outputs each script should produce (in scratch/)
4. Notes what questions each exploration should answer

This plan is for Hinton to implement. Be explicit about:
- File paths (read from ReadData/, write to Sandbox/{project_name}/scratch/)
- Expected outputs (CSVs, PNGs, statistics)
- What we need to learn before proceeding to complete analysis

Format as a clear, numbered plan Hinton can follow."""
    else:
        execution_plan_type = "complete"
        dayhoff_prompt = f"""Bohr is sharing the WORKINGPLAN for {project_name}:

{working_plan}

Please create an EXECUTIONPLAN-COMPLETE that:
1. Lists ALL scripts needed to produce the FIGURE OUTPUT:
   - {project_name}_figures.pdf (4-6 Nature-style panels, 8.5x11", placed in Sandbox/{project_name}/)
2. Describes what each script should do in detail
3. Specifies exact outputs and file paths
4. Includes a final assembly script that:
   - Combines individual panel PNGs into a single multi-panel figure
   - Saves as {project_name}_figures.pdf in the project root (not outputs/)
   - Uses matplotlib's PdfPages or similar for PDF generation

NOTE: The legends.md and summary.md will be written by Gould in Stage 5 based on the figures.

This plan is for Hinton to implement. Be explicit about:
- Script order and dependencies  
- File paths (read from ReadData/, write analysis outputs to Sandbox/{project_name}/outputs/)
- The final PDF goes to Sandbox/{project_name}/{project_name}_figures.pdf
- Statistical tests and visualizations
- Figure panel specifications (Nature style: clean, no gridlines, proper fonts)

Format as a clear, numbered plan Hinton can follow."""
    
    dayhoff_response = await dayhoff.arespond(dayhoff_prompt, max_tokens=5000)
    tokens_used += TOKEN_PER_CALL
    
    execution_plan = dayhoff_response
    
    # Save execution plan
    ep_filename = f"EXECUTIONPLAN-{execution_plan_type.upper()}.md"
    await _agent_writes_file(dayhoff, f"Sandbox/{project_name}/scratch/{ep_filename}", execution_plan)
    
    _show_agent("Dayhoff", dayhoff_response, truncate=2000)
    print(f"\n  {ep_filename} created.\n")
    
    # =========================================================================
    # STAGE 3 or 4: EXECUTION
    # =========================================================================
    
    if execution_plan_type == "exploratory":
        # STAGE 3: EXPLORATORY EXECUTION
        stage_result = await _execute_stage(
            stage_num=3,
            stage_name="Exploratory Execution",
            execution_plan=execution_plan,
            project_name=project_name,
            project_path=project_path,
            working_plan=working_plan,
            agents=agents,
            logger=logger,
            tokens_used=tokens_used,
            is_exploratory=True,
        )
        tokens_used = stage_result["tokens_used"]
        
        if stage_result.get("needs_iteration"):
            # Exploration incomplete - would loop back to Stage 2C
            print("  Exploratory execution needs iteration...")
            # For now, continue to complete execution
        
        # After exploration, create EXECUTIONPLAN-COMPLETE
        _print_substage("2C (continued): Creating Complete Execution Plan")
        
        exploration_results = stage_result.get("results", "")
        
        dayhoff_complete = await dayhoff.arespond(f"""The exploratory analysis is complete. Here are the results:

{exploration_results[:4000]}

WORKING PLAN:
{working_plan[:3000]}

Now create an EXECUTIONPLAN-COMPLETE that:
1. Incorporates what we learned from exploration
2. Lists ALL scripts for final analysis and PRIMARY OUTPUTS:
   - {project_name}_figures.pdf
   - {project_name}_legends.md  
   - {project_name}_summary.md
3. Specifies exact figure panels based on exploration results

Be explicit and detailed for Hinton.""", max_tokens=5000)
        tokens_used += TOKEN_PER_CALL
        
        execution_plan = dayhoff_complete
        execution_plan_type = "complete"
        
        await _agent_writes_file(dayhoff, f"Sandbox/{project_name}/scratch/EXECUTIONPLAN-COMPLETE.md", execution_plan)
        _show_agent("Dayhoff", dayhoff_complete, truncate=1500)
    
    # STAGE 4: COMPLETE EXECUTION
    stage_result = await _execute_stage(
        stage_num=4,
        stage_name="Complete Execution",
        execution_plan=execution_plan,
        project_name=project_name,
        project_path=project_path,
        working_plan=working_plan,
        agents=agents,
        logger=logger,
        tokens_used=tokens_used,
        is_exploratory=False,
    )
    tokens_used = stage_result["tokens_used"]
    
    # =========================================================================
    # STAGE 5: WRITE-UP
    # =========================================================================
    _print_stage("STAGE 5", "Write-up")
    if logger:
        logger.log_stage_transition("Stage 5", "Write-up")
    
    # 5A: Bohr reviews outputs with vision
    _print_substage("5A: Bohr Reviews Outputs")
    
    figures_pdf = project_path / f"{project_name}_figures.pdf"
    
    if figures_pdf.exists():
        print(f"  Bohr viewing {project_name}_figures.pdf...")
        
        bohr_review = await bohr.arespond_with_vision(
            user_message=f"""I'm reviewing the generated figures for {project_name}.

Please examine each panel and describe:
1. What each panel shows (visualization type, data, key findings)
2. Whether the formatting is appropriate (labels, legends, colors)
3. Any issues or improvements needed
4. How well the panels address the research question

Also assess if this meets Nature-style figure guidelines.""",
            pdf_path=str(figures_pdf),
            max_tokens=4000,
        )
        tokens_used += TOKEN_PER_CALL * 2  # Vision costs more
        
        _show_agent("Bohr", bohr_review, truncate=1500)
    else:
        print(f"  ⚠️ {project_name}_figures.pdf not found!")
        bohr_review = "Figures PDF not yet generated."
    
    # 5B: Gould creates legends and summary
    _print_substage("5B: Gould Creating Legends and Summary")
    
    print("  Gould writing figure legends...")
    
    legends_response = await gould.arespond(f"""I'm creating Nature-style figure legends for {project_name}.

BOHR'S FIGURE REVIEW:
{bohr_review[:2000]}

WORKING PLAN:
{working_plan[:2000]}

Please create {project_name}_legends.md with:
- Legend for each panel (a, b, c, d, etc.)
- Nature-style format with:
  - Brief description of what's shown
  - Statistical tests and p-values where applicable
  - Sample sizes (n=X)
  - Clear explanation of axes/colors/symbols
  
Output the complete legends document in markdown format.""", max_tokens=3000)
    tokens_used += TOKEN_PER_CALL
    
    # Save legends
    await _agent_writes_file(gould, f"Sandbox/{project_name}/{project_name}_legends.md", legends_response)
    _show_agent("Gould", legends_response, truncate=1000)
    
    print("  Gould writing summary document...")
    
    summary_response = await gould.arespond(f"""I'm creating the summary document for {project_name}.

BOHR'S FIGURE REVIEW:
{bohr_review[:2000]}

WORKING PLAN:
{working_plan[:3000]}

CITATIONS from literature review:
{citations_text[:1500]}

Please create {project_name}_summary.md as a miniaturized Nature-style paper:

## INTRODUCTION
Two paragraphs: background/current state of field, and rationale/hypothesis

## DISCUSSION  
Two+ paragraphs (~1 page): explicit references to figure panels, interpretation,
context for results, concluding synthesis paragraph

## METHODS
Detailed explanation of all analyses, references to scripts used

## REFERENCES
Complete bibliography with clickable DOI links
Every citation must be referenced in the text above

Ensure all claims are supported by data or citations.""", max_tokens=5000)
    tokens_used += TOKEN_PER_CALL
    
    # Save summary
    await _agent_writes_file(gould, f"Sandbox/{project_name}/{project_name}_summary.md", summary_response)
    _show_agent("Gould", summary_response, truncate=1500)
    
    print(f"\n  ✓ Write-up complete:")
    print(f"    - {project_name}_legends.md")
    print(f"    - {project_name}_summary.md\n")
    
    # =========================================================================
    # STAGE 6: CRITICAL REVIEW
    # =========================================================================
    _print_stage("STAGE 6", "Critical Review")
    if logger:
        logger.log_stage_transition("Stage 6", "Critical review")
    
    print("  Farber conducting critical review...")
    
    # Farber reviews all three outputs
    farber_review = await farber.arespond_with_vision(
        user_message=f"""I'm conducting a critical review of all outputs for {project_name}.

LEGENDS:
{legends_response[:2000]}

SUMMARY:
{summary_response[:3000]}

Please critically evaluate:
1. VALIDITY: Are the sources real and properly cited? Are conclusions supported?
2. ACCURACY: Do figure descriptions match what's shown? Are statistics correct?
3. METHODS: Are the methods appropriate and well-described?
4. FORMATTING: Does everything follow Nature-style guidelines?
5. COHERENCE: Do figures, legends, and summary align?

Be thorough and critical. Note any issues that must be fixed.""",
        pdf_path=str(figures_pdf) if figures_pdf.exists() else None,
        max_tokens=4000,
    )
    tokens_used += TOKEN_PER_CALL * 2
    
    _show_agent("Farber", farber_review, truncate=2000)
    
    # Determine if work is acceptable
    is_acceptable = not any(word in farber_review.lower() 
                           for word in ['major issue', 'unacceptable', 'must fix', 'critical error', 'reject'])
    
    if is_acceptable:
        print("\n  ✓ Farber approves the work.\n")
    else:
        print("\n  ⚠️ Farber identified issues requiring attention.\n")
    
    # Final user checkpoint
    print("=" * 70)
    print("  WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\n  Primary outputs in Sandbox/{project_name}/:")
    print(f"    - {project_name}_figures.pdf")
    print(f"    - {project_name}_legends.md")
    print(f"    - {project_name}_summary.md")
    print(f"\n  Total tokens used: ~{tokens_used:,}\n")
    
    user_input = _get_user_input(
        "Review the outputs. Approve to finish, or provide feedback for iteration:"
    )
    
    if not _user_approves(user_input):
        print("\n  User requested iteration. Would restart from Stage 2 with feedback.")
        # In full implementation, this would loop back to Stage 2 with accumulated feedback
    
    return {
        "success": True,
        "project_name": project_name,
        "project_path": str(project_path),
        "tokens_used": tokens_used,
        "outputs": {
            "figures": str(project_path / f"{project_name}_figures.pdf"),
            "legends": str(project_path / f"{project_name}_legends.md"),
            "summary": str(project_path / f"{project_name}_summary.md"),
        },
        "working_plan_version": working_plan_version,
    }


# =============================================================================
# EXECUTION HELPER (Stages 3 & 4)
# =============================================================================

async def _execute_stage(
    stage_num: int,
    stage_name: str,
    execution_plan: str,
    project_name: str,
    project_path: Path,
    working_plan: str,
    agents: Dict[str, Agent],
    logger: Optional[TranscriptLogger],
    tokens_used: int,
    is_exploratory: bool,
) -> Dict[str, Any]:
    """
    Execute analysis scripts (shared logic for Stages 3 and 4).
    
    Flow: Dayhoff → Hinton → Bayes → Dayhoff (can iterate)
    """
    _print_stage(f"STAGE {stage_num}", stage_name)
    if logger:
        logger.log_stage_transition(f"Stage {stage_num}", stage_name)
    
    hinton = agents["hinton"]
    bayes = agents["bayes"]
    dayhoff = agents["dayhoff"]
    
    scripts_dir = project_path / "scripts"
    scratch_dir = project_path / "scratch"
    
    max_iterations = 3
    iteration = 0
    execution_successful = False
    results_summary = ""
    
    while not execution_successful and iteration < max_iterations:
        iteration += 1
        print(f"\n  Execution iteration {iteration}...")
        
        # Hinton generates/updates scripts
        _print_substage(f"Hinton Generating Scripts (iteration {iteration})")
        
        existing_scripts = list(scripts_dir.glob("*.py"))
        existing_context = ""
        if existing_scripts and iteration > 1:
            existing_context = f"\nEXISTING SCRIPTS (update these):\n"
            for s in existing_scripts[:5]:
                existing_context += f"- {s.name}\n"
        
        hinton_prompt = f"""EXECUTION PLAN for {project_name}:

{execution_plan[:6000]}

{existing_context}

Please generate the required Python scripts. For each script:
1. Save scripts to Sandbox/{project_name}/scripts/
2. Read data from ReadData/
3. Write intermediate outputs to Sandbox/{project_name}/{'scratch' if is_exploratory else 'outputs'}/
4. Use micromamba minilab environment (pandas, numpy, matplotlib, seaborn, scipy, sklearn, lifelines)
5. Include proper error handling and progress prints
6. Set random seed: np.random.seed(42)
{"" if is_exploratory else f'''
7. The FINAL figures PDF must be saved to: Sandbox/{project_name}/{project_name}_figures.pdf
8. Use matplotlib.backends.backend_pdf.PdfPages or PIL to combine panels into a single PDF
9. Format for Nature: clean white backgrounds, no gridlines, 10-12pt fonts, proper axis labels'''}

{"Focus on exploratory analyses only." if is_exploratory else "Generate all scripts including final figure assembly."}

For each script, provide the complete code in ```python``` blocks."""
        
        hinton_response = await hinton.arespond(hinton_prompt, max_tokens=SCRIPT_TOKEN_LIMIT)
        tokens_used += TOKEN_PER_CALL
        
        # Extract and save scripts
        script_blocks = re.findall(r'```python\s*\n(.*?)```', hinton_response, re.DOTALL)
        script_names = re.findall(r'(?:script|file)[:\s]+[`"\']?(\w+\.py)[`"\']?', hinton_response, re.IGNORECASE)
        
        scripts_created = []
        for i, code in enumerate(script_blocks):
            name = script_names[i] if i < len(script_names) else f"script_{i+1:02d}.py"
            script_path = scripts_dir / name
            
            # Validate syntax
            is_valid, error = validate_python_syntax(code)
            if not is_valid:
                print(f"    ⚠️ {name} has syntax error: {error}")
                # Try to fix
                fix_response = await hinton.arespond(f"""Fix this code that produced an error:

Code:
```python
{code}
```

Error: {error}

Provide only the fixed code in a ```python``` block with no explanation.""", max_tokens=FIX_TOKEN_LIMIT)
                tokens_used += TOKEN_PER_CALL
                
                fixed_code = extract_code_from_response(fix_response)
                if fixed_code:
                    is_valid, _ = validate_python_syntax(fixed_code)
                    if is_valid:
                        code = fixed_code
                        print(f"    ✓ {name} fixed")
            
            with open(script_path, 'w') as f:
                f.write(code)
            scripts_created.append(name)
            print(f"    Created: {name}")
        
        if not scripts_created:
            print("    ⚠️ No scripts were created!")
            continue
        
        # Run scripts
        _print_substage("Running Scripts")
        
        # Get workspace root for proper path resolution
        workspace_root = Path.cwd()
        
        execution_results = {}
        for script_name in sorted(scripts_created):
            script_path = scripts_dir / script_name
            print(f"    Running {script_name}...")
            
            try:
                # Run from workspace root so ReadData/ and Sandbox/ paths resolve correctly
                result = subprocess.run(
                    ["micromamba", "run", "-n", "minilab", "python", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(workspace_root),
                )
                
                if result.returncode == 0:
                    print(f"      ✓ Success")
                    execution_results[script_name] = {"success": True, "output": result.stdout[-500:]}
                else:
                    print(f"      ✗ Failed: {result.stderr[:200]}")
                    execution_results[script_name] = {"success": False, "error": result.stderr}
                    
                    # Try to fix
                    with open(script_path) as f:
                        code = f.read()
                    
                    fix_response = await hinton.arespond(f"""Fix this code that produced an error:

Code:
```python
{code}
```

Error:
{result.stderr[:2000]}

Provide only the fixed code in a ```python``` block with no explanation.""", max_tokens=FIX_TOKEN_LIMIT)
                    tokens_used += TOKEN_PER_CALL
                    
                    fixed_code = extract_code_from_response(fix_response)
                    if fixed_code:
                        with open(script_path, 'w') as f:
                            f.write(fixed_code)
                        
                        # Retry - run from workspace root
                        result2 = subprocess.run(
                            ["micromamba", "run", "-n", "minilab", "python", str(script_path)],
                            capture_output=True,
                            text=True,
                            timeout=300,
                            cwd=str(workspace_root),
                        )
                        
                        if result2.returncode == 0:
                            print(f"      ✓ Fixed and succeeded")
                            execution_results[script_name] = {"success": True, "output": result2.stdout[-500:]}
                        
            except subprocess.TimeoutExpired:
                print(f"      ✗ Timed out")
                execution_results[script_name] = {"success": False, "error": "Timeout after 5 minutes"}
        
        # Count successes
        successes = sum(1 for r in execution_results.values() if r.get("success"))
        print(f"\n    Results: {successes}/{len(execution_results)} scripts succeeded")
        
        # Bayes code review
        _print_substage("Bayes Code Review")
        
        outputs_dir = scratch_dir if is_exploratory else (project_path / "outputs")
        output_files = list(outputs_dir.glob("*")) if outputs_dir.exists() else []
        
        bayes_review = await bayes.arespond_with_vision(
            user_message=f"""I'm reviewing the execution results for {project_name}.

EXECUTION RESULTS:
{str(execution_results)[:2000]}

OUTPUT FILES CREATED:
{[f.name for f in output_files[:20]]}

EXECUTION PLAN:
{execution_plan[:2000]}

Please:
1. Review if the scripts achieved the goals
2. Check statistical validity of any results
3. {"Assess if exploration answered the key questions" if is_exploratory else "Verify the figure panels are correct"}
4. Note any issues or improvements needed

Be thorough but constructive.""",
            image_paths=[str(f) for f in output_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']][:5],
            max_tokens=3000,
        )
        tokens_used += TOKEN_PER_CALL * 2
        
        _show_agent("Bayes", bayes_review, truncate=1500)
        
        # Dayhoff assesses results
        dayhoff_assess = await dayhoff.arespond(f"""Bayes has reviewed the execution:

{bayes_review[:2000]}

EXECUTION RESULTS:
{str(execution_results)[:1500]}

Please assess:
1. Did the {"exploration" if is_exploratory else "analysis"} successfully address the plan?
2. Are there critical issues that need fixing?
3. {"What did we learn that should inform the complete analysis?" if is_exploratory else "Are the PRIMARY OUTPUTS ready?"}

Respond with either:
- "SUCCESSFUL: [summary of results]" if we can proceed
- "NEEDS ITERATION: [what to fix]" if we need another attempt""", max_tokens=2000)
        tokens_used += TOKEN_PER_CALL
        
        _show_agent("Dayhoff", dayhoff_assess, truncate=1000)
        
        if "SUCCESSFUL" in dayhoff_assess.upper():
            execution_successful = True
            results_summary = dayhoff_assess
        else:
            # Update execution plan for next iteration
            execution_plan = f"{execution_plan}\n\n## ITERATION {iteration} FEEDBACK:\n{dayhoff_assess}"
    
    return {
        "success": execution_successful,
        "tokens_used": tokens_used,
        "results": results_summary,
        "needs_iteration": not execution_successful,
        "iterations": iteration,
    }
