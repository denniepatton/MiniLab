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
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

from MiniLab.agents.base import Agent
from MiniLab.storage.transcript import TranscriptLogger
from MiniLab.tools.documentation import (
    get_documentation_for_code,
    get_package_constraint_prompt,
    AVAILABLE_PACKAGES,
)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

TOKEN_PER_CALL = 2000  # Rough estimate for token tracking
SCRIPT_TOKEN_LIMIT = 24000  # Max tokens for script generation
FIX_TOKEN_LIMIT = 24000  # Max tokens for code fixes

# Context window management (inspired by CellVoyager)
# Explicit limits prevent token overflow and truncated responses
MAX_ERROR_CHARS = 2000       # ~500 tokens for error messages
MAX_CODE_CONTEXT = 4000      # ~1000 tokens for past code context
MAX_DOC_CHARS = 3000         # ~750 tokens for API documentation
MAX_OUTPUT_CHARS = 2000      # ~500 tokens for execution output

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


def dry_run_validation(code: str, workspace_root: Path) -> Tuple[bool, str]:
    """
    Perform a dry-run validation of Python code.
    
    This checks:
    1. Syntax is valid
    2. All imports succeed
    3. Data file paths that are referenced exist
    
    Inspired by CellVoyager's pre-execution validation approach.
    
    Args:
        code: Python source code
        workspace_root: Root directory for path resolution
        
    Returns:
        (is_valid, error_message)
    """
    import ast
    
    # Step 1: Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    
    # Step 2: Extract imports and verify they're available
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split('.')[0])
    
    # Check imports can be resolved
    import importlib.util
    missing_imports = []
    for imp in set(imports):
        if imp in ('__future__',):
            continue
        spec = importlib.util.find_spec(imp)
        if spec is None:
            missing_imports.append(imp)
    
    if missing_imports:
        return False, f"Missing imports: {', '.join(missing_imports)}"
    
    # Step 3: Check for data file paths and verify they exist
    # Look for common patterns like read_csv, read_excel, open(), Path()
    file_patterns = [
        r"pd\.read_csv\(['\"]([^'\"]+)['\"]",
        r"pd\.read_excel\(['\"]([^'\"]+)['\"]",
        r"pd\.read_parquet\(['\"]([^'\"]+)['\"]",
        r"open\(['\"]([^'\"]+)['\"]",
        r"Path\(['\"]([^'\"]+)['\"]",
        r"np\.load\(['\"]([^'\"]+)['\"]",
        r"np\.loadtxt\(['\"]([^'\"]+)['\"]",
    ]
    
    missing_files = []
    for pattern in file_patterns:
        matches = re.findall(pattern, code)
        for file_path in matches:
            # Resolve relative to workspace root
            full_path = workspace_root / file_path
            if not full_path.exists() and not Path(file_path).exists():
                # Only flag if it looks like an input file (not output)
                if "ReadData" in file_path or file_path.endswith(('.csv', '.xlsx', '.parquet', '.npy')):
                    if "Sandbox" not in file_path and "output" not in file_path.lower():
                        missing_files.append(file_path)
    
    if missing_files:
        return False, f"Input files not found: {', '.join(missing_files[:3])}"
    
    return True, ""


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
    citations_text = ""  # Accumulates citation text from Gould's lit review
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

IMPORTANT: My role is BRIEFING only. I do NOT write analysis code or scripts - that is Hinton's job.
I focus on scientific direction, integration, and project coordination.

Be concise but comprehensive.""", max_tokens=2000)
        tokens_used += TOKEN_PER_CALL
        
        # Gould: Literature review and citations
        print("  Gould conducting literature review...")
        gould_response = await gould.arespond(f"""Bohr has briefed me on this project:

{bohr_to_gould}

YOUR ROLE: You are the LIBRARIAN. You use your web_search tool to find literature DIRECTLY.
DO NOT write Python code, code blocks, or fake tool calls. Simply USE your tools.

Please:
1. Use web_search to find relevant papers on this research topic
2. Assemble a preliminary CITATIONS list with at least 5 key sources (with DOIs where possible)
3. Summarize typical hypotheses, analyses, and questions in this domain
4. Note any methodological considerations from the literature

CRITICAL INSTRUCTIONS:
- DO NOT output Python code or code blocks
- DO NOT write fake <function_calls> or tool invocations
- DIRECTLY use your web_search tool to find papers, then summarize what you find
- Your output should be TEXT with citations, not code

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
    
    # -------------------------------------------------------------------------
    # CRITICAL: Verify outputs exist before proceeding
    # -------------------------------------------------------------------------
    _print_substage("5A: Verify Outputs Exist")
    
    figures_pdf = project_path / f"{project_name}_figures.pdf"
    outputs_dir = project_path / "outputs"
    scratch_dir = project_path / "scratch"
    
    # Check for figures PDF
    if not figures_pdf.exists():
        print(f"  ❌ CRITICAL: {project_name}_figures.pdf does NOT exist!")
        print("     Cannot proceed to write-up without figures.")
        # Try to find any PDFs
        all_pdfs = list(project_path.glob("**/*.pdf"))
        if all_pdfs:
            print(f"     Found other PDFs: {[p.name for p in all_pdfs]}")
        return {
            "success": False,
            "error": "Figures PDF not found - cannot proceed to write-up",
            "project_name": project_name,
            "tokens_used": tokens_used,
        }
    
    print(f"  ✓ {project_name}_figures.pdf exists ({figures_pdf.stat().st_size / 1024:.1f} KB)")
    
    # Inventory all outputs
    output_files = list(outputs_dir.glob("*")) if outputs_dir.exists() else []
    scratch_files = list(scratch_dir.glob("*")) if scratch_dir.exists() else []
    png_files = [f for f in (output_files + scratch_files) if f.suffix.lower() == '.png']
    csv_files = [f for f in (output_files + scratch_files) if f.suffix.lower() == '.csv']
    json_files = [f for f in (output_files + scratch_files) if f.suffix.lower() == '.json']
    
    print(f"  Output inventory:")
    print(f"    - PNG files: {len(png_files)}")
    print(f"    - CSV files: {len(csv_files)}")
    print(f"    - JSON files: {len(json_files)}")
    
    # Read any results JSON if it exists
    results_data = ""
    for jf in json_files:
        if "result" in jf.name.lower() or "stats" in jf.name.lower():
            try:
                with open(jf) as f:
                    results_data = f.read()[:2000]
                print(f"    - Found results file: {jf.name}")
                break
            except:
                pass
    
    # -------------------------------------------------------------------------
    # 5B: Bohr reviews figures with vision (only if they exist)
    # -------------------------------------------------------------------------
    _print_substage("5B: Bohr Reviews Actual Figures")
    
    print(f"  Bohr viewing {project_name}_figures.pdf...")
    
    bohr_review = await bohr.arespond_with_vision(
        user_message=f"""I'm reviewing the generated figures for {project_name}.

CRITICAL: Describe ONLY what you can actually see in the figures. Do not invent or assume content.

For each panel you can see, describe:
1. Panel letter (a, b, c, etc.) - if visible
2. What type of visualization it is (scatter, bar, survival curve, heatmap, etc.)
3. What appears to be plotted (axes labels, data points, trends)
4. Any visible statistics (p-values, sample sizes, confidence intervals)
5. Color schemes and legends visible

If you cannot see or determine something, say "not visible" rather than guessing.

Also assess Nature-style compliance:
- Clean backgrounds
- Proper font sizes
- Clear axis labels
- Professional color palette""",
        pdf_path=str(figures_pdf),
        max_tokens=4000,
    )
    tokens_used += TOKEN_PER_CALL * 2
    
    _show_agent("Bohr", bohr_review, truncate=2000)
    
    # Check if Bohr could actually see the figures
    if "not visible" in bohr_review.lower() or "cannot see" in bohr_review.lower() or "unable to view" in bohr_review.lower():
        print("  ⚠️ Bohr may have had difficulty viewing the figures")
    
    # -------------------------------------------------------------------------
    # 5C: Gould creates legends based on ACTUAL figure content
    # -------------------------------------------------------------------------
    _print_substage("5C: Gould Creating Legends (Based on Actual Figures)")
    
    print("  Gould writing figure legends...")
    
    legends_response = await gould.arespond(f"""Create Nature-style figure legends for {project_name}.

BOHR'S DESCRIPTION OF ACTUAL FIGURES:
{bohr_review}

AVAILABLE RESULTS DATA:
{results_data if results_data else "No results JSON available - use only what Bohr describes."}

CRITICAL INSTRUCTIONS:
1. Write legends ONLY for panels that Bohr actually described seeing
2. Do NOT invent panels, statistics, or data that are not mentioned above
3. If Bohr couldn't see something clearly, write "Panel [X]: Description pending clearer view"
4. Every statistic you mention (p-values, sample sizes, effect sizes) MUST come from:
   - Bohr's figure description, OR
   - The results data above
5. If no statistics are visible, do not make them up

FORMAT for each panel:
**Figure 1[letter].** [One-sentence description of what the panel shows.]
[Details about the data, statistical test used, p-value if visible, sample size if known.]
[Description of axes, colors, and symbols if applicable.]

Output ONLY the final legends document in clean markdown format.
Do not include any "I will now..." or thinking text - just the legends.""", max_tokens=3000)
    tokens_used += TOKEN_PER_CALL
    
    # Check for hallucination indicators
    if any(phrase in legends_response.lower() for phrase in ["i will", "let me", "i'll start", "based on the plan"]):
        print("  ⚠️ Warning: Response may contain thinking text, not just legends")
    
    # Save legends
    await _agent_writes_file(gould, f"Sandbox/{project_name}/{project_name}_legends.md", legends_response)
    _show_agent("Gould", legends_response, truncate=1200)
    
    # -------------------------------------------------------------------------
    # 5D: Gould creates summary based on ACTUAL results
    # -------------------------------------------------------------------------
    _print_substage("5D: Gould Creating Summary (Based on Actual Results)")
    
    print("  Gould writing summary document...")
    
    summary_response = await gould.arespond(f"""Create a mini-paper summary for {project_name}.

BOHR'S DESCRIPTION OF ACTUAL FIGURES:
{bohr_review}

AVAILABLE RESULTS DATA:
{results_data if results_data else "No results JSON file found."}

LITERATURE CITATIONS (from earlier review):
{citations_text[:1500] if citations_text else "Use web_search to find relevant citations if needed."}

RESEARCH QUESTION:
{research_question}

CRITICAL INSTRUCTIONS:
1. Write about ONLY what the figures actually show (per Bohr's description)
2. Do NOT invent statistics, p-values, hazard ratios, or AUC values
3. Every claim must have EITHER:
   - A figure panel reference (e.g., "Figure 1b shows...")
   - A citation with DOI
4. If results are unclear, say "The analysis suggests..." not "We found statistically significant..."
5. Do not include "I will now...", "Let me...", or any thinking text

OUTPUT FORMAT (clean markdown, nothing else):

## INTRODUCTION
[Two paragraphs: Background on the field, then rationale and hypothesis for this analysis]

## RESULTS
[Describe what each figure panel actually shows. Reference specific panels (Fig 1a, 1b, etc.)]
[Include only statistics that appear in the figures or results data]

## DISCUSSION
[Interpret the results in context of the literature]
[Compare to published findings, cite relevant papers with DOIs]
[Acknowledge limitations]

## METHODS
[Describe the analysis pipeline, reference script names from Sandbox/{project_name}/scripts/]
[Statistical tests used, software versions, data sources]

## REFERENCES
[Numbered list with DOIs where possible]
[Every reference must be cited in the text above]

Output ONLY the final document. No preamble, no "Here is...", just the markdown.""", max_tokens=6000)
    tokens_used += TOKEN_PER_CALL
    
    # Check for hallucination indicators
    hallucination_phrases = [
        "hr = 0.", "auc = 0.", "p = 0.00", "p < 0.001",  # Made-up stats
        "i will", "let me", "i'll create", "here is",  # Thinking text
        "based on the plan", "as outlined",  # Plan references instead of actual data
    ]
    for phrase in hallucination_phrases:
        if phrase in summary_response.lower():
            print(f"  ⚠️ Warning: Possible hallucination detected ('{phrase}')")
            break
    
    # Save summary
    await _agent_writes_file(gould, f"Sandbox/{project_name}/{project_name}_summary.md", summary_response)
    _show_agent("Gould", summary_response, truncate=1800)
    
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
    
    # Farber reviews all three outputs with specific focus on hallucination
    farber_review = await farber.arespond_with_vision(
        user_message=f"""Conduct a RIGOROUS critical review of all outputs for {project_name}.

I need you to check for HALLUCINATION - agents making up data that doesn't exist.

LEGENDS DOCUMENT:
{legends_response[:2500]}

SUMMARY DOCUMENT:
{summary_response[:3500]}

BOHR'S ACTUAL FIGURE DESCRIPTION:
{bohr_review[:1500]}

CRITICAL REVIEW CHECKLIST:

1. HALLUCINATION CHECK:
   - Are there statistics (p-values, HR, AUC) in legends/summary that Bohr didn't see in figures?
   - Are there panel descriptions for panels that don't exist?
   - Are there supplementary figures referenced that weren't created?
   - Mark each suspect claim with "⚠️ POSSIBLE HALLUCINATION"

2. CITATION CHECK:
   - Does every factual claim have either a figure reference OR a literature citation?
   - Are the DOIs real and correctly formatted?
   - Mark uncited claims with "⚠️ UNCITED CLAIM"

3. CONSISTENCY CHECK:
   - Do the legends match what Bohr described seeing?
   - Does the summary accurately reflect the figure content?
   - Are methods descriptions accurate?

4. QUALITY CHECK:
   - Is the writing professional (no "I will...", "Let me...")?
   - Is it formatted correctly for Nature style?
   - Are all sections complete?

BE HARSH. List every issue you find. This is peer review.""",
        pdf_path=str(figures_pdf) if figures_pdf.exists() else None,
        max_tokens=5000,
    )
    tokens_used += TOKEN_PER_CALL * 2
    
    _show_agent("Farber", farber_review, truncate=2500)
    
    # Count issues
    hallucination_count = farber_review.lower().count("hallucination")
    uncited_count = farber_review.lower().count("uncited")
    
    if hallucination_count > 0 or uncited_count > 0:
        print(f"\n  ⚠️ Issues found:")
        print(f"    - Possible hallucinations: {hallucination_count}")
        print(f"    - Uncited claims: {uncited_count}")
    
    # Determine if work is acceptable
    is_acceptable = not any(word in farber_review.lower() 
                           for word in ['major issue', 'unacceptable', 'must fix', 'critical error', 
                                       'reject', 'hallucination', 'fabricated'])
    
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

MAX_FIX_ATTEMPTS_PER_SCRIPT = 5  # Max attempts to fix a single script
MAX_BUILD_ITERATIONS = 10  # Max iterations for iterative script building


async def _build_script_iteratively(
    hinton: Agent,
    script_spec: str,
    script_name: str,
    project_name: str,
    project_path: Path,
    workspace_root: Path,
    is_exploratory: bool,
    tokens_used: int,
) -> tuple[str, bool, int]:
    """
    Build a script ITERATIVELY like a VS Code agent.
    
    Instead of generating the whole script in one shot (which truncates),
    Hinton builds the script piece by piece:
    1. Write imports and setup
    2. Check syntax
    3. Write each function
    4. Check syntax after each
    5. Write main block
    6. Full validation
    7. Run and see actual output
    8. Fix based on real errors
    9. Iterate until complete and working
    
    Returns: (final_code, success, tokens_used)
    """
    output_dir_name = "scratch" if is_exploratory else "outputs"
    script_path = project_path / "scripts" / script_name
    pkg_constraints = get_package_constraint_prompt()
    
    # Initialize with empty script
    current_code = ""
    build_phase = "imports"  # imports -> functions -> main -> complete
    
    print(f"      🔨 Building script iteratively...")
    
    for iteration in range(1, MAX_BUILD_ITERATIONS + 1):
        print(f"        Iteration {iteration}/{MAX_BUILD_ITERATIONS} - Phase: {build_phase}")
        
        if build_phase == "imports":
            # PHASE 1: Generate imports and initial setup
            prompt = f"""Write the IMPORTS and INITIAL SETUP for this Python script.

TASK: {script_spec[:1500]}

{pkg_constraints}

Write ONLY the imports and any global constants/configuration.
Include:
- All necessary imports
- np.random.seed(42)
- Path definitions for ReadData/ and Sandbox/{project_name}/{output_dir_name}/

Provide ONLY Python code in a ```python``` block. Just imports and setup, NOT the functions or main block yet."""

            response = await hinton.arespond(prompt, max_tokens=1000)
            tokens_used += TOKEN_PER_CALL
            
            code_chunk = extract_code_from_response(response)
            if code_chunk:
                current_code = code_chunk
                
                # Validate syntax
                is_valid, error = validate_python_syntax(current_code)
                if is_valid:
                    print(f"          ✓ Imports valid")
                    build_phase = "functions"
                else:
                    print(f"          ✗ Syntax error in imports: {error}")
                    # Fix it
                    fix_prompt = f"""Fix this syntax error in the imports:

ERROR: {error}

CODE:
```python
{current_code}
```

Provide the corrected imports section only, in a ```python``` block."""
                    fix_response = await hinton.arespond(fix_prompt, max_tokens=1000)
                    tokens_used += TOKEN_PER_CALL
                    fixed = extract_code_from_response(fix_response)
                    if fixed:
                        current_code = fixed
                    # Stay in imports phase to re-validate
            else:
                print(f"          ⚠️ No code generated for imports")
                
        elif build_phase == "functions":
            # PHASE 2: Generate the functions/logic
            prompt = f"""Now write the FUNCTIONS for this script.

TASK: {script_spec[:1500]}

CURRENT CODE (imports already written):
```python
{current_code}
```

Write the FUNCTION DEFINITIONS that implement the main logic.
Include docstrings and print statements for progress tracking.

CRITICAL: Write ALL functions needed. Do NOT abbreviate with "..." or "similar for other cases".
Write complete, working functions.

Provide ONLY the new function definitions in a ```python``` block.
Do NOT repeat the imports - I will append your functions to the existing code."""

            response = await hinton.arespond(prompt, max_tokens=SCRIPT_TOKEN_LIMIT - 500)
            tokens_used += TOKEN_PER_CALL
            
            code_chunk = extract_code_from_response(response)
            if code_chunk:
                # Combine imports + functions
                test_code = current_code + "\n\n" + code_chunk
                
                # Validate syntax
                is_valid, error = validate_python_syntax(test_code)
                if is_valid:
                    print(f"          ✓ Functions valid")
                    current_code = test_code
                    build_phase = "main"
                else:
                    print(f"          ✗ Syntax error in functions: {error[:100]}")
                    # Fix the functions
                    fix_prompt = f"""Fix this syntax error in the functions:

ERROR: {error}

FUNCTIONS CODE:
```python
{code_chunk}
```

Provide the corrected functions in a ```python``` block."""
                    fix_response = await hinton.arespond(fix_prompt, max_tokens=SCRIPT_TOKEN_LIMIT - 500)
                    tokens_used += TOKEN_PER_CALL
                    fixed = extract_code_from_response(fix_response)
                    if fixed:
                        test_code = current_code + "\n\n" + fixed
                        is_valid2, _ = validate_python_syntax(test_code)
                        if is_valid2:
                            current_code = test_code
                            build_phase = "main"
            else:
                print(f"          ⚠️ No code generated for functions")
                
        elif build_phase == "main":
            # PHASE 3: Generate the main block
            prompt = f"""Now write the MAIN BLOCK for this script.

TASK: {script_spec[:1000]}

CURRENT CODE (imports and functions already written):
```python
{current_code}
```

Write the `if __name__ == "__main__":` block that:
1. Calls the functions above
2. Saves outputs to Sandbox/{project_name}/{output_dir_name}/
3. Ends with: print("SCRIPT COMPLETE: [list of output files]")

Provide ONLY the main block in a ```python``` block.
Do NOT repeat imports or functions - I will append your main block."""

            response = await hinton.arespond(prompt, max_tokens=2000)
            tokens_used += TOKEN_PER_CALL
            
            code_chunk = extract_code_from_response(response)
            if code_chunk:
                # Ensure it has the main guard
                if 'if __name__' not in code_chunk:
                    code_chunk = 'if __name__ == "__main__":\n    ' + code_chunk.replace('\n', '\n    ')
                
                # Combine all parts
                full_code = current_code + "\n\n" + code_chunk
                
                # Validate syntax
                is_valid, error = validate_python_syntax(full_code)
                if is_valid:
                    print(f"          ✓ Main block valid")
                    current_code = full_code
                    build_phase = "complete"
                else:
                    print(f"          ✗ Syntax error in main: {error[:100]}")
                    # Fix it
                    fix_prompt = f"""Fix this syntax error in the main block:

ERROR: {error}

MAIN BLOCK:
```python
{code_chunk}
```

Provide the corrected main block in a ```python``` block."""
                    fix_response = await hinton.arespond(fix_prompt, max_tokens=2000)
                    tokens_used += TOKEN_PER_CALL
                    fixed = extract_code_from_response(fix_response)
                    if fixed:
                        if 'if __name__' not in fixed:
                            fixed = 'if __name__ == "__main__":\n    ' + fixed.replace('\n', '\n    ')
                        full_code = current_code + "\n\n" + fixed
                        is_valid2, _ = validate_python_syntax(full_code)
                        if is_valid2:
                            current_code = full_code
                            build_phase = "complete"
            else:
                print(f"          ⚠️ No code generated for main block")
                
        elif build_phase == "complete":
            # PHASE 4: Validate and run
            print(f"        📋 Full validation...")
            
            # Dry-run validation
            dry_ok, dry_error = dry_run_validation(current_code, workspace_root)
            if not dry_ok:
                print(f"          ✗ Dry-run failed: {dry_error[:100]}")
                # Fix the issue
                fix_prompt = f"""Fix this validation error:

ERROR: {dry_error}

{pkg_constraints}

CURRENT SCRIPT:
```python
{current_code[:MAX_CODE_CONTEXT]}
```

Provide the COMPLETE fixed script in a ```python``` block."""
                fix_response = await hinton.arespond(fix_prompt, max_tokens=FIX_TOKEN_LIMIT)
                tokens_used += TOKEN_PER_CALL
                fixed = extract_code_from_response(fix_response)
                if fixed:
                    current_code = fixed
                continue  # Re-validate
            
            print(f"          ✓ Dry-run passed")
            
            # Save and run the script
            with open(script_path, 'w') as f:
                f.write(current_code)
            
            print(f"        🚀 Running script...")
            try:
                result = subprocess.run(
                    ["micromamba", "run", "-n", "minilab", "python", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(workspace_root),
                )
                
                if result.returncode == 0:
                    print(f"          ✓ SUCCESS!")
                    output_preview = result.stdout[-500:] if result.stdout else ""
                    print(f"          Output: {output_preview[:200]}...")
                    return current_code, True, tokens_used
                else:
                    error_msg = result.stderr if result.stderr else result.stdout
                    print(f"          ✗ Runtime error")
                    print(f"          Error: {error_msg[:200]}...")
                    
                    # Get documentation for fix
                    api_docs = get_documentation_for_code(current_code, max_chars=MAX_DOC_CHARS)
                    
                    # Have Hinton fix based on actual error
                    fix_prompt = f"""The script ran but got this error:

ERROR:
{error_msg[:MAX_ERROR_CHARS]}

{pkg_constraints}

{api_docs if api_docs else ""}

CURRENT SCRIPT:
```python
{current_code}
```

Fix the error and provide the COMPLETE corrected script in a ```python``` block."""
                    
                    fix_response = await hinton.arespond(fix_prompt, max_tokens=FIX_TOKEN_LIMIT)
                    tokens_used += TOKEN_PER_CALL
                    fixed = extract_code_from_response(fix_response)
                    if fixed:
                        current_code = fixed
                    # Stay in complete phase to re-run
                    
            except subprocess.TimeoutExpired:
                print(f"          ✗ Timeout (5 min)")
                return current_code, False, tokens_used
    
    # Max iterations reached
    print(f"        ⚠️ Max iterations ({MAX_BUILD_ITERATIONS}) reached")
    with open(script_path, 'w') as f:
        f.write(current_code)
    return current_code, False, tokens_used


async def _generate_single_script(
    hinton: Agent,
    script_spec: str,
    project_name: str,
    project_path: Path,
    is_exploratory: bool,
    tokens_used: int,
) -> tuple[str, int]:
    """
    DEPRECATED: Use _build_script_iteratively instead.
    
    This is kept for compatibility but should not be used.
    Single-shot generation leads to truncated scripts.
    
    Returns: (script_code, tokens_used)
    """
    output_dir_name = "scratch" if is_exploratory else "outputs"
    
    # Get package constraint prompt
    pkg_constraints = get_package_constraint_prompt()
    
    prompt = f"""Generate ONE complete Python script for the following task:

{script_spec}

{pkg_constraints}

REQUIREMENTS:
1. Read data from ReadData/
2. Write outputs to Sandbox/{project_name}/{output_dir_name}/
3. Include proper error handling and print statements for progress
4. Set random seed: np.random.seed(42) at the top
5. Include `if __name__ == "__main__":` block
6. Print "SCRIPT COMPLETE: [output files created]" at the end
7. Use absolute paths or paths relative to workspace root
{"" if is_exploratory else f'''
8. For figure scripts: Format for Nature style - white backgrounds, no gridlines, 10-12pt fonts
9. Final figures PDF goes to: Sandbox/{project_name}/{project_name}_figures.pdf'''}

CRITICAL: Write the COMPLETE script. Do not truncate or abbreviate any code.
Do not use "..." or "# similar code for other cases" - write out ALL code.

Provide ONLY the Python code in a ```python``` block, no explanation."""

    response = await hinton.arespond(prompt, max_tokens=SCRIPT_TOKEN_LIMIT)
    tokens_used += TOKEN_PER_CALL
    
    code = extract_code_from_response(response)
    return code, tokens_used


async def _critique_script(
    bayes: Agent,
    script_code: str,
    script_name: str,
    script_spec: str,
    tokens_used: int,
) -> tuple[str, bool, int]:
    """
    Have Bayes critique a script BEFORE execution (inspired by CellVoyager).
    
    This catches errors pre-execution instead of post-failure.
    
    Returns: (critique_text, needs_revision, tokens_used)
    """
    prompt = f"""Review this Python script for potential issues BEFORE it runs.

SCRIPT NAME: {script_name}

INTENDED PURPOSE:
{script_spec[:1500]}

CODE:
```python
{script_code[:MAX_CODE_CONTEXT]}
```

Check for:
1. IMPORT ERRORS: Are all imports valid? Any misspelled package names?
2. API ERRORS: Are function calls using correct parameters? (e.g., pd.read_csv, plt.savefig)
3. PATH ERRORS: Do file paths look correct? (ReadData/ for input, Sandbox/ for output)
4. LOGIC ERRORS: Any obvious bugs, missing variables, or incorrect operations?
5. STATISTICAL ERRORS: Are statistical tests appropriate for the data types?

Respond in this format:
ISSUES FOUND: [yes/no]
SEVERITY: [none/minor/major]
DETAILS: [list specific issues if any, or "Script looks correct"]
SUGGESTED FIXES: [specific fixes if needed]"""

    response = await bayes.arespond(prompt, max_tokens=2000)
    tokens_used += TOKEN_PER_CALL
    
    # Parse response to determine if revision needed
    needs_revision = (
        "ISSUES FOUND: yes" in response.lower() or 
        "SEVERITY: major" in response.lower()
    )
    
    return response, needs_revision, tokens_used


async def _fix_script(
    hinton: Agent,
    script_code: str,
    error_message: str,
    script_name: str,
    attempt: int,
    tokens_used: int,
) -> tuple[str, int]:
    """
    Have Hinton fix a failing script.
    
    Enhanced with dynamic documentation retrieval (inspired by CellVoyager)
    to ground fixes in actual API signatures.
    
    Returns: (fixed_code, tokens_used)
    """
    # Get dynamic documentation for functions used in the code
    api_docs = get_documentation_for_code(script_code, max_chars=MAX_DOC_CHARS)
    
    # Truncate error message to prevent token overflow
    truncated_error = error_message[:MAX_ERROR_CHARS]
    
    # Get package constraints
    pkg_constraints = get_package_constraint_prompt()
    
    prompt = f"""Fix this Python script that failed to run.

SCRIPT: {script_name}
ATTEMPT: {attempt}/{MAX_FIX_ATTEMPTS_PER_SCRIPT}

ERROR:
{truncated_error}

CURRENT CODE:
```python
{script_code}
```

{pkg_constraints}

{api_docs if api_docs else ""}

REQUIREMENTS:
1. Fix the specific error shown
2. Ensure all imports are at the top and are from allowed packages only
3. Ensure `if __name__ == "__main__":` block exists
4. Print "SCRIPT COMPLETE: [outputs]" at the end
5. Write the COMPLETE fixed script - do not truncate or abbreviate
6. Use the API documentation above to ensure correct function parameters

Provide ONLY the fixed Python code in a ```python``` block, no explanation."""

    response = await hinton.arespond(prompt, max_tokens=FIX_TOKEN_LIMIT)
    tokens_used += TOKEN_PER_CALL
    
    code = extract_code_from_response(response)
    return code, tokens_used


async def _run_script_with_retry(
    hinton: Agent,
    script_path: Path,
    workspace_root: Path,
    tokens_used: int,
) -> tuple[dict, int]:
    """
    Run a script with iterative fixing until it succeeds or max attempts reached.
    
    Includes dry-run validation before actual execution (inspired by CellVoyager).
    
    Returns: (result_dict, tokens_used)
    
    result_dict has:
      - success: bool
      - output: str (if success)
      - error: str (if failure)
      - attempts: int
    """
    script_name = script_path.name
    
    for attempt in range(1, MAX_FIX_ATTEMPTS_PER_SCRIPT + 1):
        print(f"      Attempt {attempt}/{MAX_FIX_ATTEMPTS_PER_SCRIPT}...")
        
        # Read current script code
        with open(script_path) as f:
            script_code = f.read()
        
        # Check for truncation indicators
        if "..." in script_code or "# similar" in script_code.lower() or "# etc" in script_code.lower():
            print(f"        ⚠️ Script appears truncated, requesting complete version...")
            fixed_code, tokens_used = await _fix_script(
                hinton, script_code, 
                "Script is truncated. Please provide the COMPLETE code without abbreviations.",
                script_name, attempt, tokens_used
            )
            if fixed_code and len(fixed_code) > len(script_code):
                with open(script_path, 'w') as f:
                    f.write(fixed_code)
                script_code = fixed_code
        
        # Step 1: Validate syntax
        is_valid, syntax_error = validate_python_syntax(script_code)
        if not is_valid:
            print(f"        ✗ Syntax error: {syntax_error}")
            fixed_code, tokens_used = await _fix_script(
                hinton, script_code, f"Syntax error: {syntax_error}",
                script_name, attempt, tokens_used
            )
            if fixed_code:
                with open(script_path, 'w') as f:
                    f.write(fixed_code)
            continue  # Try running again
        
        # Step 2: Dry-run validation (check imports and file paths)
        dry_run_ok, dry_run_error = dry_run_validation(script_code, workspace_root)
        if not dry_run_ok:
            print(f"        ✗ Dry-run failed: {dry_run_error}")
            fixed_code, tokens_used = await _fix_script(
                hinton, script_code, f"Pre-execution validation failed: {dry_run_error}",
                script_name, attempt, tokens_used
            )
            if fixed_code:
                with open(script_path, 'w') as f:
                    f.write(fixed_code)
            continue  # Try running again
        
        print(f"        ✓ Dry-run passed")
        
        # Step 3: Run the script
        try:
            result = subprocess.run(
                ["micromamba", "run", "-n", "minilab", "python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(workspace_root),
            )
            
            if result.returncode == 0:
                print(f"        ✓ SUCCESS")
                return {
                    "success": True,
                    "output": result.stdout[-MAX_OUTPUT_CHARS:],
                    "attempts": attempt,
                }, tokens_used
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                print(f"        ✗ Runtime error")
                
                if attempt < MAX_FIX_ATTEMPTS_PER_SCRIPT:
                    fixed_code, tokens_used = await _fix_script(
                        hinton, script_code, error_msg,
                        script_name, attempt, tokens_used
                    )
                    if fixed_code:
                        with open(script_path, 'w') as f:
                            f.write(fixed_code)
                else:
                    return {
                        "success": False,
                        "error": error_msg[:2000],
                        "attempts": attempt,
                    }, tokens_used
                    
        except subprocess.TimeoutExpired:
            print(f"        ✗ Timeout (5 min)")
            return {
                "success": False,
                "error": "Script timed out after 5 minutes",
                "attempts": attempt,
            }, tokens_used
    
    # Should not reach here, but just in case
    return {
        "success": False,
        "error": f"Failed after {MAX_FIX_ATTEMPTS_PER_SCRIPT} attempts",
        "attempts": MAX_FIX_ATTEMPTS_PER_SCRIPT,
    }, tokens_used


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
    
    CRITICAL APPROACH:
    - Generate scripts ONE AT A TIME
    - Each script MUST run successfully before proceeding to the next
    - Iterate with fixes until success or max attempts
    - No batch generation that produces truncated code
    
    Flow: Dayhoff → Hinton (per-script iterative) → Bayes → Dayhoff
    """
    _print_stage(f"STAGE {stage_num}", stage_name)
    if logger:
        logger.log_stage_transition(f"Stage {stage_num}", stage_name)
    
    hinton = agents["hinton"]
    bayes = agents["bayes"]
    dayhoff = agents["dayhoff"]
    
    scripts_dir = project_path / "scripts"
    scratch_dir = project_path / "scratch"
    outputs_dir = project_path / "outputs"
    workspace_root = Path.cwd()
    
    # Step 1: Have Dayhoff break down the execution plan into individual script specs
    _print_substage("Dayhoff Breaking Down Execution Plan")
    
    dayhoff_breakdown = await dayhoff.arespond(f"""Break down this execution plan into individual scripts.

EXECUTION PLAN:
{execution_plan[:5000]}

For each script, provide:
1. Script name (e.g., 01_data_loading.py, 02_preprocessing.py, etc.)
2. Purpose (one sentence)
3. Input files (from ReadData/ or previous script outputs)
4. Output files (to Sandbox/{project_name}/{'scratch' if is_exploratory else 'outputs'}/)
5. Key operations (bullet points)

IMPORTANT: Each script should be a COMPLETE, standalone piece that can run independently
(except for dependencies on outputs from previous scripts).

Format as:
### SCRIPT: 01_name.py
Purpose: ...
Inputs: ...
Outputs: ...
Operations:
- ...

List scripts in execution order.""", max_tokens=4000)
    tokens_used += TOKEN_PER_CALL
    
    _show_agent("Dayhoff", dayhoff_breakdown, truncate=1500)
    
    # Parse script specifications
    script_specs = []
    spec_pattern = r'###\s*SCRIPT:\s*(\S+\.py)(.*?)(?=###\s*SCRIPT:|$)'
    matches = re.findall(spec_pattern, dayhoff_breakdown, re.DOTALL | re.IGNORECASE)
    
    if not matches:
        # Fallback: try to find numbered scripts
        lines = dayhoff_breakdown.split('\n')
        current_spec = ""
        current_name = ""
        for line in lines:
            if re.match(r'^\d+[\.\)]\s*\w+\.py', line):
                if current_name and current_spec:
                    script_specs.append((current_name, current_spec))
                name_match = re.search(r'(\w+\.py)', line)
                current_name = name_match.group(1) if name_match else f"script_{len(script_specs)+1:02d}.py"
                current_spec = line
            elif current_name:
                current_spec += "\n" + line
        if current_name and current_spec:
            script_specs.append((current_name, current_spec))
    else:
        script_specs = [(m[0].strip(), m[1].strip()) for m in matches]
    
    if not script_specs:
        print("    ⚠️ Could not parse script specifications. Using single-script fallback.")
        script_specs = [("analysis_pipeline.py", execution_plan)]
    
    print(f"\n    Identified {len(script_specs)} scripts to generate:")
    for name, _ in script_specs:
        print(f"      - {name}")
    
    # Step 2: Build and run each script ITERATIVELY (like VS Code agent)
    _print_substage("Building and Running Scripts (Iterative Approach)")
    
    execution_results = {}
    all_succeeded = True
    
    for idx, (script_name, script_spec) in enumerate(script_specs, 1):
        print(f"\n    [{idx}/{len(script_specs)}] {script_name}")
        print(f"      📝 Building iteratively (not single-shot)...")
        
        # Build the script iteratively - Hinton writes, checks, runs, fixes
        script_code, success, tokens_used = await _build_script_iteratively(
            hinton=hinton,
            script_spec=script_spec,
            script_name=script_name,
            project_name=project_name,
            project_path=project_path,
            workspace_root=workspace_root,
            is_exploratory=is_exploratory,
            tokens_used=tokens_used,
        )
        
        if success:
            print(f"      ✅ {script_name} - COMPLETE AND WORKING")
            execution_results[script_name] = {
                "success": True,
                "output": "Built iteratively and executed successfully",
                "attempts": 1,  # Iterations are internal
            }
        else:
            print(f"      ❌ {script_name} - FAILED after max iterations")
            execution_results[script_name] = {
                "success": False,
                "error": "Failed to build working script after max iterations",
                "attempts": MAX_BUILD_ITERATIONS,
            }
            all_succeeded = False
        
        # EARLY VLM CHECK: If this script produced images, have Bayes check them immediately
        if success:
            output_dir = scratch_dir if is_exploratory else outputs_dir
            new_pngs = [f for f in output_dir.glob("*.png") if script_name.replace('.py', '') in f.stem.lower() or f.stem.startswith(script_name[:2])]
            if new_pngs:
                print(f"      Bayes verifying {len(new_pngs)} output image(s)...")
                img_check = await bayes.arespond_with_vision(
                    user_message=f"""Quick check: Do these output images from {script_name} look correct?

Check for:
1. Are the axes labeled properly?
2. Does the data look reasonable (not all zeros, not obviously wrong)?
3. Are there any visual errors (missing legends, cut-off text)?

Just respond: LOOKS GOOD or ISSUE: [brief description]""",
                    image_paths=[str(f) for f in new_pngs[:3]],
                    max_tokens=500,
                )
                tokens_used += TOKEN_PER_CALL
                if "ISSUE" in img_check.upper():
                    print(f"        ⚠️ Bayes: {img_check[:100]}")
                else:
                    print(f"        ✓ Images verified")
    
    # Summary
    successes = sum(1 for r in execution_results.values() if r.get("success"))
    print(f"\n    Script Results: {successes}/{len(execution_results)} succeeded")
    
    # Step 3: Bayes reviews execution results
    _print_substage("Bayes Code Review")
    
    output_dir = scratch_dir if is_exploratory else outputs_dir
    output_files = list(output_dir.glob("*")) if output_dir.exists() else []
    
    # Check if figures PDF exists (for complete execution)
    figures_pdf = project_path / f"{project_name}_figures.pdf"
    figures_exist = figures_pdf.exists()
    
    bayes_review = await bayes.arespond_with_vision(
        user_message=f"""I'm reviewing the execution results for {project_name}.

EXECUTION SUMMARY:
- Scripts run: {len(execution_results)}
- Succeeded: {successes}
- Failed: {len(execution_results) - successes}

DETAILED RESULTS:
{str({k: {'success': v['success'], 'attempts': v.get('attempts', 1)} for k, v in execution_results.items()})[:1500]}

OUTPUT FILES CREATED ({len(output_files)}):
{[f.name for f in output_files[:20]]}

{"FIGURES PDF EXISTS: " + str(figures_exist) if not is_exploratory else ""}

Please review:
1. Did the scripts achieve the execution plan goals?
2. Are the outputs valid and complete?
3. {"Were exploration questions answered?" if is_exploratory else "Is the figures PDF ready?"}
4. Any statistical or methodological concerns?

Be specific about what succeeded and what needs attention.""",
        image_paths=[str(f) for f in output_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']][:5],
        max_tokens=3000,
    )
    tokens_used += TOKEN_PER_CALL * 2
    
    _show_agent("Bayes", bayes_review, truncate=1500)
    
    # Step 4: Dayhoff assesses overall results
    dayhoff_assess = await dayhoff.arespond(f"""Bayes has reviewed the execution:

{bayes_review[:2000]}

EXECUTION SUMMARY:
- {successes}/{len(execution_results)} scripts succeeded
{"- Figures PDF exists: " + str(figures_exist) if not is_exploratory else ""}

OUTPUT FILES: {[f.name for f in output_files[:15]]}

Please provide final assessment:
1. Did we {"complete the exploration successfully" if is_exploratory else "generate the required outputs"}?
2. Are there critical failures that block progress?
3. {"What did we learn for complete analysis?" if is_exploratory else "Is the figures PDF ready for write-up?"}

CRITICAL: If the figures PDF does not exist for complete execution, this is a FAILURE.

Respond with:
- "SUCCESSFUL: [summary]" if we can proceed to next stage
- "NEEDS ITERATION: [specific issues]" if we need to regenerate scripts""", max_tokens=2000)
    tokens_used += TOKEN_PER_CALL
    
    _show_agent("Dayhoff", dayhoff_assess, truncate=1000)
    
    execution_successful = "SUCCESSFUL" in dayhoff_assess.upper() and all_succeeded
    
    # For complete execution, also require figures PDF to exist
    if not is_exploratory and not figures_exist:
        execution_successful = False
        print("    ⚠️ Figures PDF not found - execution incomplete")
    
    return {
        "success": execution_successful,
        "tokens_used": tokens_used,
        "results": dayhoff_assess,
        "needs_iteration": not execution_successful,
        "script_results": execution_results,
        "figures_exist": figures_exist if not is_exploratory else None,
    }
