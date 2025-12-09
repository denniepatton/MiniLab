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
from typing import Dict, Optional, Any, List

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


async def _agent_writes_file(agent: Agent, path: str, content: str) -> bool:
    """Have agent write content to a file."""
    result = await agent.use_tool("filesystem", action="write", path=path, content=content)
    return result.get("success", False)


async def _agentic_delegate(
    agent: Agent,
    task: str,
    shared_context: str = "",
    max_iterations: int = 10,
    max_tokens_per_step: int = 4000,
    verbose: bool = True,
    logger: Optional[TranscriptLogger] = None,
) -> Dict[str, Any]:
    """
    Delegate a task to an agent using TRUE agentic execution.
    
    Unlike arespond() which just generates text, this allows the agent to:
    - Use their tools (filesystem, code_editor, web_search, etc.)
    - Consult colleagues when they need help
    - Make multiple attempts until the task is done
    
    Args:
        agent: The Agent instance to delegate to
        task: The specific task for this agent to complete
        shared_context: Full project context ALL agents should know about
        max_iterations: Max ReAct iterations
        max_tokens_per_step: Token limit per agent step
        verbose: Whether to print progress
        logger: Optional TranscriptLogger for logging operations
        
    Returns:
        Result dict with 'success', 'result', 'iterations', 'tool_calls', etc.
    """
    # Build full context for the agent
    full_task = f"""SHARED CONTEXT (what all agents know):
{shared_context}

YOUR SPECIFIC TASK:
{task}

REMINDER: 
- You have tools: filesystem, code_editor, terminal, web_search, environment
- You can consult colleagues using ```colleague {{"colleague": "agent_id", "question": "..."}}```
- When done, signal with ```done {{"result": "summary of what you did", "outputs": [...]}}```
- USE your tools - don't just describe what you would do!
"""
    
    result = await agent.agentic_execute(
        task=full_task,
        context=shared_context,
        max_iterations=max_iterations,
        max_tokens_per_step=max_tokens_per_step,
        verbose=verbose,
        logger=logger,
    )
    
    # Log completion summary
    if logger:
        logger.log_system_event(
            "agentic_complete",
            f"{agent.display_name} completed task",
            {
                "iterations": result.get("iterations", 0),
                "tool_calls": len(result.get("tool_calls", [])),
                "success": result.get("success", False),
            }
        )
    
    return result


def _build_shared_context(
    project_name: str,
    research_question: str,
    files: List[str],
    manifest: str = "",
    working_plan: str = "",
    execution_plan: str = "",
    stage: str = "",
) -> str:
    """
    Build a comprehensive shared context string that ALL agents can see.
    
    This ensures every agent has the full picture - no more hallucinating
    because they don't know what exists or what's been done.
    """
    context_parts = [
        f"PROJECT: {project_name}",
        f"RESEARCH QUESTION: {research_question}",
        f"CURRENT STAGE: {stage}" if stage else "",
        f"\nDATA FILES ({len(files)} files):",
        "\n".join(f"  - {f}" for f in files[:20]),
        f"  ... and {len(files) - 20} more" if len(files) > 20 else "",
    ]
    
    if manifest:
        context_parts.append(f"\nDATA MANIFEST:\n{manifest[:2000]}")
    
    if working_plan:
        context_parts.append(f"\nWORKING PLAN:\n{working_plan[:3000]}")
    
    if execution_plan:
        context_parts.append(f"\nEXECUTION PLAN:\n{execution_plan[:2000]}")
    
    context_parts.append(f"\nPROJECT PATHS:")
    context_parts.append(f"  - Read data from: ReadData/")
    context_parts.append(f"  - Write outputs to: Sandbox/{project_name}/")
    context_parts.append(f"  - Scripts go in: Sandbox/{project_name}/scripts/")
    context_parts.append(f"  - Scratch/intermediate: Sandbox/{project_name}/scratch/")
    context_parts.append(f"  - Final outputs: Sandbox/{project_name}/outputs/")
    
    return "\n".join(part for part in context_parts if part)


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
    _print_stage("STAGE 0", "Discover Data and Set Up Project")
    if logger:
        logger.log_stage_transition("Stage 0", "Discovery and setup")
    
    # Extract target directory from research question (minimal parsing, just for context)
    dir_match = re.search(r'ReadData/[\w/]+', research_question)
    target_dir = dir_match.group(0) if dir_match else "ReadData"
    
    print(f"  Bohr discovering data and setting up project (agentic mode)...\n")
    
    # FULLY AGENTIC: Bohr does EVERYTHING - discovery, naming, user approval, directory creation
    stage0_task = f"""STAGE 0: Discover data files and set up the project.

RESEARCH QUESTION FROM USER:
"{research_question}"

YOUR TASK (do these in order):

1. DISCOVER FILES: Use terminal to list files in {target_dir}
   Example: ls -la {target_dir}

2. EXPLORE DATA: Look at the first few lines of key files to understand what this data is
   Example: head -5 {target_dir}/some_file.csv

3. PROPOSE PROJECT: Based on what you find, decide on:
   - A descriptive project name in CamelCase (e.g., PluvictoResponsePrediction)
   - A brief description of what this data represents

4. ASK USER FOR APPROVAL: Use the user_input tool to ask the user if your project name is OK
   Example: {{"tool": "user_input", "action": "ask", "params": {{"prompt": "I propose naming this project 'YourProjectName'. This data appears to be [your description]. Does this work? (yes, or suggest a different name)"}}}}

5. HANDLE USER RESPONSE: 
   - If user says yes/ok/looks good: proceed to create directories
   - If user suggests a different name: use that name instead
   - If user has concerns: address them and ask again

6. CREATE PROJECT STRUCTURE: Once you have approval, create the directories:
   - Sandbox/[ProjectName]/
   - Sandbox/[ProjectName]/scripts/
   - Sandbox/[ProjectName]/outputs/
   - Sandbox/[ProjectName]/scratch/
   Use terminal: mkdir -p Sandbox/[ProjectName]/scripts Sandbox/[ProjectName]/outputs Sandbox/[ProjectName]/scratch

7. SIGNAL COMPLETION with the project name and file list:
```done
{{"result": "Project [ProjectName] created with [N] data files", "project_name": "[ProjectName]", "files": ["list", "of", "discovered", "files"]}}
```

IMPORTANT: 
- YOU decide how to interpret user responses, not the orchestrator
- YOU create the directories, not the orchestrator  
- The project_name in your done signal will be used for subsequent stages
"""

    stage0_result = await _agentic_delegate(
        agent=bohr,
        task=stage0_task,
        shared_context=f"Starting new analysis project based on user's research question.",
        max_iterations=15,
        max_tokens_per_step=3000,
        verbose=True,
        logger=logger,
    )
    tokens_used += TOKEN_PER_CALL * stage0_result.get("iterations", 5)
    
    if not stage0_result.get("success"):
        print(f"\n  ERROR: Stage 0 failed - {stage0_result.get('error', 'Unknown error')}")
        return {"success": False, "error": "Stage 0 failed", "details": stage0_result}
    
    # Extract project info from Bohr's done signal
    result_data = stage0_result.get("result", "")
    
    # Try to parse the done signal for project_name
    import json
    try:
        # Look for JSON in the result
        if isinstance(result_data, dict):
            project_name = result_data.get("project_name", "UnnamedProject")
            files = result_data.get("files", [])
        else:
            # Fallback: extract from text
            project_name = _extract_project_name(str(result_data)) or "UnnamedProject"
            # Get files from tool calls
            files = []
            for tc in stage0_result.get("tool_calls", []):
                if "files" in str(tc.get("result", {})):
                    files = tc.get("result", {}).get("files", files)
    except:
        project_name = "UnnamedProject"
        files = []
    
    # Verify directory was created
    project_path = Path.cwd() / "Sandbox" / project_name
    if not project_path.exists():
        print(f"  ⚠️ Directory not created, creating now...")
        project_path.mkdir(parents=True, exist_ok=True)
        (project_path / "scripts").mkdir(exist_ok=True)
        (project_path / "outputs").mkdir(exist_ok=True)
        (project_path / "scratch").mkdir(exist_ok=True)
    
    print(f"\n  ✓ Stage 0 complete: Project '{project_name}' initialized.\n")
    
    # =========================================================================
    # STAGE 1: BUILD DATA MANIFEST
    # =========================================================================
    _print_stage("STAGE 1", "Build Data Manifest")
    if logger:
        logger.log_stage_transition("Stage 1", "Data manifest")
    
    print(f"  Bohr exploring data and creating manifest (agentic mode)...\n")
    
    # Build shared context
    shared_context = _build_shared_context(
        project_name=project_name,
        research_question=research_question,
        files=files,
        stage="Stage 1: Data Manifest",
    )
    
    # FULLY AGENTIC: Bohr explores data, creates manifest, and gets user approval himself
    stage1_task = f"""STAGE 1: Create a comprehensive data manifest for {project_name}.

RESEARCH QUESTION:
"{research_question}"

YOUR TASK (do these in order):

1. EXPLORE DATA FILES: Use terminal to examine the data files in detail
   - List all files (you may already know from Stage 0)
   - Check dimensions: wc -l, head -1 for column names
   - Sample content: head -5, tail -5
   - File types, sizes, etc.

2. UNDERSTAND THE DATA: Figure out:
   - What does this data represent? (clinical trial, biomarkers, etc.)
   - What is the sample/patient ID format?
   - What are the key features/columns?
   - Any data quality issues?

3. CREATE DATA MANIFEST: Save your findings to Sandbox/{project_name}/scratch/data_manifest.txt
   Use code_editor to create/write the file with:
   - Overview of data sources
   - File-by-file summary (columns, row count, content type)
   - Key variables identified
   - Data quality notes
   - Recommendations for analysis

4. PRESENT TO USER AND GET APPROVAL: Use user_input.ask to show your summary and ask:
   - "I've analyzed the data. Here's what I found: [brief summary]. Is this interpretation correct? (yes, or tell me what I got wrong)"

5. HANDLE USER FEEDBACK:
   - If user says yes/correct/proceed: signal done
   - If user has corrections: update the manifest and ask again
   - YOU interpret the user's response and decide what to do

6. SIGNAL COMPLETION:
```done
{{"result": "Data manifest created with [summary]", "manifest_summary": "[2-3 sentence summary for next stages]", "outputs": ["data_manifest.txt"]}}
```

IMPORTANT:
- YOU ask the user for approval using user_input tool
- YOU interpret their response
- YOU update the manifest if needed
- The orchestrator does NOT parse your responses or user input
"""

    stage1_result = await _agentic_delegate(
        agent=bohr,
        task=stage1_task,
        shared_context=shared_context,
        max_iterations=12,
        max_tokens_per_step=4000,
        verbose=True,
        logger=logger,
    )
    tokens_used += TOKEN_PER_CALL * stage1_result.get("iterations", 3)
    
    if not stage1_result.get("success"):
        print(f"\n  ⚠️ Stage 1 completed with issues - {stage1_result.get('error', 'Unknown')}")
    
    # Set paths for later use (should have been created by Bohr in Stage 0)
    project_path = Path.cwd() / "Sandbox" / project_name
    scratch_dir = project_path / "scratch"
    scripts_dir = project_path / "scripts"
    outputs_dir = project_path / "outputs"
    
    # Ensure directories exist (fallback)
    project_path.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(exist_ok=True)
    scripts_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    
    # Read manifest if it exists
    manifest_path = scratch_dir / "data_manifest.txt"
    if manifest_path.exists():
        manifest_content = manifest_path.read_text().strip()
        # Check if manifest has actual content (not just headers)
        if len(manifest_content) > 100 and "Sample" in manifest_content or "feature" in manifest_content.lower():
            manifest_response = manifest_content[:4000]
            print(f"  ✓ Bohr created data manifest")
        else:
            # Manifest exists but is sparse - use Bohr's result to fill it
            bohr_result = stage1_result.get("result", "")
            if bohr_result and len(str(bohr_result)) > 50:
                manifest_response = str(bohr_result)[:4000]
            else:
                manifest_response = manifest_content[:4000] if manifest_content else "Manifest created but needs content"
            print(f"  ⚠️ Manifest created but sparse, using Bohr's analysis")
    else:
        # No manifest file - create from Bohr's result
        bohr_result = stage1_result.get("result", "")
        manifest_response = str(bohr_result) if bohr_result else "Bohr completed setup but did not create manifest"
        print(f"  ⚠️ Manifest file not found, creating from Bohr's output")
        # Save the manifest content
        manifest = f"""# Data Manifest for {project_name}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Files: {len(files)}

## File Paths:
{chr(10).join(files)}

## Data Summary:
{manifest_response}
"""
        await _agent_writes_file(bohr, f"Sandbox/{project_name}/scratch/data_manifest.txt", manifest)
    
    print(f"\n  Project structure:")
    print(f"    Sandbox/{project_name}/")
    print(f"    Sandbox/{project_name}/scratch/")
    print(f"    Sandbox/{project_name}/scripts/")
    print(f"    Sandbox/{project_name}/outputs/")
    print(f"\n  Primary outputs to generate:")
    print(f"    - {project_name}_figures.pdf")
    print(f"    - {project_name}_legends.md")
    print(f"    - {project_name}_summary.md\n")
    
    # Get manifest content for later stages
    if manifest_path.exists():
        manifest_response = manifest_path.read_text()[:4000]
    else:
        # Try to get from Stage 1 result
        result_data = stage1_result.get("result", {})
        if isinstance(result_data, dict):
            manifest_response = result_data.get("manifest_summary", str(result_data))
        else:
            manifest_response = str(result_data)
    
    print(f"\n  ✓ Stage 1 complete: Data manifest created and approved.\n")
    
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
        if logger:
            logger.log_agent_response(bohr.display_name, bohr.id, bohr_to_gould[:500], TOKEN_PER_CALL)
        
        # Gould: Literature review and citations using TRUE agentic execution
        print("  Gould conducting literature review (agentic mode)...")
        
        # Build shared context for Gould
        shared_context = _build_shared_context(
            project_name=project_name,
            research_question=research_question,
            files=files,
            manifest=manifest_response,
            stage="Stage 2A: Literature Review",
        )
        
        gould_task = f"""Bohr has briefed you on this project:

{bohr_to_gould}

GOAL: Find relevant literature to inform our analysis approach.

Use your web_search tool to find papers about:
- The disease/treatment being studied
- Biomarker discovery methods for similar data
- Statistical approaches for this type of analysis

DELIVERABLES:
1. At least 5 real citations with DOIs (from PubMed)
2. Key methodological insights from the literature
3. Any relevant findings we should consider

Save your literature review to Sandbox/{project_name}/scratch/literature_review.md

When done: ```done {{"result": "Found X relevant papers. Key insights: [brief summary]", "outputs": ["literature_review.md"]}}```
"""
        
        gould_result = await _agentic_delegate(
            agent=gould,
            task=gould_task,
            shared_context=shared_context,
            max_iterations=8,
            max_tokens_per_step=4000,
            verbose=True,
            logger=logger,
        )
        tokens_used += TOKEN_PER_CALL * gould_result.get("iterations", 3)
        
        gould_response = gould_result.get("result", "Literature review failed to complete")
        if not gould_result.get("success"):
            # Fallback - use whatever output we got
            gould_response = gould_result.get("final_output", gould_response)
        
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
        if logger:
            logger.log_agent_response(farber.display_name, farber.id, farber_response[:500], TOKEN_PER_CALL)
        
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
        if logger:
            logger.log_agent_response(bohr.display_name, bohr.id, bohr_synthesis[:500], TOKEN_PER_CALL)
        
        working_plan = bohr_synthesis
        working_plan_version += 1
        
        # Save versioned working plan
        plan_path = _get_version_path(scratch_dir, "WORKINGPLAN.md", working_plan_version)
        await _agent_writes_file(bohr, str(plan_path.relative_to(Path.cwd())), working_plan)
        
        _show_agent("Bohr", bohr_synthesis, truncate=2000)
        
        # Brief summary for user
        print(f"\n  WORKINGPLAN v{working_plan_version} created.\n")
        
        # FULLY AGENTIC: Bohr asks user for approval and handles feedback
        stage2a_approval_task = f"""You've just created WORKINGPLAN v{working_plan_version}. The user has seen it above.

YOUR TASK: Get user approval for the working plan.

1. Use user_input.ask to ask the user:
   {{"tool": "user_input", "action": "ask", "params": {{"prompt": "Do you approve this working plan? (yes to proceed, or tell me what to change)"}}}}

2. If user says yes/approve/proceed/looks good: Signal done with approval
3. If user has feedback/concerns: Note them and signal done with the feedback

```done
{{"result": "approved" or "feedback: [user's feedback]", "user_approved": true or false, "feedback": "[any feedback]"}}
```
"""
        
        approval_result = await _agentic_delegate(
            agent=bohr,
            task=stage2a_approval_task,
            shared_context=f"Working plan v{working_plan_version} just shown to user.",
            max_iterations=5,
            max_tokens_per_step=2000,
            verbose=True,
            logger=logger,
        )
        tokens_used += TOKEN_PER_CALL * approval_result.get("iterations", 2)
        
        # Check if user approved
        result_data = approval_result.get("result", {})
        if isinstance(result_data, dict):
            user_satisfied = result_data.get("user_approved", False)
            user_feedback = result_data.get("feedback", "")
        else:
            # Parse from string
            result_str = str(result_data).lower()
            user_satisfied = "approved" in result_str or "true" in result_str
            user_feedback = str(result_data) if not user_satisfied else ""
        
        if not user_satisfied and user_feedback:
            print(f"  Incorporating user feedback: {user_feedback[:100]}...")
            working_plan = f"{working_plan}\n\n## USER FEEDBACK (iteration {synthesis_iterations}):\n{user_feedback}"
    
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
    if logger:
        logger.log_agent_response(feynman.display_name, feynman.id, feynman_response[:300], TOKEN_PER_CALL)
        logger.log_agent_response(shannon.display_name, shannon.id, shannon_response[:300], TOKEN_PER_CALL)
        logger.log_agent_response(greider.display_name, greider.id, greider_response[:300], TOKEN_PER_CALL)
    
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
    if logger:
        logger.log_agent_response(bohr.display_name, bohr.id, bohr_theory_synthesis[:500], TOKEN_PER_CALL)
    
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
    if logger:
        logger.log_agent_response(dayhoff.display_name, dayhoff.id, dayhoff_response[:500], TOKEN_PER_CALL)
    
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
        if logger:
            logger.log_agent_response(dayhoff.display_name, dayhoff.id, dayhoff_complete[:500], TOKEN_PER_CALL)
        
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
    # 5C: Gould creates legends based on ACTUAL figure content (agentic)
    # -------------------------------------------------------------------------
    _print_substage("5C: Gould Creating Legends (Agentic Mode)")
    
    print("  Gould writing figure legends using tools to view actual outputs...")
    
    # Build shared context for Gould
    shared_context = _build_shared_context(
        project_name=project_name,
        research_question=research_question,
        files=files,
        manifest=manifest if 'manifest' in dir() else "",
        working_plan=working_plan if 'working_plan' in dir() else "",
        stage="Stage 5C: Writing Figure Legends",
    )
    
    gould_legends_task = f"""GOAL: Write Nature-style figure legends for {project_name}.

BOHR SAW IN THE FIGURES:
{bohr_review}

STEPS:
1. Look at actual output files in Sandbox/{project_name}/outputs/ to find statistics
2. Write legends that describe what each panel shows
3. Include sample sizes and statistics ONLY if you find them in files

Save to: Sandbox/{project_name}/{project_name}_legends.md

CRITICAL: Only describe what actually exists. If you can't find a statistic, write "statistics in figure" not a made-up number.

When done: ```done {{"result": "Legends written for X panels", "outputs": ["{project_name}_legends.md"]}}```
"""
    
    gould_legends_result = await _agentic_delegate(
        agent=gould,
        task=gould_legends_task,
        shared_context=shared_context,
        max_iterations=8,
        max_tokens_per_step=4000,
        verbose=True,
        logger=logger,
    )
    tokens_used += TOKEN_PER_CALL * gould_legends_result.get("iterations", 3)
    
    legends_response = gould_legends_result.get("result", "Legends creation incomplete")
    
    # Verify the file was created
    legends_path = project_path / f"{project_name}_legends.md"
    if legends_path.exists():
        print(f"  ✓ Legends file created: {legends_path.name}")
        legends_response = legends_path.read_text()[:2000]
    else:
        print(f"  ⚠️ Legends file not found, using agent output")
        # Save whatever we got
        await _agent_writes_file(gould, f"Sandbox/{project_name}/{project_name}_legends.md", legends_response)
    
    _show_agent("Gould", legends_response, truncate=1200)
    
    # -------------------------------------------------------------------------
    # 5D: Gould creates summary based on ACTUAL results (agentic)
    # -------------------------------------------------------------------------
    _print_substage("5D: Gould Creating Summary (Agentic Mode)")
    
    print("  Gould writing summary document using tools to view results and search literature...")
    
    gould_summary_task = f"""GOAL: Write a mini-paper summary for {project_name}.

BOHR SAW IN THE FIGURES:
{bohr_review}

RESEARCH QUESTION:
{research_question}

Write a short scientific summary with:
- INTRODUCTION: Brief background and hypothesis
- RESULTS: What the figures show (reference them: "Figure 1a shows...")
- DISCUSSION: What it means
- METHODS: Brief description
- REFERENCES: Find 3-5 real citations using web_search

Save to: Sandbox/{project_name}/{project_name}_summary.md

CRITICAL: Only describe actual results. Don't invent statistics.

When done: ```done {{"result": "Summary written", "outputs": ["{project_name}_summary.md"]}}```
"""
    
    gould_summary_result = await _agentic_delegate(
        agent=gould,
        task=gould_summary_task,
        shared_context=shared_context,
        max_iterations=10,
        max_tokens_per_step=5000,
        verbose=True,
        logger=logger,
    )
    tokens_used += TOKEN_PER_CALL * gould_summary_result.get("iterations", 4)
    
    summary_response = gould_summary_result.get("result", "Summary creation incomplete")
    
    # Verify the file was created
    summary_path = project_path / f"{project_name}_summary.md"
    if summary_path.exists():
        print(f"  ✓ Summary file created: {summary_path.name}")
        summary_response = summary_path.read_text()[:3000]
    else:
        print(f"  ⚠️ Summary file not found, using agent output")
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
    
    # Final status and outputs
    print("=" * 70)
    print("  WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\n  Primary outputs in Sandbox/{project_name}/:")
    print(f"    - {project_name}_figures.pdf")
    print(f"    - {project_name}_legends.md")
    print(f"    - {project_name}_summary.md")
    print(f"\n  Total tokens used: ~{tokens_used:,}\n")
    
    # FULLY AGENTIC: Bohr asks user for final approval
    final_approval_task = f"""WORKFLOW COMPLETE. The analysis outputs have been generated:
- {project_name}_figures.pdf
- {project_name}_legends.md
- {project_name}_summary.md

Farber's review: {"Approved" if is_acceptable else "Identified issues"}

YOUR TASK: Get user's final approval.

1. Use user_input.ask to ask the user:
   {{"tool": "user_input", "action": "ask", "params": {{"prompt": "The analysis is complete. Do you approve the outputs, or would you like to iterate? (approve to finish, or describe changes needed)"}}}}

2. If user approves: Signal done with approval
3. If user wants changes: Note their feedback and signal done with the feedback

```done
{{"result": "approved" or "iterate: [feedback]", "approved": true or false, "feedback": "[any feedback]"}}
```
"""
    
    final_result = await _agentic_delegate(
        agent=bohr,
        task=final_approval_task,
        shared_context="Final workflow checkpoint",
        max_iterations=5,
        max_tokens_per_step=2000,
        verbose=True,
        logger=logger,
    )
    tokens_used += TOKEN_PER_CALL * final_result.get("iterations", 2)
    
    # Check if user approved
    result_data = final_result.get("result", {})
    if isinstance(result_data, dict):
        user_approved = result_data.get("approved", False)
        user_feedback = result_data.get("feedback", "")
    else:
        result_str = str(result_data).lower()
        user_approved = "approved" in result_str or "true" in result_str
        user_feedback = str(result_data) if not user_approved else ""
    
    if not user_approved and user_feedback:
        print(f"\n  User requested iteration: {user_feedback[:200]}...")
        print("  (In full implementation, would restart from Stage 2 with feedback)")
    
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
        "user_approved": user_approved,
        "user_feedback": user_feedback,
    }


# =============================================================================
# EXECUTION HELPER (Stages 3 & 4)
# =============================================================================

MAX_BUILD_ITERATIONS = 20  # Max iterations for agentic script building


async def _build_script_with_tools(
    hinton: Agent,
    script_spec: str,
    script_name: str,
    project_name: str,
    project_path: Path,
    workspace_root: Path,
    is_exploratory: bool,
    tokens_used: int,
    logger: Optional[TranscriptLogger] = None,
) -> tuple[str, bool, int]:
    """
    Build a script using TRUE agentic execution (like a VS Code agent).
    
    Hinton operates autonomously in a ReAct loop:
    1. Thinks about what to do
    2. Uses code_editor tools to create/edit files
    3. Runs the script and sees output
    4. Makes surgical fixes based on errors
    5. Continues until script works or max iterations
    
    This is TRUE agentic behavior - Hinton drives the process, not us.
    
    Returns: (final_code, success, tokens_used)
    """
    output_dir_name = "scratch" if is_exploratory else "outputs"
    script_rel_path = f"Sandbox/{project_name}/scripts/{script_name}"
    pkg_constraints = get_package_constraint_prompt()
    
    print(f"      🤖 Hinton working autonomously (agentic mode)...")
    
    # GOAL-ORIENTED task - tell Hinton WHAT to achieve, not HOW
    task = f"""GOAL: Create a working Python script that accomplishes:

{script_spec}

FILE TO CREATE: {script_rel_path}
OUTPUT FILES GO TO: Sandbox/{project_name}/{output_dir_name}/
INPUT DATA IS IN: ReadData/

PACKAGE CONSTRAINTS:
{pkg_constraints}

MANDATORY WORKFLOW - FOLLOW THIS EXACTLY:
1. FIRST: Use code_editor.create to write the COMPLETE script
   {{"tool": "code_editor", "action": "create", "params": {{"path": "{script_rel_path}", "content": "...full script code..."}}}}
   
2. THEN: Run it with terminal
   {{"tool": "terminal", "action": "execute", "params": {{"command": "micromamba run -n minilab python {script_rel_path}"}}}}
   
3. IF IT FAILS: Fix and retry
   - Use code_editor.view to see the code with line numbers
   - Use code_editor.replace to fix specific lines
   - Run again with terminal
   
4. REPEAT until script works

SUCCESS CRITERIA:
- Script runs without errors
- Script produces output files in Sandbox/{project_name}/{output_dir_name}/
- Script prints "SCRIPT COMPLETE: [list of output files]"

CRITICAL RULES:
- Write COMPLETE scripts (never use "..." or abbreviate)
- ALWAYS create the file BEFORE trying to run it
- Use RELATIVE paths (Sandbox/..., ReadData/...) not absolute paths
- Do NOT use cd commands - all paths are relative to workspace root

When the script runs successfully: ```done {{"result": "Script works", "outputs": ["list files created"]}}```
"""
    
    # Let Hinton work autonomously - MORE iterations, trust the agent
    result = await hinton.agentic_execute(
        task=task,
        context=f"Building script for project {project_name}. This is {'exploratory' if is_exploratory else 'complete'} execution.",
        max_iterations=MAX_BUILD_ITERATIONS + 5,  # Give more room to iterate
        max_tokens_per_step=8000,  # More tokens per step for complete scripts
        verbose=True,
        logger=logger,
    )
    
    # Log completion
    if logger:
        logger.log_system_event(
            "script_build",
            f"Hinton {'completed' if result.get('success') else 'failed'} {script_name}",
            {
                "iterations": result.get("iterations", 0),
                "success": result.get("success", False),
                "tool_calls": len(result.get("tool_calls", [])),
            }
        )
    
    # Estimate tokens used (each iteration uses ~TOKEN_PER_CALL)
    tokens_used += result.get("iterations", 1) * TOKEN_PER_CALL
    
    if result.get("success"):
        print(f"        ✅ Script built successfully in {result.get('iterations', '?')} iterations")
        print(f"        Tool calls: {len(result.get('tool_calls', []))}")
        
        # Read the final code
        final_view = await hinton.use_tool("code_editor", action="view", path=script_rel_path)
        if final_view.get("success"):
            # Strip line numbers from view output
            lines = []
            for line in final_view.get("content", "").split('\n'):
                if ' | ' in line:
                    lines.append(line.split(' | ', 1)[1])
                else:
                    lines.append(line)
            final_code = '\n'.join(lines)
            return final_code, True, tokens_used
        else:
            # Fallback: try to read from file directly
            script_path = project_path / "scripts" / script_name
            if script_path.exists():
                final_code = script_path.read_text()
                return final_code, True, tokens_used
    
    # Failed
    print(f"        ❌ Script building failed after {result.get('iterations', '?')} iterations")
    if result.get("error"):
        print(f"        Error: {result.get('error')[:200]}")
    
    # Try to return whatever code exists
    script_path = project_path / "scripts" / script_name
    if script_path.exists():
        final_code = script_path.read_text()
        return final_code, False, tokens_used
    
    return "", False, tokens_used


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
    
    GOAL-ORIENTED APPROACH:
    - Give agents clear goals, not step-by-step recipes
    - Let them iterate freely until goals are met
    - Verify outcomes (files exist, figures generated) not process
    
    Flow: Dayhoff plans → Hinton executes iteratively → Bayes verifies → repeat if needed
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
    
    # Define success criteria based on stage type
    if is_exploratory:
        success_criteria = "PNG files in scratch/ showing data exploration"
        output_location = "scratch"
    else:
        success_criteria = f"{project_name}_figures.pdf in project root AND PNG files in outputs/"
        output_location = "outputs"
    
    # Step 1: Have Dayhoff create a simple task breakdown (not over-planned)
    _print_substage("Dayhoff: Quick Task Breakdown")
    
    dayhoff_breakdown = await dayhoff.arespond(f"""Break down this execution plan into 2-4 scripts maximum.

EXECUTION PLAN:
{execution_plan[:4000]}

Keep it simple. For each script, just provide:
- Script name (e.g., 01_explore.py, 02_analysis.py, 03_figures.py)
- One-sentence goal
- What outputs it should create

Format:
### 01_scriptname.py
Goal: [one sentence]
Outputs: [list files]

Don't over-plan. Hinton will figure out the details.""", max_tokens=2000)
    tokens_used += TOKEN_PER_CALL
    if logger:
        logger.log_agent_response(dayhoff.display_name, dayhoff.id, dayhoff_breakdown[:500], TOKEN_PER_CALL)
    
    _show_agent("Dayhoff", dayhoff_breakdown, truncate=1000)
    
    # Parse script names (simple extraction)
    script_specs = []
    for line in dayhoff_breakdown.split('\n'):
        if '.py' in line and ('###' in line or line.strip().startswith('0') or line.strip().startswith('1')):
            match = re.search(r'(\d+_?\w*\.py)', line)
            if match:
                script_name = match.group(1)
                # Get the next few lines as spec
                idx = dayhoff_breakdown.find(line)
                spec = dayhoff_breakdown[idx:idx+500].split('###')[0]
                script_specs.append((script_name, spec))
    
    if not script_specs:
        # Fallback: single comprehensive script
        script_specs = [("analysis_pipeline.py", execution_plan[:2000])]
    
    print(f"\n    Scripts to build: {[s[0] for s in script_specs]}")
    
    # Step 2: Let Hinton build and run each script autonomously
    _print_substage("Hinton: Autonomous Script Development")
    
    execution_results = {}
    
    for idx, (script_name, script_spec) in enumerate(script_specs, 1):
        print(f"\n    [{idx}/{len(script_specs)}] {script_name}")
        
        script_code, success, tokens_used = await _build_script_with_tools(
            hinton=hinton,
            script_spec=script_spec,
            script_name=script_name,
            project_name=project_name,
            project_path=project_path,
            workspace_root=workspace_root,
            is_exploratory=is_exploratory,
            tokens_used=tokens_used,
            logger=logger,
        )
        
        execution_results[script_name] = {"success": success, "code": script_code[:500]}
        
        if success:
            print(f"      ✅ {script_name} complete")
        else:
            print(f"      ⚠️ {script_name} had issues but continuing...")
    
    # Step 3: VERIFY OUTCOMES (not process)
    _print_substage("Outcome Verification")
    
    output_dir = scratch_dir if is_exploratory else outputs_dir
    png_files = list(output_dir.glob("*.png")) if output_dir.exists() else []
    figures_pdf = project_path / f"{project_name}_figures.pdf"
    
    print(f"    Checking outputs...")
    print(f"    - PNG files in {output_location}/: {len(png_files)}")
    print(f"    - Figures PDF exists: {figures_pdf.exists()}")
    
    # If outputs are missing, give Hinton another chance with a simpler task
    if len(png_files) < 2 and not is_exploratory:
        print(f"\n    ⚠️ Insufficient outputs. Giving Hinton a focused recovery task...")
        
        recovery_script = f"Sandbox/{project_name}/scripts/recovery_figures.py"
        recovery_task = f"""URGENT: The analysis didn't produce enough figures.

SITUATION:
- Expected: Multiple PNG files in Sandbox/{project_name}/outputs/
- Found: {len(png_files)} PNG files

GOAL: Create a script that:
1. Loads data from ReadData/
2. Produces at least 3-4 publication-quality figures
3. Saves them as PNGs to Sandbox/{project_name}/outputs/

WORKING PLAN SUMMARY:
{working_plan[:2000]}

MANDATORY WORKFLOW - FOLLOW THIS EXACTLY:
1. FIRST: Use code_editor.create to write the COMPLETE script
   {{"tool": "code_editor", "action": "create", "params": {{"path": "{recovery_script}", "content": "...full script code..."}}}}
   
2. THEN: Run it with terminal
   {{"tool": "terminal", "action": "execute", "params": {{"command": "micromamba run -n minilab python {recovery_script}"}}}}
   
3. IF IT FAILS: Fix and retry
   - Use code_editor.view to see the code with line numbers
   - Use code_editor.replace to fix specific lines
   - Run again with terminal

CRITICAL RULES:
- Write COMPLETE scripts (never use "..." or abbreviate)
- ALWAYS create the file BEFORE trying to run it
- Use RELATIVE paths (Sandbox/..., ReadData/...) not absolute paths
- Do NOT use cd commands - all paths are relative to workspace root

When figures exist: ```done {{"result": "Figures created", "outputs": ["list pngs"]}}```
"""
        
        recovery_result = await hinton.agentic_execute(
            task=recovery_task,
            context=f"Recovery task for {project_name}",
            max_iterations=15,  # Extra iterations for recovery
            max_tokens_per_step=5000,
            verbose=True,
            logger=logger,
        )
        tokens_used += recovery_result.get("iterations", 1) * TOKEN_PER_CALL
        
        # Re-check outputs
        png_files = list(output_dir.glob("*.png")) if output_dir.exists() else []
        print(f"    After recovery: {len(png_files)} PNG files")
    
    # Step 4: Bayes quick validation (if outputs exist)
    if png_files:
        _print_substage("Bayes: Quick Quality Check")
        
        bayes_check = await bayes.arespond_with_vision(
            user_message=f"""Quick quality check on these analysis outputs.

Are these figures:
1. Properly labeled (axes, titles)?
2. Showing real data patterns (not all zeros/noise)?
3. Publication-ready or close?

Just respond: GOOD / NEEDS_WORK: [one-line issue] / FAIL: [critical problem]""",
            image_paths=[str(f) for f in png_files[:4]],
            max_tokens=500,
        )
        tokens_used += TOKEN_PER_CALL
        
        print(f"    Bayes says: {bayes_check[:100]}")
    
    # Compile results summary
    successes = sum(1 for r in execution_results.values() if r.get("success"))
    figures_exist = (project_path / f"{project_name}_figures.pdf").exists()
    
    return {
        "success": successes > 0 and (len(png_files) >= 2 or is_exploratory),
        "tokens_used": tokens_used,
        "results": f"Scripts: {successes}/{len(execution_results)} succeeded. Outputs: {len(png_files)} PNGs.",
        "needs_iteration": len(png_files) < 2 and not is_exploratory,
        "script_results": execution_results,
        "figures_exist": figures_exist if not is_exploratory else None,
    }
