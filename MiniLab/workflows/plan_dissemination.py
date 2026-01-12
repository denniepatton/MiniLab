"""
Plan Dissemination: Make the full workflow plan explicit and visible to agents.

This module provides utilities to:
1. Format the task graph as a clear, human-readable plan
2. Extract agent responsibilities from planning committee output
3. Inject explicit guardrails (outputs, file locations, naming conventions)
4. Build comprehensive context strings for agent prompts

Key Insight: Agents given a full plan + explicit guardrails = fewer fragments, 
better coherence. This prevents the root cause of Issue 2 (folder fragmentation)
and Issue 3 (plan not disseminated).
"""

from typing import Any, Optional, Dict, List
from pathlib import Path


def format_task_graph_as_plan(task_graph: Any) -> str:
    """
    Format a TaskGraph as a human-readable plan for agent consumption.
    
    Args:
        task_graph: TaskGraph object
        
    Returns:
        Formatted plan string
    """
    if not task_graph:
        return "No task graph available."
    
    try:
        nodes = task_graph.nodes
        edges = task_graph.edges
    except (AttributeError, TypeError):
        return "Task graph format not supported."
    
    if not nodes:
        return "Task graph is empty."
    
    lines = ["WORKFLOW TASK GRAPH", "=" * 60, ""]
    
    # Group tasks by status
    pending = []
    in_progress = []
    completed = []
    failed = []
    
    for node_id, node_data in nodes.items():
        status = node_data.get("status", "pending").lower()
        node_info = {
            "id": node_id,
            "title": node_data.get("title", node_id),
            "agents": node_data.get("agents", []),
            "outputs": node_data.get("outputs", []),
        }
        
        if status == "completed":
            completed.append(node_info)
        elif status == "in_progress":
            in_progress.append(node_info)
        elif status == "failed":
            failed.append(node_info)
        else:
            pending.append(node_info)
    
    # Output status summary
    lines.append("## STATUS SUMMARY")
    lines.append(f"- Pending: {len(pending)} tasks")
    lines.append(f"- In Progress: {len(in_progress)} tasks")
    lines.append(f"- Completed: {len(completed)} tasks")
    lines.append(f"- Failed: {len(failed)} tasks")
    lines.append("")
    
    # Output pending tasks (most relevant to current work)
    if pending:
        lines.append("## PENDING TASKS (What needs to be done)")
        for task in pending:
            lines.append(f"\n### Task: {task['id']}")
            lines.append(f"Title: {task['title']}")
            if task['agents']:
                lines.append(f"Assigned to: {', '.join(task['agents'])}")
            if task['outputs']:
                lines.append(f"Required outputs: {', '.join(task['outputs'])}")
    
    # Output task dependencies
    if edges:
        lines.append("\n## TASK DEPENDENCIES (Workflow order)")
        for edge_id, edge_data in edges.items():
            source = edge_data.get("source", "")
            target = edge_data.get("target", "")
            if source and target:
                lines.append(f"- {source} → {target}")
    
    # Output completed tasks (for reference)
    if completed:
        lines.append("\n## COMPLETED TASKS (For context)")
        for task in completed:
            lines.append(f"- {task['id']}: {task['title']}")
    
    return "\n".join(lines)


def extract_agent_responsibilities(
    planning_output: str,
    task_graph: Optional[Any] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Extract agent responsibilities from planning committee output.
    
    This parses the analysis_plan and responsibilities from Bohr's planning
    to extract what each agent should do, what outputs they must produce,
    and where those outputs go.
    
    Args:
        planning_output: The analysis_plan text from planning_committee workflow
        task_graph: Optional TaskGraph for more structured info
        
    Returns:
        Dict mapping agent_name -> {tasks, outputs, deliverables}
    """
    responsibilities: Dict[str, Dict[str, Any]] = {}
    
    # Parse responsibilities from planning output
    # Look for patterns like "Agent X should...", "Agent X will produce...", etc.
    
    agents = [
        "bohr", "feynman", "hinton", "dayhoff", "bayes", "shannon",
        "greider", "gould", "farber"
    ]
    
    for agent in agents:
        responsibilities[agent] = {
            "tasks": [],
            "outputs": [],
            "deliverables": [],
            "file_locations": [],
        }
    
    # If planning_output contains explicit assignments, extract them
    # This is a heuristic parser - real implementation would be more sophisticated
    lines = planning_output.split('\n')
    current_agent = None
    
    for line in lines:
        line_lower = line.lower()
        
        # Check if this line mentions an agent
        for agent in agents:
            if f"{agent}" in line_lower or f"agent {agent}" in line_lower:
                current_agent = agent
                break
        
        # If we found an agent in a line with task keywords, extract the task
        if current_agent and any(kw in line_lower for kw in ["should", "will", "task", "produce", "output"]):
            # Clean up the line and add to current agent's tasks
            clean_task = line.strip('- •*').strip()
            if clean_task and len(clean_task) > 5:  # Skip very short lines
                responsibilities[current_agent]["tasks"].append(clean_task)
    
    return responsibilities


def build_agent_context(
    agent_name: str,
    task_graph_plan: str,
    responsibilities: Dict[str, Any],
    project_spec: str,
    additional_context: Optional[str] = None,
) -> str:
    """
    Build a comprehensive context string for an agent prompt.
    
    This combines:
    1. The full task graph (so agent knows what fits in the overall plan)
    2. Agent's specific responsibilities (what THIS agent should do)
    3. Project specification (what we're trying to accomplish)
    4. Explicit guardrails (outputs, file paths, naming conventions)
    
    Args:
        agent_name: Name of the agent getting the context
        task_graph_plan: Formatted task graph (from format_task_graph_as_plan)
        responsibilities: Agent responsibilities (from extract_agent_responsibilities)
        project_spec: Project specification string
        additional_context: Optional additional context
        
    Returns:
        Formatted context string for agent prompt
    """
    lines = []
    
    # Header
    lines.append("=" * 70)
    lines.append(f"CONTEXT FOR AGENT: {agent_name.upper()}")
    lines.append("=" * 70)
    lines.append("")
    
    # Project context
    lines.append("## PROJECT SPECIFICATION")
    lines.append(project_spec[:1000])  # First 1000 chars
    if len(project_spec) > 1000:
        lines.append(f"...[+{len(project_spec) - 1000} characters]...")
    lines.append("")
    
    # Full workflow plan
    lines.append("## FULL WORKFLOW PLAN (Your role in context)")
    lines.append(task_graph_plan)
    lines.append("")
    
    # This agent's specific responsibilities
    lines.append(f"## YOUR RESPONSIBILITIES (Agent {agent_name.upper()})")
    agent_resp = responsibilities.get(agent_name, {})
    
    tasks = agent_resp.get("tasks", [])
    if tasks:
        lines.append("Tasks to perform:")
        for i, task in enumerate(tasks, 1):
            lines.append(f"  {i}. {task}")
    else:
        # Fallback if no specific tasks extracted - provide guidance
        lines.append("Based on the plan above, determine your role and specific tasks.")
    
    outputs = agent_resp.get("outputs", [])
    if outputs:
        lines.append("\nRequired outputs:")
        for output in outputs:
            lines.append(f"  - {output}")
    
    lines.append("")
    
    # Explicit guardrails
    lines.append("## EXPLICIT GUARDRAILS (Must follow these)")
    lines.append("1. File Creation:")
    lines.append("   - Only create files in designated directories (specified in task)")
    lines.append("   - Follow naming conventions: descriptive, lowercase, hyphens for spaces")
    lines.append("   - Example: analysis_results.md, feature_importance.png, model_performance.json")
    lines.append("")
    lines.append("2. Output Formats:")
    lines.append("   - Code: Well-commented, executable files in /analysis/code/")
    lines.append("   - Data: CSV/JSON in /analysis/data/")
    lines.append("   - Visualizations: PNG/PDF in /analysis/figures/")
    lines.append("   - Documentation: Markdown in /analysis/ or /outputs/")
    lines.append("")
    lines.append("3. Integration Points:")
    lines.append("   - Reference outputs from previous tasks when available")
    lines.append("   - Use consistent naming and formats across agents")
    lines.append("   - Each output should have metadata (date, agent, purpose)")
    lines.append("")
    lines.append("4. Critical Constraint:")
    lines.append("   - ONLY create outputs you are explicitly tasked to create")
    lines.append("   - Do NOT create duplicate files or redundant outputs")
    lines.append("   - Do NOT create files outside the project directory")
    lines.append("")
    
    # Additional context
    if additional_context:
        lines.append("## ADDITIONAL CONTEXT")
        lines.append(additional_context)
        lines.append("")
    
    return "\n".join(lines)


def inject_plan_into_prompt(
    original_prompt: str,
    agent_name: str,
    task_graph: Any,
    responsibilities: Dict[str, Dict[str, Any]],
    project_spec: str,
) -> str:
    """
    Inject task graph and plan context into an agent prompt.
    
    This is the main entry point: takes an agent's task prompt and prepends
    comprehensive context about the overall plan and their role.
    
    Args:
        original_prompt: The task prompt from the workflow
        agent_name: Agent's name
        task_graph: TaskGraph object
        responsibilities: Agent responsibilities
        project_spec: Project specification
        
    Returns:
        Enhanced prompt with plan context
    """
    task_graph_plan = format_task_graph_as_plan(task_graph)
    agent_context = build_agent_context(
        agent_name,
        task_graph_plan,
        responsibilities,
        project_spec,
    )
    
    # Prepend context to prompt with clear separator
    return f"""{agent_context}

================================================================================
YOUR TASK (Complete this assignment following the guardrails above)
================================================================================

{original_prompt}"""


def get_output_guardrails(agent_name: str, task_type: str) -> Dict[str, str]:
    """
    Get strict output guardrails for an agent's specific task.
    
    Args:
        agent_name: Agent name
        task_type: Type of task (e.g., 'data_prep', 'model_dev', 'visualization')
        
    Returns:
        Dict with output requirements
    """
    guardrails = {
        # Code outputs
        "code_file": {
            "directory": "analysis/code",
            "extension": ".py",
            "naming": "descriptive-task-name.py",
            "requirements": "Executable, well-commented, error handling"
        },
        # Data outputs
        "data_file": {
            "directory": "analysis/data",
            "extension": ".csv or .json",
            "naming": "descriptive-data-name.csv",
            "requirements": "Clean, documented columns, consistent format"
        },
        # Visualization outputs
        "figure": {
            "directory": "analysis/figures",
            "extension": ".png or .pdf",
            "naming": "figure-01-descriptive-title.png",
            "requirements": "High quality, readable labels, legend/caption"
        },
        # Documentation
        "document": {
            "directory": "outputs or analysis",
            "extension": ".md",
            "naming": "section-name.md",
            "requirements": "Markdown format, structured headers, citations"
        },
    }
    
    return guardrails.get(task_type, {})
