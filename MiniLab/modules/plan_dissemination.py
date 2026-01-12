"""
Plan Dissemination: Make the full plan explicit and visible to agents.

This module provides utilities to:
1. Format the task graph as a clear, human-readable plan
2. Extract agent responsibilities from planning output
3. Inject explicit guardrails (outputs, file locations, naming conventions)
4. Build comprehensive context strings for agent prompts
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
    
    lines = ["TASK GRAPH", "=" * 60, ""]
    
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
    
    # Output pending tasks
    if pending:
        lines.append("## PENDING TASKS")
        for task in pending:
            lines.append(f"\n### Task: {task['id']}")
            lines.append(f"Title: {task['title']}")
            if task['agents']:
                lines.append(f"Assigned to: {', '.join(task['agents'])}")
            if task['outputs']:
                lines.append(f"Required outputs: {', '.join(task['outputs'])}")
    
    # Output task dependencies
    if edges:
        lines.append("\n## TASK DEPENDENCIES")
        for edge_id, edge_data in edges.items():
            source = edge_data.get("source", "")
            target = edge_data.get("target", "")
            if source and target:
                lines.append(f"- {source} → {target}")
    
    # Output completed tasks
    if completed:
        lines.append("\n## COMPLETED TASKS")
        for task in completed:
            lines.append(f"- {task['id']}: {task['title']}")
    
    return "\n".join(lines)


def extract_agent_responsibilities(
    planning_output: str,
    task_graph: Optional[Any] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Extract agent responsibilities from planning output.
    
    Args:
        planning_output: The analysis_plan text from planning
        task_graph: Optional TaskGraph for more structured info
        
    Returns:
        Dict mapping agent_name -> {tasks, outputs, deliverables}
    """
    responsibilities: Dict[str, Dict[str, Any]] = {}
    
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
    
    lines = planning_output.split('\n')
    current_agent = None
    
    for line in lines:
        line_lower = line.lower()
        
        for agent in agents:
            if f"{agent}" in line_lower or f"agent {agent}" in line_lower:
                current_agent = agent
                break
        
        if current_agent and any(kw in line_lower for kw in ["should", "will", "task", "produce", "output"]):
            clean_task = line.strip('- •*').strip()
            if clean_task and len(clean_task) > 5:
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
    
    Args:
        agent_name: Name of the agent getting the context
        task_graph_plan: Formatted task graph
        responsibilities: Agent responsibilities
        project_spec: Project specification string
        additional_context: Optional additional context
        
    Returns:
        Formatted context string for agent prompt
    """
    lines = []
    
    lines.append("=" * 70)
    lines.append(f"CONTEXT FOR AGENT: {agent_name.upper()}")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append("## PROJECT SPECIFICATION")
    lines.append(project_spec[:1000])
    if len(project_spec) > 1000:
        lines.append(f"...[+{len(project_spec) - 1000} characters]...")
    lines.append("")
    
    lines.append("## FULL TASK GRAPH")
    lines.append(task_graph_plan)
    lines.append("")
    
    lines.append(f"## YOUR RESPONSIBILITIES (Agent {agent_name.upper()})")
    agent_resp = responsibilities.get(agent_name, {})
    
    tasks = agent_resp.get("tasks", [])
    if tasks:
        lines.append("Tasks to perform:")
        for i, task in enumerate(tasks, 1):
            lines.append(f"  {i}. {task}")
    else:
        lines.append("Based on the plan above, determine your role and specific tasks.")
    
    outputs = agent_resp.get("outputs", [])
    if outputs:
        lines.append("\nRequired outputs:")
        for output in outputs:
            lines.append(f"  - {output}")
    
    lines.append("")
    
    # Explicit guardrails aligned with outline directory structure
    lines.append("## OUTPUT GUARDRAILS")
    lines.append("1. File Creation:")
    lines.append("   - Scripts go in: scripts/")
    lines.append("   - Results go in: results/figures/ or results/tables/")
    lines.append("   - Reports go in: reports/")
    lines.append("   - Artifacts go in: artifacts/")
    lines.append("   - Data outputs go in: data/processed/")
    lines.append("")
    lines.append("2. Naming Conventions:")
    lines.append("   - Descriptive, lowercase, underscores for spaces")
    lines.append("   - Example: 01_preprocess_data.py, survival_curve.png")
    lines.append("")
    lines.append("3. Critical Constraint:")
    lines.append("   - ONLY create outputs you are explicitly tasked to create")
    lines.append("   - Do NOT create duplicate files or redundant outputs")
    lines.append("")
    
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
    
    Args:
        original_prompt: The task prompt
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
    
    return f"""{agent_context}

================================================================================
YOUR TASK
================================================================================

{original_prompt}"""


def get_output_guardrails(agent_name: str, task_type: str) -> Dict[str, str]:
    """
    Get strict output guardrails for an agent's specific task.
    
    Args:
        agent_name: Agent name
        task_type: Type of task
        
    Returns:
        Dict with output requirements
    """
    guardrails = {
        "code_file": {
            "directory": "scripts",
            "extension": ".py",
            "naming": "NN_descriptive_name.py",
            "requirements": "Executable, well-commented, error handling"
        },
        "data_file": {
            "directory": "data/processed",
            "extension": ".csv or .json",
            "naming": "descriptive_name.csv",
            "requirements": "Clean, documented columns, consistent format"
        },
        "figure": {
            "directory": "results/figures",
            "extension": ".png or .pdf",
            "naming": "figure_NN_title.png",
            "requirements": "High quality, readable labels, legend/caption"
        },
        "document": {
            "directory": "reports or artifacts",
            "extension": ".md or .docx",
            "naming": "section_name.md",
            "requirements": "Markdown format, structured headers, citations"
        },
    }
    
    return guardrails.get(task_type, {})
