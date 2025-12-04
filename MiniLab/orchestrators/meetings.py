from __future__ import annotations

from typing import Dict, List, Optional

from MiniLab.agents.base import Agent
from MiniLab.storage.transcript import TranscriptLogger

__all__ = [
    "run_user_team_meeting",
    "run_pi_coordinated_meeting", 
    "run_internal_team_meeting",
    "run_triads_meeting",
]


async def run_user_team_meeting(
    agents: Dict[str, Agent],
    user_prompt: str,
    project_context: str | None = None,
) -> dict:
    """
    You ask a question; all agents respond in parallel.
    Returns a dict of agent_id -> response.
    """
    import asyncio
    
    # Run all agents in parallel
    tasks = {
        agent_id: agent.arespond(user_prompt, context=project_context)
        for agent_id, agent in agents.items()
    }
    
    # Gather all responses
    responses = {}
    for agent_id, task in tasks.items():
        responses[agent_id] = await task
    
    return responses


async def run_pi_coordinated_meeting(
    agents: Dict[str, Agent],
    pi_agent_id: str,
    user_prompt: str,
    project_context: str | None = None,
    max_total_tokens: int = 300000,
    logger: Optional[TranscriptLogger] = None,
) -> dict:
    """
    User speaks to PI, who coordinates with other agents as needed.
    
    The PI receives the user's question and decides which agents to consult.
    Other agents discuss among themselves, then PI synthesizes their input
    and responds to the user.
    
    Args:
        agents: All available agents
        pi_agent_id: The PI agent (usually "bohr")
        user_prompt: User's question
        project_context: Optional project context
        max_total_tokens: Maximum tokens across all API calls
        
    Returns:
        dict with 'pi_response', 'consultations', 'token_usage'
    """
    import asyncio
    import sys
    
    pi_agent = agents[pi_agent_id]
    other_agents = {k: v for k, v in agents.items() if k != pi_agent_id}
    
    # Token tracking (approximate - actual tracking would need API response parsing)
    estimated_tokens = 0
    token_limit_per_call = 2000  # Conservative estimate per response
    
    # Spinner for progress indication
    def show_progress(message):
        sys.stdout.write(f"\r{message}")
        sys.stdout.flush()
    
    # Step 1: PI receives user query and decides who to consult
    coordination_prompt = f"""You are coordinating the MiniLab team to answer this user question:

User Question: {user_prompt}

Available team members:
{chr(10).join(f"- {agent.display_name} ({aid}): {agent.role}" for aid, agent in other_agents.items())}

CRITICAL INSTRUCTIONS:
1. For file operations: delegate to Hinton with filesystem tool
2. To RUN code/commands: delegate to Hinton/Shannon/Bayes with terminal tool
3. For consultation: use CONSULT format
4. For simple answers: respond directly

DELEGATION FORMATS:

Write a file:
DELEGATE: hinton
tool: filesystem
action: write
path: analysis.py
content: import pandas as pd
print("Hello")
---END

Run a command/script:
DELEGATE: hinton
tool: terminal
command: cd Sandbox/ProjectName && python analysis.py
---END

List directory:
DELEGATE: hinton
tool: filesystem
action: list
path: ReadData/Pluvicto
---END

Project context: {project_context or "None"}"""

    show_progress(f"⚙️  {pi_agent.display_name} is thinking...")
    pi_coordination = await pi_agent.arespond(coordination_prompt)
    estimated_tokens += token_limit_per_call
    sys.stdout.write("\r" + " " * 50 + "\r")
    sys.stdout.flush()
    
    # Step 2: Check if PI wants to delegate to another agent for tool use
    tool_results = []
    pi_response = None
    
    if "DELEGATE:" in pi_coordination:
        # Parse delegation
        try:
            delegate_section = pi_coordination.split("DELEGATE:")[1].split("---END")[0].strip()
            lines = [l.strip() for l in delegate_section.split("\n") if l.strip()]
            
            # First line is the agent to delegate to
            delegate_agent_id = lines[0].strip()
            
            # Parse tool parameters
            tool_name = "filesystem"  # Default
            tool_params = {}
            current_key = None
            content_lines = []
            
            for line in lines[1:]:
                if line.startswith("tool:"):
                    tool_name = line.split(":", 1)[1].strip()
                elif line.startswith("action:"):
                    if current_key == "content" and content_lines:
                        tool_params["content"] = "\n".join(content_lines)
                        content_lines = []
                    current_key = "action"
                    tool_params["action"] = line.split(":", 1)[1].strip()
                elif line.startswith("path:"):
                    if current_key == "content" and content_lines:
                        tool_params["content"] = "\n".join(content_lines)
                        content_lines = []
                    current_key = "path"
                    tool_params["path"] = line.split(":", 1)[1].strip()
                elif line.startswith("command:"):
                    if current_key == "content" and content_lines:
                        tool_params["content"] = "\n".join(content_lines)
                        content_lines = []
                    current_key = "command"
                    tool_params["command"] = line.split(":", 1)[1].strip()
                elif line.startswith("content:"):
                    current_key = "content"
                    value = line.split(":", 1)[1].strip() if ":" in line and len(line.split(":", 1)) > 1 else ""
                    if value:
                        content_lines.append(value)
                elif current_key == "content":
                    content_lines.append(line)
            
            # Add any remaining content
            if current_key == "content" and content_lines:
                tool_params["content"] = "\n".join(content_lines)
            
            # Validate required parameters based on tool
            validation_failed = False
            if tool_name == "filesystem" and "action" not in tool_params:
                print(f"\n⚠️  Missing 'action' parameter for filesystem tool. Skipping delegation.\n")
                pi_response = "Delegation failed: filesystem tool requires 'action' parameter (list, read, write, create_dir, etc.)"
                validation_failed = True
            elif tool_name == "terminal" and "command" not in tool_params:
                print(f"\n⚠️  Missing 'command' parameter for terminal tool. Skipping delegation.\n")
                pi_response = "Delegation failed: terminal tool requires 'command' parameter"
                validation_failed = True
            
            # Get the delegated agent and execute tool only if validation passed
            if not validation_failed:
                delegated_agent = other_agents.get(delegate_agent_id)
                if delegated_agent and delegated_agent.has_tool(tool_name):
                    show_progress(f"🔧 {pi_agent.display_name} is delegating to {delegated_agent.display_name}...")
                    result = await delegated_agent.use_tool(tool_name, **tool_params)
                    sys.stdout.write("\r" + " " * 60 + "\r")
                    sys.stdout.flush()
                    
                    tool_results.append(result)
                    
                    # Check if tool operation failed
                    if not result.get("success", False):
                        error_details = result.get("error", "Unknown error")
                        
                        # Determine if this is a RECOVERABLE error or a BLOCKER
                        recoverable_patterns = [
                            "Cannot read directory",
                            "Use action='list'",
                            "Not a directory",
                            "Use action='read'",
                            "Unknown action: create",
                        ]
                        
                        path_param = tool_params.get('path', '')
                        is_missing_sandbox_dir = (
                            "Directory not found" in error_details and 
                            path_param.startswith("Sandbox/")
                        )
                        
                        is_bare_sandbox_access = (
                            path_param == "Sandbox" or 
                            "File not found: Sandbox" in error_details
                        )
                        
                        is_outputs_access_attempt = (
                            "Directory not found" in error_details and
                            path_param.startswith("Outputs/")
                        )
                        
                        is_recoverable = (
                            any(pattern in error_details for pattern in recoverable_patterns) or
                            is_missing_sandbox_dir or
                            is_outputs_access_attempt or
                            is_bare_sandbox_access
                        )
                        
                        if is_recoverable:
                            # AUTO-CORRECT common mistakes
                            if "Cannot read directory" in error_details and "Use action='list'" in error_details:
                                print(f"\n🔧 Auto-correcting: changing 'read' to 'list' for directory")
                                tool_params['action'] = 'list'
                                result = await delegated_agent.use_tool(tool_name, **tool_params)
                                tool_results[-1] = result
                                
                                if result.get("success"):
                                    print(f"✓ Auto-correction successful\n")
                                    pi_response = f"I delegated a task to {delegated_agent.display_name}, and it was successful after auto-correction."
                                else:
                                    print(f"⚠️  Auto-correction also failed: {result.get('error')}\n")
                                    pi_response = f"Tool operation failed even after auto-correction: {result.get('error')}"
                            elif "Unknown action: create" in error_details:
                                print(f"\n🔧 Auto-correcting: changing 'create' to 'write' for file creation")
                                tool_params['action'] = 'write'
                                result = await delegated_agent.use_tool(tool_name, **tool_params)
                                tool_results[-1] = result
                                
                                if result.get("success"):
                                    print(f"✓ Auto-correction successful\n")
                                    pi_response = f"I delegated a task to {delegated_agent.display_name}, and it was successful after auto-correction."
                                else:
                                    print(f"⚠️  Auto-correction also failed: {result.get('error')}\n")
                                    pi_response = f"Tool operation failed even after auto-correction: {result.get('error')}"
                            elif is_bare_sandbox_access:
                                pi_response = (
                                    f"Error: You tried to access 'Sandbox' directly. Use your project directory instead.\n"
                                    f"All your files should go in the project directory that was created for you.\n"
                                    f"Do NOT use bare 'Sandbox' - work in your project subdirectory."
                                )
                            elif is_outputs_access_attempt:
                                pi_response = (
                                    f"Error: Outputs/ directory not accessible. Work in your project directory.\n"
                                    f"Deliverables will be auto-copied at the end."
                                )
                            elif is_missing_sandbox_dir:
                                pi_response = (
                                    f"The directory doesn't exist yet, but you can create it:\n\n"
                                    f"Error: {error_details}\n\n"
                                    f"Solution: First use action='create_dir' to create the directory, then proceed."
                                )
                            else:
                                pi_response = (
                                    f"The tool operation failed, but this is correctable:\n\n"
                                    f"Error: {error_details}\n\n"
                                    f"Please try again with the correct action."
                                )
                        else:
                            # BLOCKER ERROR: Stop immediately
                            print(f"\n{'=' * 80}")
                            print(f"🛑 QUICK-FAIL: Unrecoverable tool error - stopping execution")
                            print(f"{'=' * 80}")
                            print(f"Agent: {delegated_agent.display_name}")
                            print(f"Tool: {tool_name}")
                            print(f"Error: {error_details}")
                            print(f"{'=' * 80}\n")
                            
                            return {
                                "pi_response": f"Tool operation failed: {error_details}",
                                "consultations": {},
                                "tool_results": tool_results,
                                "estimated_tokens": estimated_tokens,
                                "error": error_details,
                                "failed_quick": True,
                            }
                    else:
                        # Success case
                        pi_response = f"I delegated a task to {delegated_agent.display_name}, and it was successful!"
                else:
                    error_msg = f"Delegation target '{delegate_agent_id}' is not available or doesn't have the '{tool_name}' tool."
                    print(f"\n{'=' * 80}")
                    print(f"🛑 QUICK-FAIL: Delegation failed")
                    print(f"{'=' * 80}")
                    print(f"Attempted to delegate to: {delegate_agent_id}")
                    print(f"Tool required: {tool_name}")
                    print(f"Available agents: {list(other_agents.keys())}")
                    if delegated_agent:
                        print(f"Agent's tools: {list(delegated_agent.tool_instances.keys())}")
                    print(f"{'=' * 80}\n")
                    
                    return {
                        "pi_response": error_msg,
                        "consultations": {},
                        "tool_results": [],
                        "estimated_tokens": estimated_tokens,
                        "error": error_msg,
                        "failed_quick": True,
                    }
                
        except Exception as e:
            print(f"⚠️  Error parsing delegation: {e}")
            import traceback
            traceback.print_exc()
            pi_response = f"I tried to delegate a task but encountered a parsing error: {e}"
    
    # Step 3: Parse PI's coordination response for consultations
    consultations = {}
    if "CONSULT:" in pi_coordination and not tool_results:
        # PI wants to consult others
        consult_section = pi_coordination.split("CONSULT:")[1].strip()
        lines = [l.strip() for l in consult_section.split("\n") if l.strip().startswith("-")]
        
        for line in lines:
            if estimated_tokens + token_limit_per_call > max_total_tokens:
                print(f"⚠️  Token limit approaching ({estimated_tokens}/{max_total_tokens}), skipping remaining consultations")
                break
                
            # Parse: "- agent_id: question"
            parts = line[1:].strip().split(":", 1)
            if len(parts) == 2:
                agent_id = parts[0].strip()
                question = parts[1].strip()
                
                if agent_id in other_agents:
                    show_progress(f"💬 Consulting {other_agents[agent_id].display_name}...")
                    response = await other_agents[agent_id].arespond(
                        question,
                        context=project_context
                    )
                    sys.stdout.write("\r" + " " * 60 + "\r")  # Clear spinner
                    sys.stdout.flush()
                    consultations[agent_id] = {
                        "question": question,
                        "response": response
                    }
                    estimated_tokens += token_limit_per_call
        
        # Step 4: PI synthesizes consultation results
        if consultations:
            consultation_summary = "\n\n".join([
                f"{other_agents[aid].display_name}'s input:\nQuestion: {c['question']}\nResponse: {c['response']}"
                for aid, c in consultations.items()
            ])
            
            synthesis_prompt = f"""Original user question: {user_prompt}

You consulted with team members and received these responses:

{consultation_summary}

Now synthesize their input and provide a comprehensive answer to the user. Be concise but complete."""

            show_progress(f"🧠 {pi_agent.display_name} is synthesizing responses...")
            pi_response = await pi_agent.arespond(synthesis_prompt)
            estimated_tokens += token_limit_per_call
            sys.stdout.write("\r" + " " * 60 + "\r")  # Clear spinner
            sys.stdout.flush()
        else:
            pi_response = pi_coordination
    
    # Handle case where no tool use or consultation happened
    if pi_response is None:
        pi_response = pi_coordination
    
    return {
        "pi_response": pi_response,
        "consultations": consultations,
        "tool_results": tool_results,
        "estimated_tokens": estimated_tokens,
    }


async def run_internal_team_meeting(
    agents: Dict[str, Agent],
    agenda: str,
    project_context: str | None = None,
    rounds: int = 2,
) -> List[dict]:
    """
    All agents discuss among themselves for a few rounds.
    Very simple initial implementation: in each round,
    we aggregate previous messages into shared context and ask each agent again.
    """
    history: List[dict] = []
    shared_context = project_context or ""

    for r in range(rounds):
        round_responses = {}
        prompt = (
            f"Internal team meeting, round {r+1}.\n"
            f"Agenda:\n{agenda}\n\n"
            "Shared context so far:\n"
            f"{shared_context}"
        )
        for agent_id, agent in agents.items():
            reply = await agent.arespond(
                user_message=prompt,
                context=None,  # prompt already contains context
            )
            round_responses[agent_id] = reply

        # Update shared context: you can later summarize instead of concatenating.
        shared_context += "\n\n" + "\n".join(
            f"{agents[a_id].display_name}: {txt}"
            for a_id, txt in round_responses.items()
        )
        history.append(round_responses)

    return history


async def run_triads_meeting(
    agents: Dict[str, Agent],
    triad_members: list[str],
    agenda: str,
    project_context: str | None = None,
    rounds: int = 2,
) -> List[dict]:
    """
    Same as internal team meeting, but restricted to triad_members.
    """
    triad_agents = {aid: agents[aid] for aid in triad_members}
    return await run_internal_team_meeting(
        agents=triad_agents,
        agenda=agenda,
        project_context=project_context,
        rounds=rounds,
    )

