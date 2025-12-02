from __future__ import annotations

from typing import Dict, List

from MiniLab.agents.base import Agent

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
) -> dict:
    """
    User speaks to PI, who coordinates with other agents as needed.
    
    The PI receives the user's question and decides which agents to consult.
    Other agents discuss among themselves, then PI synthesizes their input
    and responds to the user.
    
    Args:
        agents: All available agents
        pi_agent_id: The PI agent (usually "franklin")
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
1. For ANY file operation (create, save, write, edit files), you MUST delegate to Lee using the DELEGATE format below
2. For scientific/technical questions, consult appropriate team members using CONSULT format
3. For simple questions, answer directly

For file operations, use this EXACT format as your ENTIRE response:

DELEGATE: lee
tool: filesystem
action: write
path: filename.ext
content: [the full content here]
---END

For team consultation (scientific questions only):

CONSULT:
- [agent_id]: [specific question for this agent]
---END

For direct answers, just provide your answer (no special format).

EXAMPLE - User asks "write a hello world script":
DELEGATE: lee
tool: filesystem
action: write
path: hello_world.py
content: #!/usr/bin/env python3
print("Hello, World!")
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
            tool_name = "filesystem"
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
            
            # Get the delegated agent and execute tool
            delegated_agent = other_agents.get(delegate_agent_id)
            if delegated_agent and delegated_agent.has_tool(tool_name):
                show_progress(f"🔧 {pi_agent.display_name} is delegating to {delegated_agent.display_name}...")
                result = await delegated_agent.use_tool(tool_name, **tool_params)
                sys.stdout.write("\r" + " " * 60 + "\r")
                sys.stdout.flush()
                
                tool_results.append(result)
                
                # Generate response based on result
                if result.get("success"):
                    pi_response = f"I delegated the file operation to {delegated_agent.display_name}, and it was successful! The file has been created at `Sandbox/{result.get('path', tool_params.get('path'))}`. You can now use it."
                else:
                    pi_response = f"I asked {delegated_agent.display_name} to handle the file operation, but there was an issue: {result.get('error', 'Unknown error')}"
            else:
                pi_response = f"I tried to delegate to {delegate_agent_id}, but that agent is not available or doesn't have the required tool."
                
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

