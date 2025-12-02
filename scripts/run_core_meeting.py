#!/usr/bin/env python
"""
Run a meeting with a subset of agents to avoid rate limits.
"""

import asyncio

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from MiniLab import load_agents
from MiniLab.orchestrators.meetings import run_user_team_meeting


async def main():
    # Load all agents
    all_agents = load_agents()
    
    # Select a smaller subset to avoid rate limits
    # Start with just the directional core (3 agents)
    selected_agents = {
        "franklin": all_agents["franklin"],
        "watson": all_agents["watson"],
        "carroll": all_agents["carroll"],
    }
    
    print("=" * 60)
    print("MiniLab Meeting - Directional Core Team")
    print("Participants: Franklin, Watson, Carroll")
    print("=" * 60)
    
    user_prompt = input("\nEnter your question for the team:\n> ")

    print(f"\nProcessing your question with {len(selected_agents)} agents...")
    print("This may take ~10-15 seconds...\n")
    
    responses = await run_user_team_meeting(
        selected_agents,
        user_prompt=user_prompt,
        project_context=None,
        sequential=True,
    )

    print("\n" + "=" * 60)
    print("MiniLab Responses")
    print("=" * 60)
    for agent_id, reply in responses.items():
        print(f"\n[{selected_agents[agent_id].display_name} ({agent_id})]")
        print(reply)
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
