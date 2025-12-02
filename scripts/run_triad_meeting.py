#!/usr/bin/env python
"""
Triad Meeting Script

Run a focused discussion with one of the specialized triads:
- Directional Core: Franklin, Watson, Carroll
- Theory Modeling: Feynman, Shannon, Greider
- Implementation/Data: Bayes, Lee, Dayhoff
"""

import asyncio
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from MiniLab import load_agents
from MiniLab.agents.registry import load_triads
from MiniLab.orchestrators.meetings import run_triads_meeting
from MiniLab.storage.state_store import StateStore


async def main():
    # Load agents and triads
    agents = load_agents()
    triads = load_triads()
    
    print("\n" + "=" * 60)
    print("MiniLab Triad Meeting")
    print("=" * 60)
    print("\nAvailable triads:")
    for i, (triad_name, members) in enumerate(triads.items(), 1):
        member_names = [agents[m].display_name for m in members["members"]]
        print(f"{i}. {triad_name}: {', '.join(member_names)}")
    
    # Select triad
    choice = input("\nSelect triad number: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(triads)):
        print("Invalid selection")
        return
    
    triad_name = list(triads.keys())[int(choice) - 1]
    triad_members = triads[triad_name]["members"]
    
    print(f"\n✓ Selected: {triad_name}")
    print(f"Participants: {', '.join(agents[m].display_name for m in triad_members)}")
    
    # Get agenda
    print("\nWhat should the triad discuss?")
    agenda = input("> ").strip()
    
    if not agenda:
        print("No agenda provided")
        return
    
    # Optional project context
    use_project = input("\nLoad project context? (y/n): ").strip().lower()
    project_context = None
    
    if use_project == 'y':
        store = StateStore()
        projects = store.list_projects()
        
        if projects:
            print("\nAvailable projects:")
            for i, proj in enumerate(projects, 1):
                print(f"{i}. {proj['name']} (ID: {proj['project_id']})")
            
            proj_choice = input("Select project number: ").strip()
            if proj_choice.isdigit():
                idx = int(proj_choice) - 1
                if 0 <= idx < len(projects):
                    project_id = projects[idx]["project_id"]
                    project = store.load_project(project_id)
                    
                    # Build context summary
                    context_parts = [
                        f"Project: {project.name}",
                        f"Description: {project.description}",
                        f"Citations: {len(project.citations)}",
                        f"Recent ideas: {len(project.ideas)}",
                    ]
                    
                    if project.citations:
                        context_parts.append("\nKey citations:")
                        for key, cit in list(project.citations.items())[:5]:
                            context_parts.append(f"  - [{cit.key}] {cit.title}")
                    
                    if project.ideas:
                        context_parts.append("\nRecent ideas:")
                        for idea in project.ideas[-3:]:
                            context_parts.append(f"  - {idea['title']}")
                    
                    project_context = "\n".join(context_parts)
    
    # Number of rounds
    rounds_input = input("\nNumber of discussion rounds (default 2): ").strip()
    rounds = int(rounds_input) if rounds_input.isdigit() else 2
    
    print(f"\n🚀 Starting {rounds}-round discussion...")
    print("=" * 60)
    
    # Run the meeting
    try:
        history = await run_triads_meeting(
            agents=agents,
            triad_members=triad_members,
            agenda=agenda,
            project_context=project_context,
            rounds=rounds,
        )
        
        # Display results
        for round_num, round_responses in enumerate(history, 1):
            print(f"\n{'=' * 60}")
            print(f"ROUND {round_num}")
            print('=' * 60)
            
            for agent_id, response in round_responses.items():
                print(f"\n[{agents[agent_id].display_name}]")
                print(response)
                print("-" * 60)
        
        # Save option
        save = input("\nSave transcript? (y/n): ").strip().lower()
        if save == 'y':
            import datetime
            filename = f"triad_{triad_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(filename, 'w') as f:
                f.write(f"Triad Meeting: {triad_name}\n")
                f.write(f"Agenda: {agenda}\n")
                f.write("=" * 60 + "\n\n")
                
                for round_num, round_responses in enumerate(history, 1):
                    f.write(f"\n{'=' * 60}\n")
                    f.write(f"ROUND {round_num}\n")
                    f.write('=' * 60 + "\n")
                    
                    for agent_id, response in round_responses.items():
                        f.write(f"\n[{agents[agent_id].display_name}]\n")
                        f.write(response + "\n")
                        f.write("-" * 60 + "\n")
            
            print(f"✓ Saved to {filename}")
        
    except Exception as e:
        print(f"\nError during meeting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
