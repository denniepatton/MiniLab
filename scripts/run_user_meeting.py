import asyncio

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from MiniLab import load_agents
from MiniLab.orchestrators.meetings import run_pi_coordinated_meeting


async def main():
    agents = load_agents()
    
    print("\n" + "=" * 60)
    print("MiniLab - PI-Coordinated Meeting")
    print("You'll speak with Franklin, who will coordinate with the team")
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 60 + "\n")
    
    # Accumulate conversation history for context
    conversation_history = []
    total_tokens = 0
    
    while True:
        user_prompt = input("\nYou: ").strip()
        
        if user_prompt.lower() in ["exit", "quit", "q"]:
            print("\nEnding session. Goodbye!")
            break
        
        if not user_prompt:
            continue
        
        # Build project context from conversation history
        if conversation_history:
            project_context = "Previous conversation:\n" + "\n".join(
                f"User: {entry['user']}\nFranklin: {entry['response'][:200]}..."
                for entry in conversation_history[-3:]  # Last 3 exchanges for context
            )
        else:
            project_context = None

        result = await run_pi_coordinated_meeting(
            agents,
            pi_agent_id="franklin",
            user_prompt=user_prompt,
            project_context=project_context,
            max_total_tokens=300000,
        )

        print("\n" + "-" * 60)
        print("Franklin:")
        print("-" * 60)
        print(result["pi_response"])
        
        if result.get("tool_results"):
            print("\n" + "-" * 60)
            print("Tool Operations:")
            print("-" * 60)
            for tool_result in result["tool_results"]:
                if tool_result.get("success"):
                    print(f"✓ {tool_result.get('action', 'operation')} completed: {tool_result.get('path', '')}")
                else:
                    print(f"✗ Error: {tool_result.get('error', 'Unknown error')}")
        
        if result["consultations"]:
            print("\n" + "-" * 60)
            print(f"[Consulted with: {', '.join([agents[aid].display_name for aid in result['consultations']])}]")
            print("-" * 60)
        
        total_tokens += result["estimated_tokens"]
        print(f"\n[Session tokens: ~{total_tokens:,} | Last query: ~{result['estimated_tokens']:,}]")
        
        # Store in history
        conversation_history.append({
            "user": user_prompt,
            "response": result["pi_response"],
            "consultations": result["consultations"],
        })


if __name__ == "__main__":
    asyncio.run(main())

