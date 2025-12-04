"""
MiniLab Main Entry Point

Provides a menu for selecting different interaction modes:
- Single Analysis: Comprehensive guild-based research workflow
- Regular Meeting: Interactive conversation with PI coordination
- Direct Team Meeting: All agents respond in parallel (future)
"""

import asyncio
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def display_menu():
    """Display the main menu."""
    print("\n" + "=" * 80)
    print("MiniLab - Multi-Agent Research Assistant")
    print("=" * 80)
    print("\nSelect a mode:")
    print("  1. Single Analysis - Team-based research workflow")
    print("       • Comprehensive planning with specialist input")
    print("       • Token budget: 100,000")
    print("       • Outputs: figures.pdf, figure_legends.md, summary.pdf")
    print()
    print("  2. Regular Meeting - NOT AVAILABLE (removed)")
    print("       • Use option 1 only")
    print("       • Continuous conversation mode")
    print("       • Token budget: 300,000 per session")
    print()
    print("  3. Exit")
    print("=" * 80)
    print()


async def run_single_analysis_mode():
    """Launch Single Analysis mode."""
    # Import here to avoid loading everything upfront
    from MiniLab import load_agents
    from MiniLab.orchestrators.single_analysis import run_single_analysis
    from MiniLab.storage.transcript import TranscriptLogger
    
    print("\n" + "=" * 80)
    print("Single Analysis Mode")
    print("=" * 80 + "\n")
    
    print("Enter your research question and path to data in ReadData/:")
    research_question = input("> ").strip()
    
    if not research_question:
        print("No research question provided. Returning to main menu.")
        return
    
    print("Loading agents...")
    
    agents = load_agents()
    print(f"Loaded {len(agents)} agents\n")
    
    # Initialize transcript logger (will be renamed with proper project name)
    transcripts_dir = Path(__file__).parent.parent / "Transcripts"
    logger = TranscriptLogger(transcripts_dir)
    logger.start_session("single_analysis")
    logger.log_user_message(research_question)
    
    print("Starting Single Analysis workflow...\n")
    
    # Run the analysis
    result = await run_single_analysis(
        agents=agents,
        research_question=research_question,
        max_tokens=100_000,
        logger=logger,
    )
    
    # Save transcript
    print("\nSaving transcript...")
    transcript_path = logger.save_transcript()
    print(f"Transcript saved to: {transcript_path}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Check if analysis failed early (quick-fail)
    if result.get('success') == False:
        print(f"❌ Analysis failed at Stage {result.get('failed_at_stage', 'unknown')}")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"Total Tokens Used: ~{result.get('tokens_used', 0):,}")
        if result.get('output_path'):
            print(f"Output Directory: {result['output_path']}")
    else:
        # Normal completion
        print(f"Research Question: {result['research_question']}")
        print(f"Output Directory: {result['output_dir']}")
        print(f"Total Tokens Used: ~{result['token_count']:,}")
    
    print("=" * 80 + "\n")
    
    input("Press Enter to return to main menu...")


async def run_regular_meeting_mode():
    """Launch Regular Meeting mode."""
    from MiniLab import load_agents
    from MiniLab.orchestrators.meetings import run_pi_coordinated_meeting
    from MiniLab.storage.transcript import TranscriptLogger
    
    print("\n" + "=" * 80)
    print("Regular Meeting Mode - PI-Coordinated")
    print("=" * 80)
    print("You'll speak with Franklin, who will coordinate with the team")
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 80 + "\n")
    
    agents = load_agents()
    
    # Initialize transcript logger
    transcripts_dir = Path(__file__).parent.parent / "Transcripts"
    logger = TranscriptLogger(transcripts_dir)
    logger.start_session("pi_coordinated_meeting")
    
    # Accumulate conversation history for context
    conversation_history = []
    total_tokens = 0
    
    while True:
        user_prompt = input("\nYou: ").strip()
        
        if user_prompt.lower() in ["exit", "quit", "q"]:
            print("\nEnding session...")
            # Save transcript
            transcript_path = logger.save_transcript()
            print(f"Transcript saved to: {transcript_path}")
            print("Returning to main menu...")
            return
        
        if not user_prompt:
            continue
        
        # Log user message
        logger.log_user_message(user_prompt)
        
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
            logger=logger,
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


async def main():
    """Main menu loop."""
    while True:
        display_menu()
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            await run_single_analysis_mode()
        elif choice == "2":
            print("\n⚠️  Option 2 is not available. Please use option 1.")
        elif choice == "3":
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print("\n⚠️  Invalid choice. Please enter 1 or 3.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
