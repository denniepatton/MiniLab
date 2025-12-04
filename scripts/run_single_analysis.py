"""
Single Analysis Entry Point

Runs a comprehensive guild-based research analysis workflow.
"""

import asyncio
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from MiniLab import load_agents
from MiniLab.orchestrators.single_analysis import run_single_analysis
from MiniLab.storage.transcript import TranscriptLogger


async def main():
    print("\n" + "=" * 80)
    print("MiniLab - Single Analysis Mode")
    print("=" * 80)
    print("\nThis mode runs a comprehensive 4-stage guild-based research workflow:")
    print("  1. Guild leads create initial analysis plans")
    print("  2. Guild members collaborate and provide feedback")
    print("  3. Bohr synthesizes all inputs into a master plan")
    print("  4. Execute plan with delegation and iteration until complete, revisiting earlier stages as needed")
    print("\nFeatures:")
    print("  • Goal-driven execution (continues until deliverables complete)")
    print("  • Token Budget: 100,000 tokens")
    print("  • Interactive: Press Enter anytime during execution to pause and provide guidance")
    print("  • Each iteration = one round of Bohr coordinating with team to take actions")
    print("=" * 80 + "\n")
    
    # Get research question
    print("Enter your research question:")
    research_question = input("> ").strip()
    
    if not research_question:
        print("No research question provided. Exiting.")
        return
    
    # Output directory - orchestrator will create properly named subdirectory
    output_dir = Path(__file__).parent.parent / "Outputs"
    
    print("\nLoading agents...")
    
    agents = load_agents()
    
    print(f"Loaded {len(agents)} agents")
    print("\nInitializing transcript logger...")
    
    # Initialize transcript logger
    transcripts_dir = Path(__file__).parent.parent / "Transcripts"
    logger = TranscriptLogger(transcripts_dir)
    
    # Note: Session name will be updated after Stage 0 determines project name
    # For now, use a temporary name
    temp_session_name = f"single_analysis_{research_question[:30].replace(' ', '_')}"
    logger.start_session(temp_session_name)
    
    # Log the research question
    logger.log_user_message(research_question)
    
    print("\nStarting Single Analysis workflow...\n")
    
    # Run the analysis
    result = await run_single_analysis(
        agents=agents,
        research_question=research_question,
        output_dir=output_dir,
        max_tokens=100_000,
        logger=logger,
    )
    
    # Update session name with actual project name
    if result.get("project_name"):
        logger.update_session_name(f"single_analysis_{result['project_name']}")
    
    # Save transcript
    print("\nSaving transcript...")
    transcript_path = logger.save_transcript()
    print(f"Transcript saved to: {transcript_path}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Research Question: {result['research_question']}")
    print(f"Output Directory: {result['output_dir']}")
    print(f"Total Tokens Used: ~{result['token_count']:,}")
    print(f"\nStages Completed:")
    print(f"  Stage 1 (Plans): {result['stages_completed']['stage_1_plans']} guilds")
    print(f"  Stage 2 (Collaboration): {result['stages_completed']['stage_2_collaborations']} guilds")
    print(f"  Stage 3 (Synthesis): {'Yes' if result['stages_completed']['stage_3_synthesis'] else 'No'}")
    print(f"  Stage 4 (Execution): {result['stages_completed']['stage_4_iterations']} iterations")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
