"""
Single Analysis v2 Entry Point

Runs the comprehensive 7-stage MiniLab research analysis workflow.
"""

import asyncio
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from MiniLab.agents.registry import load_agents
from MiniLab.orchestrators.single_analysis_v2 import run_single_analysis, _ask_user_permission
from MiniLab.storage.transcript import TranscriptLogger


async def main():
    print("\n" + "=" * 80)
    print("MiniLab - Single Analysis v2")
    print("=" * 80)
    print("\nThis workflow produces 3 primary outputs:")
    print("  • XXX_figures.pdf - Nature-style 4-6 panel figure")
    print("  • XXX_legends.md - Figure legends")
    print("  • XXX_summary.md - Mini-paper (Intro, Discussion, Methods, References)")
    print("\nWorkflow Stages:")
    print("  0. Confirm files and project naming")
    print("  1. Build project structure and summarize data")
    print("  2. Plan analysis (Synthesis → Theory → Implementation cores)")
    print("  3. Exploratory execution (if needed)")
    print("  4. Complete execution")
    print("  5. Write-up (legends, summary)")
    print("  6. Critical review")
    print("\nFeatures:")
    print("  • User checkpoints at each stage")
    print("  • Parallel agent consultation where beneficial")
    print("  • Cross-agent collaboration (any agent can ask others)")
    print("  • Package install requires your approval")
    print("  • ReadData/ is READ-ONLY, Sandbox/ is READ-WRITE")
    print("=" * 80 + "\n")
    
    # Get research question
    print("Enter your research question (include the ReadData path):")
    print("Example: Analyze the files in ReadData/Pluvicto/ to study treatment response")
    research_question = input("> ").strip()
    
    if not research_question:
        print("No research question provided. Exiting.")
        return
    
    print("\nLoading agents...")
    
    # Load agents with permission callback for package installs
    agents = load_agents(permission_callback=_ask_user_permission)
    
    print(f"Loaded {len(agents)} agents:")
    for agent_id, agent in agents.items():
        print(f"  • {agent.display_name} ({agent.role})")
    
    print("\nInitializing transcript logger...")
    
    # Initialize transcript logger
    transcripts_dir = Path(__file__).parent.parent / "Transcripts"
    logger = TranscriptLogger(transcripts_dir)
    
    # Temporary session name (updated after project naming)
    temp_name = research_question[:40].replace(' ', '_').replace('/', '_')
    logger.start_session(f"analysis_{temp_name}")
    
    # Log the research question
    logger.log_user_message(research_question)
    
    print("\nStarting Single Analysis workflow...\n")
    print("=" * 80)
    
    try:
        # Run the analysis
        result = await run_single_analysis(
            agents=agents,
            research_question=research_question,
            max_tokens=2_000_000,
            logger=logger,
        )
        
        # Update session name with actual project name
        if result.get("project_name"):
            logger.update_session_name(f"analysis_{result['project_name']}")
        
        # Save transcript
        print("\nSaving transcript...")
        transcript_path = logger.save_transcript()
        print(f"Transcript saved to: {transcript_path}")
        
        # Display summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        if result.get("success"):
            print(f"\n✓ Project: {result.get('project_name')}")
            print(f"✓ Location: {result.get('project_path')}")
            print(f"✓ Tokens used: ~{result.get('tokens_used', 0):,}")
            
            outputs = result.get("outputs", {})
            print("\nPrimary Outputs:")
            for name, path in outputs.items():
                exists = Path(path).exists()
                status = "✓" if exists else "⚠"
                print(f"  {status} {name}: {path}")
        else:
            print(f"\n✗ Workflow failed: {result.get('error', 'Unknown error')}")
        
        print("=" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        transcript_path = logger.save_transcript()
        print(f"Partial transcript saved to: {transcript_path}")
    
    except Exception as e:
        print(f"\n\nError during workflow: {e}")
        import traceback
        traceback.print_exc()
        transcript_path = logger.save_transcript()
        print(f"Transcript saved to: {transcript_path}")


if __name__ == "__main__":
    asyncio.run(main())
