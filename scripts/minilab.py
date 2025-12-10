#!/usr/bin/env python3
"""
MiniLab - Multi-Agent Scientific Lab Assistant

Command-line interface for running MiniLab analysis sessions.

Usage:
    python scripts/minilab.py "Analyze the genomic data"
    python scripts/minilab.py --project my_project --workflow start_project
    python scripts/minilab.py --resume Sandbox/my_project
    python scripts/minilab.py --interactive
"""

import argparse
import asyncio
import sys
from pathlib import Path
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MiniLab: Multi-agent scientific lab assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a new analysis
  python scripts/minilab.py "Analyze the Pluvicto genomic data"
  
  # Specify workflow explicitly
  python scripts/minilab.py "What is the state of the art in proteomics?" --workflow literature_review
  
  # Resume an existing project
  python scripts/minilab.py --resume Sandbox/pluvicto_analysis_20241215
  
  # Interactive mode
  python scripts/minilab.py --interactive
  
Major Workflows:
  brainstorming      - Explore ideas and approaches
  literature_review  - Background research
  start_project      - Full analysis pipeline (default)
  work_on_existing   - Continue existing project
  explore_dataset    - Data exploration focus

Mini-Workflow Modules (used within major workflows):
  CONSULTATION       - User discussion and requirement gathering
  LITERATURE REVIEW  - Background research with PubMed/arXiv
  PLANNING COMMITTEE - Multi-agent deliberation on approach
  EXECUTE ANALYSIS   - Dayhoff->Hinton->Bayes implementation loop
  WRITE-UP RESULTS   - Documentation and reporting
  CRITICAL REVIEW    - Quality assessment and recommendations
        """,
    )
    
    parser.add_argument(
        "request",
        nargs="?",
        help="Analysis request or question",
    )
    
    parser.add_argument(
        "--project", "-p",
        help="Project name (auto-generated if not specified)",
    )
    
    parser.add_argument(
        "--workflow", "-w",
        choices=["brainstorming", "literature_review", "start_project", "work_on_existing", "explore_dataset"],
        help="Workflow to run (auto-detected if not specified)",
    )
    
    parser.add_argument(
        "--resume", "-r",
        help="Path to existing project to resume",
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode with prompts",
    )
    
    parser.add_argument(
        "--list-projects", "-l",
        action="store_true",
        help="List existing projects in Sandbox",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def list_projects() -> None:
    """List existing projects in Sandbox."""
    from MiniLab.utils import console
    
    sandbox = Path(__file__).parent.parent / "Sandbox"
    
    if not sandbox.exists():
        console.warning("No Sandbox directory found.")
        return
    
    # Find projects (have session.json)
    projects = [d for d in sandbox.iterdir() if d.is_dir() and (d / "session.json").exists()]
    
    # Find other directories
    other_dirs = [d for d in sandbox.iterdir() if d.is_dir() and not (d / "session.json").exists()]
    
    console.header("MiniLab Projects")
    
    if projects:
        print("\n📁 Projects (can be resumed):")
        console.separator("-", 40)
        
        for project in sorted(projects):
            try:
                with open(project / "session.json") as f:
                    session = json.load(f)
                print(f"\n  {project.name}")
                print(f"    Started: {session.get('started_at', 'Unknown')[:19]}")
                completed = session.get('completed_workflows', [])
                print(f"    Completed: {', '.join(completed) if completed else 'None'}")
                current = session.get('current_workflow')
                if current:
                    print(f"    Current: {current}")
            except Exception:
                print(f"\n  {project.name} (unable to read session)")
    
    if other_dirs:
        print(f"\n📂 Other directories ({len(other_dirs)}):")
        console.separator("-", 40)
        for d in sorted(other_dirs)[:10]:
            print(f"  {d.name}")
        if len(other_dirs) > 10:
            print(f"  ... and {len(other_dirs) - 10} more")
    
    if not projects and not other_dirs:
        console.info("No projects found.")
    
    print()


async def interactive_mode() -> None:
    """Run MiniLab in interactive mode."""
    from MiniLab.utils import console
    from MiniLab.orchestrators import BohrOrchestrator
    from MiniLab.orchestrators.bohr_orchestrator import run_minilab
    
    console.header("MiniLab - Multi-Agent Scientific Lab Assistant")
    print("\nWelcome! I'm Bohr, your orchestrator for scientific analysis.")
    print("I coordinate a team of 9 specialist agents to help you with:")
    print("  • Data analysis and modeling")
    print("  • Literature review and synthesis")
    print("  • Statistical validation")
    print("  • Biological interpretation")
    print("\nCommands: 'quit' | 'help' | 'list' | 'resume <path>'")
    console.separator("─", 60)
    
    orchestrator = BohrOrchestrator()
    
    while True:
        try:
            user_input = console.user_prompt("YOU")
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! Your projects are saved in Sandbox/")
                break
            
            if user_input.lower() == "help":
                print("\nCommands:")
                print("  quit/exit    - End session")
                print("  help         - Show this help")
                print("  list         - List existing projects")
                print("  resume <path> - Resume a project")
                print("\nWorkflows (specify with --workflow or auto-detected):")
                print("  brainstorming     - Explore ideas")
                print("  literature_review - Background research")
                print("  start_project     - Full analysis")
                print("  work_on_existing  - Continue project")
                print("  explore_dataset   - Data exploration")
                print("\nOr just type your analysis request!")
                continue
            
            if user_input.lower() == "list":
                list_projects()
                continue
            
            if user_input.lower().startswith("resume "):
                path = user_input.split(" ", 1)[1]
                project_path = Path(path)
                if not project_path.is_absolute():
                    project_path = Path(__file__).parent.parent / path
                
                try:
                    await orchestrator.resume_session(project_path)
                    console.workflow_start("Resuming session")
                    results = await orchestrator.run()
                    console.agent_message("BOHR", results.get('final_summary', 'Session completed.'))
                except Exception as e:
                    console.error(f"Could not resume: {e}")
                continue
            
            # Regular analysis request
            console.agent_message("BOHR", "Starting analysis session...")
            console.info("I'll first consult with you to understand your needs,")
            print("        then coordinate the appropriate specialists.\n")
            
            results = await run_minilab(request=user_input)
            
            console.agent_message("BOHR", results.get('final_summary', 'Analysis complete.'))
            
            if orchestrator.session:
                console.info(f"Project saved to: {orchestrator.session.project_path}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue with a new request.")
        except Exception as e:
            console.error(str(e))
            import traceback
            traceback.print_exc()


def show_welcome() -> str:
    """Display welcome message and prompt for input."""
    from MiniLab.utils import console
    
    console.header("MiniLab - Multi-Agent Scientific Lab Assistant")
    print()
    print("  Welcome! I'm \033[1;36mBohr\033[0m, your orchestrator for scientific analysis.")
    print("  I coordinate a team of 9 specialist AI agents to help you with research.")
    print()
    # Box using .ljust() for guaranteed alignment across all terminals
    w = 73  # inner width
    print("  ┌" + "─" * w + "┐")
    print("  │" + " What can I help you with?".ljust(w) + "│")
    print("  │" + "".ljust(w) + "│")
    print("  │" + "  • Perform exploratory analysis of data in ReadData/TestData".ljust(w) + "│")
    print("  │" + "  • What is CHIP and how does it relate to cancer therapies?".ljust(w) + "│")
    print("  │" + "  • Review the literature on STEAP-1 targeting ADCs".ljust(w) + "│")
    print("  │" + "  • Help me brainstorm hypotheses relating to ...".ljust(w) + "│")
    print("  │" + "  • Let's iterate on the project in Sandbox/EHS-AI ...".ljust(w) + "│")
    print("  │" + "".ljust(w) + "│")
    print("  │" + " Or, simply describe what you need!".ljust(w) + "│")
    print("  └" + "─" * w + "┘")
    print()
    
    # Prompt for input
    try:
        user_input = input("  \033[1;32m▶ Your request:\033[0m ").strip()
        return user_input
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Goodbye!")
        return ""


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    from MiniLab.utils import console
    
    if args.verbose:
        console.set_verbose(True)
    
    if args.list_projects:
        list_projects()
        return 0
    
    if args.interactive:
        await interactive_mode()
        return 0
    
    from MiniLab.orchestrators import BohrOrchestrator
    from MiniLab.orchestrators.bohr_orchestrator import run_minilab
    
    if args.resume:
        # Resume existing project
        project_path = Path(args.resume)
        if not project_path.is_absolute():
            project_path = Path(__file__).parent.parent / args.resume
        
        orchestrator = BohrOrchestrator()
        try:
            await orchestrator.resume_session(project_path)
            console.workflow_start("Resuming session")
            results = await orchestrator.run()
            console.workflow_complete("Session", results.get('final_summary', 'Done'))
            return 0
        except Exception as e:
            console.error(f"Error resuming session: {e}")
            return 1
    
    # If no request provided, show welcome and prompt for input
    request = args.request
    if not request:
        request = show_welcome()
        if not request:
            return 0  # User cancelled
        
        # Check for special commands
        if request.lower() in ['--help', '-h', 'help']:
            parse_args()  # This will show help and exit
            return 0
        if request.lower() in ['--list-projects', '-l', 'list']:
            list_projects()
            return 0
        if request.lower() in ['--interactive', '-i', 'interactive']:
            await interactive_mode()
            return 0
    
    # Run analysis
    try:
        print()  # Clean spacing before Bohr takes over
        
        results = await run_minilab(
            request=request,
            project_name=args.project,
            workflow=args.workflow,
        )
        
        print()
        console.workflow_complete("Analysis", results.get('final_summary', 'Done'))
        
        return 0
        
    except Exception as e:
        console.error(str(e))
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main() -> None:
    """Main entry point."""
    args = parse_args()
    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
