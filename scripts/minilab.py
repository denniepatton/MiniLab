#!/usr/bin/env python3
"""
MiniLab - Multi-Agent Scientific Lab Assistant

Command-line interface for running MiniLab analysis sessions.
All interactions flow through Bohr's consultation module.

Terminology (aligned with minilab_outline.md):
- Task: A project-DAG node representing a user-meaningful milestone
- Module: A reusable procedure that composes tools and possibly agents
- Tool: An atomic, side-effectful capability with typed I/O

Usage:
    python scripts/minilab.py
    python scripts/minilab.py --resume Sandbox/my_project
    python scripts/minilab.py --list-projects
    python scripts/minilab.py --timing  # Enable performance timing
    python scripts/minilab.py --graph   # Generate task graph visualization
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
import json
import select

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def _ensure_minilab_environment() -> None:
    """Hard-require execution from the 'minilab' environment.

    MiniLab relies on the micromamba/conda env being consistent so tools like
    Graphviz, scientific Python deps, and analysis libraries are always present.
    """
    env_name = (
        os.getenv("MAMBA_DEFAULT_ENV")
        or os.getenv("CONDA_DEFAULT_ENV")
        or os.getenv("VIRTUAL_ENV")
        or ""
    )
    exe = Path(sys.executable).as_posix()

    # Common, reliable signals
    env_name_ok = str(env_name).strip().lower() == "minilab"
    exe_ok = ("/envs/minilab/" in exe) or exe.endswith("/minilab/bin/python")

    if env_name_ok or exe_ok:
        return

    msg = (
        "MiniLab must be run from the 'minilab' micromamba/conda environment.\n\n"
        f"Detected python: {sys.executable}\n"
        f"Detected env: {env_name or '(none)'}\n\n"
        "Activate and retry:\n"
        "  micromamba activate minilab\n"
        "  python scripts/minilab.py\n"
    )
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MiniLab: Multi-agent scientific lab assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MiniLab uses AI agents coordinated by Bohr to help with scientific analysis.
All sessions begin with a consultation to understand your needs.

Terminology:
  - Task: A project-DAG node representing a user-meaningful milestone
  - Module: A reusable procedure that composes tools and possibly agents
  - Tool: An atomic, side-effectful capability with typed I/O

Examples:
  python scripts/minilab.py                    # Start new session
  python scripts/minilab.py --list-projects    # See existing projects
  python scripts/minilab.py --resume Sandbox/my_project  # Continue project
  python scripts/minilab.py --timing           # Enable performance metrics
  python scripts/minilab.py --graph Sandbox/my_project   # Show task graph
        """,
    )
    
    parser.add_argument(
        "--resume", "-r",
        help="Path to existing project to resume",
    )
    
    parser.add_argument(
        "--list-projects", "-l",
        action="store_true",
        help="List existing projects in Sandbox",
    )
    
    parser.add_argument(
        "--graph", "-g",
        help="Show task graph visualization for a project",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    parser.add_argument(
        "--timing", "-t",
        action="store_true",
        help="Enable performance timing metrics",
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
                # Support both old (completed_workflows) and new (completed_modules) keys
                completed = session.get('completed_modules') or session.get('completed_workflows', [])
                print(f"    Completed: {', '.join(completed) if completed else 'None'}")
                current = session.get('current_module') or session.get('current_workflow')
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


def show_task_graph(project_path: str) -> None:
    """Show task graph visualization for a project."""
    from MiniLab.utils import console
    from MiniLab.core.task_graph import TaskGraph
    
    path = Path(project_path)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / project_path
    
    task_graph_path = path / "planning" / "task_graph.json"
    if not task_graph_path.exists():
        task_graph_path = path / "task_graph.json"  # Legacy location
    
    if not task_graph_path.exists():
        console.warning(f"No task graph found at {task_graph_path}")
        return
    
    try:
        graph = TaskGraph.load(task_graph_path)
        dot = graph.to_dot()
        print("\nTask Graph (DOT format):")
        print("-" * 40)
        print(dot)
        print("-" * 40)
        
        # Try to render PNG
        try:
            png_path = path / "planning" / "task_graph.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            graph.render_png(png_path)
            console.info(f"PNG saved to: {png_path}")
        except Exception as e:
            console.debug(f"Could not render PNG: {e}")
        
        # Show progress
        progress = graph.get_progress()
        print(f"\nProgress: {progress['completed']}/{progress['total']} tasks ({progress['percentage']:.1f}%)")
        
    except Exception as e:
        console.error(f"Error loading task graph: {e}")


async def show_welcome() -> str:
    """
    Display welcome message and get user request.
    
    Returns:
        User's request string (empty if cancelled)
    """
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
    print("  \033[2m💡 Tip: Press Ctrl+C anytime to pause and get options\033[0m")
    print()
    
    # Prompt for input
    try:
        first_line = input("  \033[1;32m▶ Your request:\033[0m ")

        # IMPORTANT: Users often paste multi-line requests.
        # `input()` reads only the first line, leaving the rest buffered in stdin.
        # That buffered text can get consumed by later prompts
        # making it look like the CLI didn't wait for input.
        lines = [first_line.rstrip("\n")]
        try:
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0)
                if not r:
                    break
                extra = sys.stdin.readline()
                if extra == "":
                    break
                lines.append(extra.rstrip("\n"))
        except Exception:
            # If select isn't available or stdin isn't a real file descriptor, fall back.
            pass

        user_input = "\n".join(lines).strip()
        return user_input
        
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Goodbye!")
        return ""


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    from MiniLab.utils import console

    # Hard requirement: always run under the micromamba/conda 'minilab' env.
    _ensure_minilab_environment()
    
    # Enable timing if requested
    if args.timing:
        os.environ["MINILAB_TIMING"] = "1"
    
    if args.verbose:
        console.set_verbose(True)
    
    if args.list_projects:
        list_projects()
        return 0
    
    if args.graph:
        show_task_graph(args.graph)
        return 0
    
    from MiniLab.orchestrator import MiniLabOrchestrator
    from MiniLab.orchestrator.orchestrator import run_minilab
    
    if args.resume:
        # Resume existing project
        project_path = Path(args.resume)
        if not project_path.is_absolute():
            project_path = Path(__file__).parent.parent / args.resume
        
        orchestrator = MiniLabOrchestrator()
        try:
            await orchestrator.resume_session(project_path)
            console.workflow_start("Resuming session")
            results = await orchestrator.run()
            console.workflow_complete("Session", results.get('final_summary', 'Done'))
            return 0
        except Exception as e:
            console.error(f"Error resuming session: {e}")
            return 1
    
    # Show welcome and get user request (no project name yet - Bohr handles that)
    request = await show_welcome()
    if not request:
        return 0  # User cancelled
    
    # Check for special commands
    if request.lower() in ['list', 'projects', 'list projects']:
        list_projects()
        return 0
    
    # Run analysis - Bohr will handle understanding, naming, planning, and approval
    try:
        print()  # Clean spacing before Bohr takes over
        
        results = await run_minilab(request=request)
        
        # Check if session was cancelled or interrupted
        status = results.get('status', 'unknown')
        if status == 'cancelled':
            console.info("Session cancelled.")
            return 0
        elif status == 'interrupted':
            console.info(results.get('message', 'Session interrupted.'))
            return 0
        
        # Only print completion for actual completed work
        print()
        final_summary = results.get('final_summary', '')
        if final_summary:
            console.workflow_complete("Analysis", final_summary)
        else:
            console.info("Session ended.")
        
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
