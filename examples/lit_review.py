#!/usr/bin/env python3
"""
Literature Review Example
=========================

Demonstrates a literature review workflow that produces reproducible artifacts.

Usage:
    python examples/lit_review.py --goal "Review CHIP mutations in cancer"
    python examples/lit_review.py --dry-run  # No API calls

Outputs:
    outputs/<run_id>/
    ├── provenance.json      # Run metadata and provenance
    ├── summary.md           # Human-readable summary
    ├── runlog.jsonl         # Event stream
    └── literature/
        ├── references.md    # Bibliography
        └── summary.md       # Literature synthesis
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_run_id() -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"lit_review_{timestamp}"


def create_output_dir(run_id: str) -> Path:
    """Create output directory structure."""
    base = Path(__file__).parent.parent / "outputs" / run_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "literature").mkdir(exist_ok=True)
    return base


def write_provenance(output_dir: Path, run_id: str, goal: str, dry_run: bool) -> Path:
    """Write provenance.json with run metadata."""
    provenance = {
        "run_id": run_id,
        "workflow": "literature_review",
        "goal": goal,
        "dry_run": dry_run,
        "started_at": datetime.now().isoformat(),
        "environment": {
            "python_version": sys.version,
            "cwd": str(Path.cwd()),
        },
        "inputs": {
            "goal": goal,
        },
        "tool_calls": [],
        "artifacts": [],
    }
    
    path = output_dir / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2))
    return path


def write_runlog_event(output_dir: Path, event: dict[str, Any]) -> None:
    """Append event to runlog.jsonl."""
    event["timestamp"] = datetime.now().isoformat()
    path = output_dir / "runlog.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


def run_dry(output_dir: Path, run_id: str, goal: str) -> None:
    """Run in dry mode (no API calls)."""
    print(f"[DRY RUN] Literature review for: {goal}")
    print(f"[DRY RUN] Output directory: {output_dir}")
    
    # Log planning event
    write_runlog_event(output_dir, {
        "type": "plan",
        "workflow": "literature_review",
        "goal": goal,
        "dry_run": True,
    })
    
    # Generate placeholder artifacts
    summary_content = f"""# Literature Review Summary

**Goal:** {goal}

**Run ID:** {run_id}

**Status:** DRY RUN - No actual searches performed

## Overview

This is a dry run output. In a real run, this would contain:

1. **Search Strategy** - PubMed and arXiv queries
2. **Key Papers** - Relevant publications with citations
3. **Synthesis** - Key themes and findings
4. **Gaps** - Areas for further research

## Next Steps

Run without `--dry-run` to perform actual literature search.
"""
    
    (output_dir / "literature" / "summary.md").write_text(summary_content)
    
    references_content = f"""# References

**Goal:** {goal}

**Run ID:** {run_id}

## Bibliography

*No references collected (dry run)*

In a real run, this would contain:
- PubMed citations with PMIDs
- arXiv preprints with IDs
- Full citation metadata
"""
    
    (output_dir / "literature" / "references.md").write_text(references_content)
    
    # Write main summary
    main_summary = f"""# Literature Review Run Summary

**Run ID:** {run_id}
**Goal:** {goal}
**Mode:** DRY RUN
**Completed:** {datetime.now().isoformat()}

## Artifacts Produced

- `provenance.json` - Run metadata
- `runlog.jsonl` - Event log
- `literature/summary.md` - Literature synthesis
- `literature/references.md` - Bibliography

## Status

✅ Dry run completed successfully.

Run without `--dry-run` flag to perform actual literature search.
"""
    
    (output_dir / "summary.md").write_text(main_summary)
    
    # Log completion
    write_runlog_event(output_dir, {
        "type": "complete",
        "status": "success",
        "dry_run": True,
        "artifacts": [
            "provenance.json",
            "summary.md",
            "runlog.jsonl",
            "literature/summary.md",
            "literature/references.md",
        ],
    })
    
    print(f"[DRY RUN] Artifacts written to: {output_dir}")


async def run_real(output_dir: Path, run_id: str, goal: str) -> None:
    """Run actual literature review (requires API key)."""
    from MiniLab.orchestrators import BohrOrchestrator
    
    print(f"Running literature review for: {goal}")
    print(f"Output directory: {output_dir}")
    
    # Log planning event
    write_runlog_event(output_dir, {
        "type": "plan",
        "workflow": "literature_review",
        "goal": goal,
    })
    
    try:
        orchestrator = BohrOrchestrator()
        
        # Start session
        await orchestrator.start_session(
            user_request=f"Perform a literature review on: {goal}",
            project_name=run_id,
        )
        
        # Run with literature focus
        results = await orchestrator.run()
        
        # Copy artifacts to output directory
        if orchestrator.session:
            project_path = orchestrator.session.project_path
            
            # Copy literature files if they exist
            lit_src = project_path / "literature"
            if lit_src.exists():
                for f in lit_src.iterdir():
                    if f.is_file():
                        (output_dir / "literature" / f.name).write_text(f.read_text())
        
        # Write summary
        summary = results.get("final_summary", "No summary generated")
        (output_dir / "summary.md").write_text(f"""# Literature Review Run Summary

**Run ID:** {run_id}
**Goal:** {goal}
**Completed:** {datetime.now().isoformat()}

## Summary

{summary}

## Artifacts

See `literature/` directory for full outputs.
""")
        
        # Log completion
        write_runlog_event(output_dir, {
            "type": "complete",
            "status": "success",
        })
        
        print(f"Literature review complete. Artifacts in: {output_dir}")
        
    except Exception as e:
        write_runlog_event(output_dir, {
            "type": "error",
            "error": str(e),
        })
        print(f"Error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run a literature review workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--goal", "-g",
        default="Review recent advances in CHIP mutations and cancer therapy",
        help="Literature review goal/topic",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Dry run mode (no API calls)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override run ID (auto-generated if not specified)",
    )
    
    args = parser.parse_args()
    
    # Check for dry run via environment
    if os.environ.get("MINILAB_DRY_RUN") == "1":
        args.dry_run = True
    
    # Generate run ID and create output directory
    run_id = args.run_id or generate_run_id()
    output_dir = create_output_dir(run_id)
    
    # Write initial provenance
    write_provenance(output_dir, run_id, args.goal, args.dry_run)
    
    if args.dry_run:
        run_dry(output_dir, run_id, args.goal)
    else:
        import asyncio
        asyncio.run(run_real(output_dir, run_id, args.goal))


if __name__ == "__main__":
    main()
