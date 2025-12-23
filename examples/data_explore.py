#!/usr/bin/env python3
"""
Data Exploration Example
========================

Demonstrates a data exploration workflow that produces reproducible artifacts
including a Jupyter notebook skeleton.

Usage:
    python examples/data_explore.py --data ReadData/Pluvicto/
    python examples/data_explore.py --dry-run  # No API calls

Outputs:
    outputs/<run_id>/
    ├── provenance.json      # Run metadata and provenance
    ├── summary.md           # Human-readable summary
    ├── runlog.jsonl         # Event stream
    └── analysis.ipynb       # Jupyter notebook skeleton
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_run_id() -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"data_explore_{timestamp}"


def create_output_dir(run_id: str) -> Path:
    """Create output directory structure."""
    base = Path(__file__).parent.parent / "outputs" / run_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_provenance(
    output_dir: Path, 
    run_id: str, 
    data_path: Optional[str], 
    dry_run: bool
) -> Path:
    """Write provenance.json with run metadata."""
    provenance = {
        "run_id": run_id,
        "workflow": "data_exploration",
        "data_path": data_path,
        "dry_run": dry_run,
        "started_at": datetime.now().isoformat(),
        "environment": {
            "python_version": sys.version,
            "cwd": str(Path.cwd()),
        },
        "inputs": {
            "data_path": data_path,
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


def scan_data_files(data_path: Optional[str]) -> list[dict[str, Any]]:
    """Scan data directory for CSV/parquet files."""
    files = []
    
    if not data_path:
        return files
    
    path = Path(data_path)
    if not path.exists():
        return files
    
    for ext in ["*.csv", "*.parquet", "*.tsv"]:
        for f in path.rglob(ext):
            try:
                size = f.stat().st_size
                files.append({
                    "name": f.name,
                    "path": str(f),
                    "size_bytes": size,
                    "extension": f.suffix,
                })
            except Exception:
                pass
    
    return files


def create_notebook(
    output_dir: Path, 
    run_id: str, 
    data_files: list[dict[str, Any]],
    dry_run: bool = False,
) -> Path:
    """Create a Jupyter notebook skeleton for data exploration."""
    
    # Build file loading cells based on discovered files
    file_cells = []
    for i, f in enumerate(data_files[:5]):  # Limit to first 5 files
        file_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"### {f['name']}\n", f"Path: `{f['path']}`"]
        })
        
        if f["extension"] == ".csv":
            file_cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": [
                    f"# Load {f['name']}\n",
                    f"df_{i} = pd.read_csv(r\"{f['path']}\")\n",
                    f"print(f\"Shape: {{df_{i}.shape}}\")\n",
                    f"df_{i}.head()"
                ],
                "outputs": [],
                "execution_count": None,
            })
        elif f["extension"] == ".parquet":
            file_cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": [
                    f"# Load {f['name']}\n",
                    f"df_{i} = pd.read_parquet(r\"{f['path']}\")\n",
                    f"print(f\"Shape: {{df_{i}.shape}}\")\n",
                    f"df_{i}.head()"
                ],
                "outputs": [],
                "execution_count": None,
            })
    
    # Build the notebook structure
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            },
            "minilab": {
                "run_id": run_id,
                "workflow": "data_exploration",
                "dry_run": dry_run,
                "generated_at": datetime.now().isoformat(),
            }
        },
        "cells": [
            # Title cell
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Data Exploration\n",
                    "\n",
                    f"**Run ID:** `{run_id}`\n",
                    f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
                    f"**Mode:** {'DRY RUN' if dry_run else 'LIVE'}\n",
                    "\n",
                    "---\n",
                ]
            },
            # Imports cell
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Standard imports\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Configure display\n",
                    "pd.set_option('display.max_columns', 50)\n",
                    "pd.set_option('display.max_rows', 100)\n",
                    "%matplotlib inline\n",
                    "\n",
                    "print('Libraries loaded successfully')"
                ],
                "outputs": [],
                "execution_count": None,
            },
            # Data loading section
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Data Loading\n",
                    f"\n",
                    f"Found {len(data_files)} data files.\n",
                ]
            },
            *file_cells,
            # QC section
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Basic Quality Control\n",
                    "\n",
                    "Check for missing values, data types, and basic statistics.\n",
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# QC function for any dataframe\n",
                    "def basic_qc(df, name='DataFrame'):\n",
                    "    \"\"\"Run basic QC on a dataframe.\"\"\"\n",
                    "    print(f'=== QC Report for {name} ===')\n",
                    "    print(f'Shape: {df.shape}')\n",
                    "    print(f'\\nData types:')\n",
                    "    print(df.dtypes.value_counts())\n",
                    "    print(f'\\nMissing values:')\n",
                    "    missing = df.isnull().sum()\n",
                    "    missing = missing[missing > 0]\n",
                    "    if len(missing) > 0:\n",
                    "        print(missing.sort_values(ascending=False))\n",
                    "    else:\n",
                    "        print('No missing values')\n",
                    "    print(f'\\nNumeric summary:')\n",
                    "    display(df.describe())\n",
                    "\n",
                    "# Run QC on first dataframe (if loaded)\n",
                    "# basic_qc(df_0, 'df_0')"
                ],
                "outputs": [],
                "execution_count": None,
            },
            # Summary tables section
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Summary Tables\n",
                    "\n",
                    "Generate summary statistics and value counts.\n",
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Example: value counts for categorical columns\n",
                    "# for col in df_0.select_dtypes(include=['object', 'category']).columns:\n",
                    "#     print(f'\\n=== {col} ===')\n",
                    "#     print(df_0[col].value_counts().head(10))"
                ],
                "outputs": [],
                "execution_count": None,
            },
            # Visualization section
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Visualizations\n",
                    "\n",
                    "Generate exploratory plots.\n",
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Example: distribution plot\n",
                    "# fig, ax = plt.subplots(figsize=(10, 6))\n",
                    "# df_0['column_name'].hist(ax=ax, bins=30)\n",
                    "# ax.set_xlabel('Value')\n",
                    "# ax.set_ylabel('Count')\n",
                    "# ax.set_title('Distribution of column_name')\n",
                    "# plt.tight_layout()\n",
                    "# plt.show()"
                ],
                "outputs": [],
                "execution_count": None,
            },
            # Next steps section
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Next Steps\n",
                    "\n",
                    "Based on the exploration above:\n",
                    "\n",
                    "1. **Data Cleaning:** Address any missing values or data quality issues\n",
                    "2. **Feature Engineering:** Create derived features if needed\n",
                    "3. **Deeper Analysis:** Focus on specific hypotheses\n",
                    "4. **Modeling:** Build predictive or explanatory models\n",
                ]
            },
        ]
    }
    
    path = output_dir / "analysis.ipynb"
    path.write_text(json.dumps(notebook, indent=2))
    return path


def run_dry(output_dir: Path, run_id: str, data_path: Optional[str]) -> None:
    """Run in dry mode (no API calls)."""
    print(f"[DRY RUN] Data exploration")
    print(f"[DRY RUN] Data path: {data_path or 'None specified'}")
    print(f"[DRY RUN] Output directory: {output_dir}")
    
    # Log planning event
    write_runlog_event(output_dir, {
        "type": "plan",
        "workflow": "data_exploration",
        "data_path": data_path,
        "dry_run": True,
    })
    
    # Scan for data files
    data_files = scan_data_files(data_path)
    print(f"[DRY RUN] Found {len(data_files)} data files")
    
    # Create notebook skeleton
    notebook_path = create_notebook(output_dir, run_id, data_files, dry_run=True)
    print(f"[DRY RUN] Created notebook: {notebook_path}")
    
    # Write summary
    summary = f"""# Data Exploration Run Summary

**Run ID:** {run_id}
**Data Path:** {data_path or 'None specified'}
**Mode:** DRY RUN
**Completed:** {datetime.now().isoformat()}

## Data Files Found

{len(data_files)} files discovered:

"""
    for f in data_files:
        summary += f"- `{f['name']}` ({f['size_bytes']:,} bytes)\n"
    
    summary += f"""

## Artifacts Produced

- `provenance.json` - Run metadata
- `runlog.jsonl` - Event log
- `summary.md` - This summary
- `analysis.ipynb` - Jupyter notebook skeleton

## Status

✅ Dry run completed successfully.

Open `analysis.ipynb` in Jupyter to begin exploration.
"""
    
    (output_dir / "summary.md").write_text(summary)
    
    # Log completion
    write_runlog_event(output_dir, {
        "type": "complete",
        "status": "success",
        "dry_run": True,
        "data_files_found": len(data_files),
        "artifacts": [
            "provenance.json",
            "summary.md",
            "runlog.jsonl",
            "analysis.ipynb",
        ],
    })
    
    print(f"[DRY RUN] Artifacts written to: {output_dir}")


async def run_real(output_dir: Path, run_id: str, data_path: Optional[str]) -> None:
    """Run actual data exploration (requires API key)."""
    from MiniLab.orchestrators import BohrOrchestrator
    
    print(f"Running data exploration")
    print(f"Data path: {data_path or 'None specified'}")
    print(f"Output directory: {output_dir}")
    
    # Log planning event
    write_runlog_event(output_dir, {
        "type": "plan",
        "workflow": "data_exploration",
        "data_path": data_path,
    })
    
    # Scan for data files
    data_files = scan_data_files(data_path)
    print(f"Found {len(data_files)} data files")
    
    try:
        # Create notebook skeleton first
        notebook_path = create_notebook(output_dir, run_id, data_files, dry_run=False)
        print(f"Created notebook skeleton: {notebook_path}")
        
        # Run orchestrator for deeper analysis if we have data
        if data_files and data_path:
            orchestrator = BohrOrchestrator()
            
            # Start session
            await orchestrator.start_session(
                user_request=f"Perform exploratory data analysis on data in: {data_path}",
                project_name=run_id,
            )
            
            # Run
            results = await orchestrator.run()
            
            summary = results.get("final_summary", "Exploration complete")
        else:
            summary = "No data files found. Notebook skeleton created for manual exploration."
        
        # Write summary
        main_summary = f"""# Data Exploration Run Summary

**Run ID:** {run_id}
**Data Path:** {data_path or 'None specified'}
**Completed:** {datetime.now().isoformat()}

## Summary

{summary}

## Data Files

{len(data_files)} files found.

## Artifacts

- `provenance.json` - Run metadata
- `runlog.jsonl` - Event log
- `analysis.ipynb` - Jupyter notebook

Open `analysis.ipynb` in Jupyter to continue exploration.
"""
        
        (output_dir / "summary.md").write_text(main_summary)
        
        # Log completion
        write_runlog_event(output_dir, {
            "type": "complete",
            "status": "success",
        })
        
        print(f"Data exploration complete. Artifacts in: {output_dir}")
        
    except Exception as e:
        write_runlog_event(output_dir, {
            "type": "error",
            "error": str(e),
        })
        print(f"Error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run a data exploration workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data", "-d",
        default=None,
        help="Path to data directory",
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
    write_provenance(output_dir, run_id, args.data, args.dry_run)
    
    if args.dry_run:
        run_dry(output_dir, run_id, args.data)
    else:
        import asyncio
        asyncio.run(run_real(output_dir, run_id, args.data))


if __name__ == "__main__":
    main()
