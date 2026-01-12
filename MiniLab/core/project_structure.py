"""
Project Structure Initialization.

Creates the standard project directory structure as specified in minilab_outline.md:

Sandbox/{project}/
├── artifacts/          # Mandatory intermediate docs (plan.md, evidence.md, decisions.md, interpretation.md)
├── planning/           # Task DAG JSON, DOT, PNG
├── transcripts/        # Full agent transcript
├── logs/               # Python logs, terminal output
├── data/
│   ├── raw/           # Original immutable data
│   ├── interim/       # Intermediate processed data
│   └── processed/     # Final analysis-ready data
├── scripts/            # Generated Python scripts
├── results/
│   ├── figures/       # Generated plots
│   └── tables/        # Generated tables
├── reports/            # Final output documents
├── env/                # Environment files (requirements.txt, env.yaml)
├── eval/               # Critical review outputs
├── memory/
│   ├── notes/         # Agent working notes
│   ├── sources/       # Bibliography and source records
│   └── index/         # Embeddings and search indices
└── cache/              # Temporary cached data
"""

from pathlib import Path
from typing import Optional
import json
from datetime import datetime


# Standard project directory structure
PROJECT_STRUCTURE = {
    "artifacts": {
        "description": "Mandatory intermediate documents",
        "contents": ["plan.md", "evidence.md", "decisions.md", "interpretation.md"],
    },
    "planning": {
        "description": "Task DAG and execution plans",
        "contents": ["task_dag.json", "task_dag.dot", "task_dag.png"],
    },
    "transcripts": {
        "description": "Full agent transcript",
        "contents": ["transcript.md"],
    },
    "logs": {
        "description": "Python logs and terminal output",
        "contents": [],
    },
    "data": {
        "description": "Data files",
        "subdirs": {
            "raw": "Original immutable data",
            "interim": "Intermediate processed data",
            "processed": "Final analysis-ready data",
        },
    },
    "scripts": {
        "description": "Generated Python scripts",
        "contents": [],
    },
    "results": {
        "description": "Analysis results",
        "subdirs": {
            "figures": "Generated plots and visualizations",
            "tables": "Generated data tables",
        },
    },
    "reports": {
        "description": "Final output documents",
        "contents": [],
    },
    "env": {
        "description": "Environment files",
        "contents": ["requirements.txt", "environment.yaml"],
    },
    "eval": {
        "description": "Critical review outputs",
        "contents": ["critical_review.md"],
    },
    "memory": {
        "description": "Agent memory and working storage",
        "subdirs": {
            "notes": "Agent working notes",
            "sources": "Bibliography and source records",
            "index": "Embeddings and search indices",
        },
    },
    "cache": {
        "description": "Temporary cached data",
        "contents": [],
    },
}


def create_project_structure(
    project_path: Path,
    include_gitkeep: bool = True,
    include_readme: bool = True,
) -> dict[str, Path]:
    """
    Create the standard project directory structure.
    
    Args:
        project_path: Base path for the project (e.g., Sandbox/my_project)
        include_gitkeep: Whether to add .gitkeep files to empty directories
        include_readme: Whether to add README.md explaining structure
        
    Returns:
        Dict mapping directory names to their full paths
    """
    created_dirs: dict[str, Path] = {}
    
    project_path = Path(project_path)
    project_path.mkdir(parents=True, exist_ok=True)
    created_dirs["root"] = project_path
    
    def create_dir(parent: Path, name: str, description: str = "") -> Path:
        """Create a directory with optional .gitkeep."""
        dir_path = parent / name
        dir_path.mkdir(parents=True, exist_ok=True)
        
        if include_gitkeep:
            gitkeep = dir_path / ".gitkeep"
            if not gitkeep.exists() and not any(dir_path.iterdir()):
                gitkeep.touch()
        
        return dir_path
    
    # Create top-level directories
    for dir_name, config in PROJECT_STRUCTURE.items():
        dir_path = create_dir(
            project_path,
            dir_name,
            config.get("description", ""),
        )
        created_dirs[dir_name] = dir_path
        
        # Create subdirectories
        subdirs = config.get("subdirs", {})
        for subdir_name, subdir_desc in subdirs.items():
            subdir_path = create_dir(dir_path, subdir_name, subdir_desc)
            created_dirs[f"{dir_name}/{subdir_name}"] = subdir_path
    
    # Create README if requested
    if include_readme:
        _create_structure_readme(project_path)
    
    # Create project_state.json stub
    _create_project_state_stub(project_path)
    
    return created_dirs


def _create_structure_readme(project_path: Path) -> None:
    """Create a README explaining the project structure."""
    readme_content = f"""# {project_path.name}

Created by MiniLab on {datetime.now().strftime('%Y-%m-%d %H:%M')}.

## Directory Structure

```
{project_path.name}/
├── artifacts/          # Mandatory intermediate docs
│   ├── plan.md         # Project plan
│   ├── evidence.md     # Evidence packets
│   ├── decisions.md    # Key decisions
│   └── interpretation.md # Analysis interpretation
├── planning/           # Task DAG JSON, DOT, PNG
├── transcripts/        # Full agent transcript
├── logs/               # Python logs, terminal output
├── data/
│   ├── raw/           # Original immutable data
│   ├── interim/       # Intermediate processed data
│   └── processed/     # Final analysis-ready data
├── scripts/            # Generated Python scripts
├── results/
│   ├── figures/       # Generated plots
│   └── tables/        # Generated tables
├── reports/            # Final output documents
├── env/                # Environment files
├── eval/               # Critical review outputs
├── memory/
│   ├── notes/         # Agent working notes
│   ├── sources/       # Bibliography records
│   └── index/         # Search indices
└── cache/              # Temporary cached data
```

## Key Files

- `artifacts/plan.md`: The project plan generated during consultation
- `planning/task_dag.json`: Task graph in JSON format
- `planning/task_dag.png`: Visual representation of task dependencies
- `transcripts/transcript.md`: Complete conversation transcript
- `reports/`: Final deliverables

## Agents

This project uses MiniLab's multi-agent system:
- **Bohr**: Project leader and coordinator
- **Gould**: Literature review specialist
- **Farber**: Claims discipline and narrative
- **Feynman**: Conceptual clarity
- **Shannon**: Information theory
- **Greider**: Mechanistic biology
- **Dayhoff**: Bioinformatics pipelines
- **Hinton**: Computational methods
- **Bayes**: Statistical analysis
"""
    
    readme_path = project_path / "README.md"
    if not readme_path.exists():
        readme_path.write_text(readme_content)


def _create_project_state_stub(project_path: Path) -> None:
    """Create initial project_state.json."""
    state = {
        "project_name": project_path.name,
        "created_at": datetime.now().isoformat(),
        "status": "planning",
        "version": "1.0",
        "structure_version": "minilab_outline_v1",
    }
    
    state_path = project_path / "project_state.json"
    if not state_path.exists():
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)


def get_project_paths(project_path: Path) -> dict[str, Path]:
    """
    Get standard paths for a project.
    
    Args:
        project_path: Base project path
        
    Returns:
        Dict with named paths for common locations
    """
    p = Path(project_path)
    return {
        "root": p,
        "artifacts": p / "artifacts",
        "plan": p / "artifacts" / "plan.md",
        "evidence": p / "artifacts" / "evidence.md",
        "decisions": p / "artifacts" / "decisions.md",
        "interpretation": p / "artifacts" / "interpretation.md",
        "planning": p / "planning",
        "task_dag_json": p / "planning" / "task_dag.json",
        "task_dag_dot": p / "planning" / "task_dag.dot",
        "task_dag_png": p / "planning" / "task_dag.png",
        "transcripts": p / "transcripts",
        "transcript": p / "transcripts" / "transcript.md",
        "logs": p / "logs",
        "data_raw": p / "data" / "raw",
        "data_interim": p / "data" / "interim",
        "data_processed": p / "data" / "processed",
        "scripts": p / "scripts",
        "figures": p / "results" / "figures",
        "tables": p / "results" / "tables",
        "reports": p / "reports",
        "env": p / "env",
        "eval": p / "eval",
        "critical_review": p / "eval" / "critical_review.md",
        "memory_notes": p / "memory" / "notes",
        "memory_sources": p / "memory" / "sources",
        "memory_index": p / "memory" / "index",
        "cache": p / "cache",
        "state": p / "project_state.json",
    }


def validate_project_structure(project_path: Path) -> tuple[bool, list[str]]:
    """
    Validate that a project has the expected structure.
    
    Args:
        project_path: Path to project
        
    Returns:
        Tuple of (is_valid, list of missing items)
    """
    missing: list[str] = []
    
    required_dirs = [
        "artifacts",
        "planning",
        "transcripts",
        "logs",
        "data",
        "data/raw",
        "data/interim",
        "data/processed",
        "scripts",
        "results",
        "results/figures",
        "results/tables",
        "reports",
        "env",
        "eval",
        "memory",
        "memory/notes",
        "memory/sources",
        "memory/index",
        "cache",
    ]
    
    p = Path(project_path)
    
    for dir_name in required_dirs:
        if not (p / dir_name).is_dir():
            missing.append(dir_name)
    
    return len(missing) == 0, missing


def ensure_project_structure(project_path: Path) -> None:
    """
    Ensure project has complete structure, creating missing dirs.
    
    Args:
        project_path: Path to project
    """
    is_valid, missing = validate_project_structure(project_path)
    
    if not is_valid:
        # Create missing directories
        p = Path(project_path)
        for dir_name in missing:
            (p / dir_name).mkdir(parents=True, exist_ok=True)
