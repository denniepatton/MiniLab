# MiniLab

**Multi-Agent Scientific Research Platform**

MiniLab is an autonomous multi-agent system for scientific data analysis, literature synthesis, and publication-ready document generation. It coordinates specialized AI agents through a directed acyclic graph (DAG) execution model, providing full reproducibility, transparent resource usage, and modular extensibility.

---

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Architecture](#architecture)
- [Agents](#agents)
- [Modules](#modules)
- [Tools](#tools)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Guardrails and Permissions](#guardrails-and-permissions)
- [Token Budget Learning](#token-budget-learning)
- [Development](#development)
- [License](#license)

---

## Overview

MiniLab addresses the challenge of conducting rigorous, reproducible scientific analysis using large language models. Rather than relying on a single general-purpose agent, MiniLab employs a team of nine specialized agents—each with domain expertise, bounded responsibilities, and explicit tool access policies.

The system implements a three-layer architecture:

1. **Tasks**: Project-DAG nodes representing user-meaningful milestones (e.g., literature review, statistical analysis)
2. **Modules**: Reusable procedures that compose tools and coordinate agents
3. **Tools**: Atomic, side-effectful capabilities with typed input/output schemas

This hierarchy enables principled orchestration: the DAG-based orchestrator schedules tasks respecting dependencies, modules encapsulate execution logic, and tools provide guardrail-enforced operations.

### Design Influences

MiniLab integrates patterns from contemporary multi-agent research systems:

- **CellVoyager** (bioRxiv 2025): Autonomous biological analysis workflows
- **VirtualLab** (bioRxiv 2024): Multi-agent collaborative research coordination
- **VS Code Agent Infrastructure**: Explicit orchestration with code-enforced guardrails

---

## Core Concepts

### Terminology

| Term | Definition | Examples |
|:-----|:-----------|:---------|
| **Task** | A project-DAG node representing a user-meaningful milestone | `LITERATURE_REVIEW`, `ANALYSIS_EXECUTION`, `CRITICAL_REVIEW` |
| **Module** | A reusable procedure that composes tools and possibly agents | `ConsultationModule`, `BuildReportModule`, `EvidenceGatheringModule` |
| **Tool** | An atomic, side-effectful capability with typed I/O | `fs.read`, `search.pubmed`, `doc.create_pdf` |

### Execution Model

```
User Request
     │
     ▼
┌─────────────────────┐
│  ConsultationModule │ ──► TaskGraph (DAG)
└─────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│          DAG Orchestrator           │
│  (dependency resolution, budgeting) │
└─────────────────────────────────────┘
           │
     ┌─────┼─────┬─────────┐
     ▼     ▼     ▼         ▼
  Task1  Task2  Task3 ... TaskN
     │     │     │         │
     └─────┴─────┴─────────┘
           │
           ▼
   CriticalReviewModule
           │
           ▼
    Publication Outputs
```

Each task executes one or more modules. Modules invoke tools through the orchestrator, which enforces path policies, logs provenance, and tracks token consumption.

---

## Architecture

### Package Structure

```
MiniLab/
├── agents/                 # AI agent implementations
│   ├── base.py            # Agent class with ReAct loop
│   ├── prompts.py         # Structured prompt construction
│   └── registry.py        # Agent instantiation and configuration
├── config/                 # Configuration management
│   ├── minilab_config.py  # Centralized configuration (SSOT)
│   ├── budget_manager.py  # Token budget tracking
│   └── budget_history.py  # Bayesian usage learning
├── context/               # RAG and context management
│   ├── context_manager.py # Document retrieval
│   └── embeddings.py      # Sentence transformer integration
├── core/                  # Core infrastructure
│   ├── token_account.py   # Real-time token tracking
│   ├── token_learning.py  # JSONL-based learning system
│   ├── task_graph.py      # DAG definition and execution
│   ├── project_structure.py # Standard directory layout
│   └── budget_isolation.py # Budget slices for agents
├── formatting/            # Output formatting
│   └── nature_formatter.py # Nature Journal style formatting
├── infrastructure/        # System utilities
│   ├── features.py        # Feature registry
│   └── errors.py          # Error categorization
├── llm_backends/          # LLM provider adapters
│   ├── anthropic_backend.py # Claude integration
│   └── openai_backend.py   # OpenAI integration
├── modules/               # Modular execution components
│   ├── base.py            # Module abstract base class
│   ├── consultation.py    # User intent → TaskGraph
│   ├── team_discussion.py # Multi-agent feedback
│   ├── evidence_gathering.py # Search + evidence packets
│   ├── build_report.py    # Document assembly
│   └── ... (20+ modules)
├── orchestrator/          # Module coordination
│   ├── orchestrator.py    # Main MiniLab orchestrator
│   └── dag_orchestrator.py # Pure DAG executor
├── security/              # Access control
│   └── path_guard.py      # File operation validation
├── storage/               # Persistence
│   └── transcript.py      # Human-readable logging
├── tools/                 # Agent capabilities
│   ├── base.py            # Tool ABC with prepare/invoke
│   ├── namespaces.py      # Tool namespace registry
│   ├── filesystem.py      # File operations (fs.*)
│   ├── document.py        # DOCX/PDF generation (doc.*)
│   ├── figure.py          # Plot generation (fig.*)
│   ├── permission.py      # User confirmations (permission.*)
│   ├── pubmed.py          # PubMed search (search.pubmed)
│   ├── arxiv.py           # arXiv search (search.arxiv)
│   └── web_search.py      # Web search (search.web)
└── utils/                 # Utilities
    └── timing.py          # Performance metrics
```

---

## Agents

MiniLab coordinates nine specialized agents, organized into functional guilds:

### Synthesis Guild (Cross-Cutting Integration)

| Agent | Named After | Role |
|:------|:------------|:-----|
| **Bohr** | Niels Bohr | Project Manager: orchestration, user communication, delegation |
| **Farber** | Sidney Farber | Clinician Critic: experimental design, medical interpretation |
| **Gould** | Stephen Jay Gould | Librarian Writer: literature review, manuscript preparation |

### Theory Guild (Analytical Foundations)

| Agent | Named After | Role |
|:------|:------------|:-----|
| **Feynman** | Richard Feynman | Theoretician: physics, first-principles reasoning |
| **Shannon** | Claude Shannon | Information Theorist: statistics, signal processing |
| **Greider** | Carol Greider | Molecular Biologist: genetics, cellular mechanisms |

### Implementation Guild (Execution)

| Agent | Named After | Role |
|:------|:------------|:-----|
| **Bayes** | Thomas Bayes | Statistician: Bayesian inference, uncertainty quantification |
| **Hinton** | Geoffrey Hinton | ML Expert: machine learning, neural network implementation |
| **Dayhoff** | Margaret Dayhoff | Bioinformatician: sequence analysis, pipeline design |

### Communication Modes

Agents calibrate response depth based on context:

- **Primary Executor Mode**: Full ownership, tool usage, comprehensive output
- **Consulted Expert Mode**: Focused expertise, concise recommendations (3–8 sentences)

---

## Modules

Modules are reusable procedures that compose tools and coordinate agent execution. Each module has:

- Defined inputs/outputs with validation
- Budget allocation rules
- Checkpointing for resumption
- Provenance logging

### Coordination Modules

| Module | Purpose |
|:-------|:--------|
| `ConsultationModule` | Transform user request into TaskGraph |
| `TeamDiscussionModule` | Multi-agent feedback on plans |
| `OneOnOneModule` | Deep consultation with specific expert |
| `PlanningModule` | Full plan production |
| `CoreInputModule` | Core subgroup answer generation |

### Evidence and Writing Modules

| Module | Purpose |
|:-------|:--------|
| `EvidenceGatheringModule` | Search execution with evidence packet generation |
| `WriteArtifactModule` | Mandatory write gateway (SSOT enforcement) |
| `BuildReportModule` | Assemble narrative documents |
| `LiteratureReviewModule` | Literature synthesis |

### Execution and Verification Modules

| Module | Purpose |
|:-------|:--------|
| `GenerateCodeModule` | Produce executable scripts |
| `AnalysisExecutionModule` | Run analysis pipelines |
| `RunChecksModule` | Tests, linting, smoke checks |
| `SanityCheckDataModule` | Data validation |
| `InterpretStatsModule` | Statistical output interpretation |
| `InterpretPlotModule` | Visual output interpretation |
| `CitationCheckModule` | Citation integrity verification |
| `FormattingCheckModule` | Rubric compliance validation |

### Review Modules

| Module | Purpose |
|:-------|:--------|
| `CriticalReviewModule` | Peer-review-style scrutiny |
| `ConsultExternalExpertModule` | Ephemeral expert consultation |

---

## Tools

Tools are atomic capabilities with typed I/O schemas. The orchestrator validates inputs, enforces permissions, and logs all invocations.

### Tool Namespaces

| Namespace | Purpose | Representative Operations |
|:----------|:--------|:--------------------------|
| `fs.*` | Filesystem operations | `read`, `write`, `list`, `exists`, `mkdir` |
| `search.*` | Literature and web search | `pubmed`, `arxiv`, `web` |
| `doc.*` | Document generation | `create_docx`, `create_pdf`, `markdown_to_pdf` |
| `fig.*` | Figure generation | `create`, `save`, `compose_panels` |
| `render.*` | Rendering and inspection | `pdf_to_images`, `image_info` |
| `permission.*` | User approval requests | `confirm`, `approve` |
| `code.*` | Code operations | `read`, `write`, `apply_diff`, `execute` |
| `env.*` | Environment management | `get_env`, `list_packages` |
| `terminal.*` | Command execution | `run` |
| `citation.*` | Citation management | `add`, `search`, `format` |

### Tool Guardrails

- **Path allowlists**: Read-only access to `ReadData/`, read-write to `Sandbox/{project}/`
- **Permission-gated operations**: Package installation, external downloads, budget changes
- **Provenance logging**: All invocations logged with inputs, outputs, timestamps, and artifact pointers

---

## Project Structure

Each project follows a standard directory layout (per `minilab_outline.md`):

```
Sandbox/{project_name}/
├── artifacts/           # Final deliverables (SSOT authority)
│   ├── plan.md         # Project scope and decisions
│   ├── evidence.md     # Triage notes and citations
│   ├── decisions.md    # Rationale for DAG changes
│   └── acceptance_checks.md
├── planning/
│   └── task_graph.json # DAG definition
├── transcripts/         # Human-readable session logs
├── logs/                # Technical JSON event logs
├── data/
│   ├── raw/            # Unmodified input (read-only)
│   ├── interim/        # Intermediate processing
│   └── processed/      # Analysis-ready data
├── scripts/            # Generated code
├── results/
│   ├── figures/        # Generated visualizations
│   └── tables/         # Data tables
├── reports/            # Final documents (DOCX/PDF)
├── env/                # Environment snapshots
├── eval/               # Evaluation metrics
└── memory/
    ├── notes/          # Persistent agent notes
    ├── sources/        # Source document references
    └── index/          # Vector indices
```

### Single Source of Truth (SSOT)

The `artifacts/` directory is the authoritative record. All substantive writes flow through `WriteArtifactModule` to ensure provenance tracking and consistency.

---

## Installation

### Prerequisites

- Python 3.11 or later
- [micromamba](https://mamba.readthedocs.io/en/latest/installation.html) or conda (recommended)
- Anthropic API key

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/MiniLab.git
cd MiniLab

# Create and activate environment
micromamba create -f environment.yaml
micromamba activate minilab

# Install in development mode
pip install -e .

# Configure API key
export ANTHROPIC_API_KEY="your-key-here"
```

### Verify Installation

```bash
python -c "from MiniLab import __version__; print(f'MiniLab v{__version__}')"
```

---

## Usage

### Command-Line Interface

```bash
# Start interactive session
python scripts/minilab.py

# List existing projects
python scripts/minilab.py --list-projects

# Resume an existing project
python scripts/minilab.py --resume Sandbox/my_project

# Visualize task graph (requires Graphviz)
python scripts/minilab.py --graph Sandbox/my_project

# Enable timing metrics
python scripts/minilab.py --timing
```

### Programmatic API

```python
import asyncio
from MiniLab import run_minilab

async def main():
    results = await run_minilab(
        request="Analyze genomic predictors of treatment response",
        project_name="genomic_analysis",
    )
    print(f"Status: {results.get('status')}")
    print(f"Summary: {results.get('final_summary')}")

asyncio.run(main())
```

### Task Graph Visualization

```python
from MiniLab.core import TaskGraph

# Load existing graph
graph = TaskGraph.load("Sandbox/my_project/planning/task_graph.json")

# Export DOT format for Graphviz
print(graph.to_dot())

# Render PNG (requires Graphviz installation)
graph.render_png("Sandbox/my_project/planning/task_graph.png")
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|:---------|:------------|:--------|
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `MINILAB_SANDBOX` | Project output directory | `./Sandbox` |
| `MINILAB_TIMING` | Enable timing metrics | `0` |

### Configuration Files

| File | Purpose |
|:-----|:--------|
| `config/agents_unified.yaml` | Agent personas, tool access, permissions |
| `config/agent_flexibility.yaml` | Autonomy and adaptation guidance |
| `config/formatting_rubric.md` | Output formatting standards |

---

## Guardrails and Permissions

### Path Policies

- **Read-only**: `ReadData/` (input data)
- **Read-write**: `Sandbox/{project}/` (project outputs)
- **Denied**: All other paths

### Permission-Gated Operations

The following actions require explicit user approval via `permission.request`:

- Installing packages or modifying environments
- Downloading external datasets
- Large compute jobs or long-running operations
- Budget changes beyond agreed policy

### Write-Through Policy

Substantive file writes must flow through `WriteArtifactModule` (or orchestrator-controlled paths) to maintain provenance logging and formatting consistency.

---

## Token Budget Learning

MiniLab maintains a compact, living model of token usage to improve estimation over time.

### Components

1. **Token Model** (`config/token_model.md`): Per-task priors/posteriors with recency-weighted updates
2. **Rolling Log** (`config/token_runs_recent.jsonl`): Recent runs for debugging (bounded length)

### Learning Process

- After each project (or major phase), the orchestrator compacts and refreshes the model
- Bayesian updates weight recent runs more heavily (exponential decay)
- Historical data improves future budget allocations

---

## Development

### Running Tests

```bash
# Activate environment
micromamba activate minilab

# Verify imports
python -c "from MiniLab import run_minilab; print('OK')"

# Test specific components
python -c "from MiniLab.modules import Module; print('modules OK')"
python -c "from MiniLab.tools import DocumentTool; print('tools OK')"
python -c "from MiniLab.core import TaskGraph; print('task_graph OK')"
```

### Code Style

- Type hints for all public APIs
- Docstrings following Google style
- Black formatting with 100-character line length

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use MiniLab in your research, please cite:

```bibtex
@software{minilab2024,
  title = {MiniLab: Multi-Agent Scientific Research Platform},
  author = {Patton, Robert},
  year = {2024},
  url = {https://github.com/yourusername/MiniLab}
}
```
