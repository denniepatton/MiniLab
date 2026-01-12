<p align="center">
  <h1 align="center">🔬 MiniLab</h1>
  <p align="center">
    <strong>Autonomous Multi-Agent Scientific Research Platform</strong>
  </p>
  <p align="center">
    <em>Professional-grade scientific analysis through collaborative AI agents</em>
  </p>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#agents">Agents</a> •
  <a href="#documentation">Documentation</a>
</p>

---

## Overview

MiniLab is a multi-agent AI system designed for professional scientific research workflows. It coordinates a team of nine specialized AI agents to perform literature reviews, data analysis, hypothesis generation, and publication-ready document creation—all with full reproducibility and transparent resource usage.

### Key Capabilities

- **Literature Synthesis**: Deep literature reviews with critical analysis and gap identification
- **Data Analysis**: Exploratory analysis, statistical modeling, and ML pipelines
- **Hypothesis Generation**: Evidence-based brainstorming grounded in peer-reviewed research
- **Publication-Ready Outputs**: Nature Journal-formatted PDFs with proper citations and figures
- **Full Reproducibility**: Complete audit trails, checkpointing, and session resume

### Design Philosophy

MiniLab integrates insights from state-of-the-art multi-agent research:

- **[CellVoyager](https://www.biorxiv.org/content/10.1101/2025.06.03.657517v1)**: Autonomous biological analysis patterns
- **[VirtualLab](https://www.biorxiv.org/content/10.1101/2024.11.11.623004v1)**: Multi-agent collaborative research
- **VS Code Agent Infrastructure**: Hard-coded orchestration with explicit guardrails

---

## Features

### 🤖 Nine Specialized Agents

| Agent | Role | Expertise |
|:------|:-----|:----------|
| **Bohr** | Project Manager | Planning, synthesis, user communication |
| **Gould** | Science Writer | Literature review, citations, documentation |
| **Farber** | Clinical Expert | Experimental design, medical interpretation |
| **Feynman** | Theoretician | Physics, mathematics, first principles |
| **Shannon** | Information Theorist | Statistics, signal processing, feature selection |
| **Greider** | Molecular Biologist | Genetics, cellular mechanisms |
| **Dayhoff** | Bioinformatician | Sequence analysis, computational biology |
| **Hinton** | ML Expert | Machine learning, neural networks, modeling |
| **Bayes** | Statistician | Bayesian inference, uncertainty quantification |

### 📊 DAG-Based Workflow Execution

MiniLab uses a **TaskGraph** (directed acyclic graph) to coordinate complex, multi-step analyses:

```
User Request
     ↓
Consultation → TaskGraph Generated
     ↓
┌────────────────────────────────────────┐
│         DAG Orchestrator               │
│  (respects dependencies, tracks budget)│
└─────────────┬──────────────────────────┘
              │
    ┌─────────┼─────────┬─────────┐
    ↓         ↓         ↓         ↓
Literature  Analysis  Modeling  Writeup
 Review    Execution            Results
    │         │         │         │
    └─────────┴─────────┴─────────┘
              ↓
      Critical Review
              ↓
   Publication-Ready Outputs
```

### 💰 Self-Aware Token Management

- **Real-time tracking**: Per-agent, per-workflow, per-phase granularity
- **Bayesian learning**: Historical usage improves future allocations
- **Budget-aware loops**: Agents adapt iterations based on remaining tokens
- **Transparent reporting**: Users see usage at every checkpoint

### 🔒 Security & Reproducibility

- **PathGuard**: Code-enforced file access control (not prompt-based)
- **Session checkpointing**: Resume interrupted analyses without re-running work
- **Complete audit trails**: JSONL events + human-readable transcripts
- **Atomic operations**: Rollback on failure prevents partial state

### 📝 VS Code-Style Tool Patterns

MiniLab implements modern agent-tool interaction patterns:

- **Two-phase execution**: `prepare()` → `invoke()` for validation before action
- **Typed response streaming**: Structured progress reporting (not just strings)
- **EditSession**: Atomic batched file edits with preview/commit/rollback
- **Tool selection control**: Runtime per-agent tool enablement

---

## Installation

### Prerequisites

- Python 3.11+
- [micromamba](https://mamba.readthedocs.io/en/latest/installation.html) or conda (recommended)
- Anthropic API key (for Claude models)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/MiniLab.git
cd MiniLab

# Create environment (recommended)
micromamba create -f environment.yml
micromamba activate minilab

# Or using pip
pip install -e .

# Configure API keys
cp example.env .env
# Edit .env with your ANTHROPIC_API_KEY
```

### Verify Installation

```bash
python -c "from MiniLab import run_minilab; print('✓ MiniLab installed successfully')"
```

---

## Quickstart

### Basic Usage

```bash
# Start interactive session
python scripts/minilab.py

# With custom token budget
python scripts/minilab.py --budget 500000
```

### Example Session

```
╭────────────────────────────────────────╮
│            🔬 MiniLab                  │
│    Autonomous Research Assistant       │
╰────────────────────────────────────────╯

What would you like to analyze?
> Analyze the Pluvicto clinical trial data to identify genomic predictors of treatment response

[Bohr] Understood. I'll coordinate a comprehensive analysis:
  1. Literature review of Pluvicto response biomarkers
  2. Exploratory data analysis of the genomic features
  3. Statistical modeling for response prediction
  4. Critical review and documentation

Proceed with this plan? [y/n] > y

[Gould] Starting literature review...
...
```

### Programmatic API

```python
import asyncio
from MiniLab import run_minilab

async def main():
    results = await run_minilab(
        request="Analyze genomic predictors of treatment response",
        project_name="pluvicto_analysis",
        budget=500_000,
    )
    
    print(f"Status: {results.status}")
    print(f"Outputs: {results.artifacts}")

asyncio.run(main())
```

---

## Architecture

### Directory Structure

```
MiniLab/
├── MiniLab/                    # Core package
│   ├── agents/                 # AI agent implementations
│   │   ├── base.py            # Agent with ReAct loop
│   │   ├── prompts.py         # Prompt construction
│   │   └── registry.py        # Agent instantiation
│   ├── config/                 # Configuration management
│   │   ├── budget_manager.py  # Token budget tracking
│   │   └── budget_history.py  # Bayesian usage learning
│   ├── context/               # RAG and context management
│   │   ├── context_manager.py # Document retrieval
│   │   └── embeddings.py      # Sentence transformers
│   ├── core/                  # Core infrastructure
│   │   ├── token_account.py   # Real-time token tracking
│   │   ├── task_graph.py      # DAG execution planning
│   │   ├── project_ssot.py    # Single source of truth
│   │   └── budget_isolation.py# Budget slices for agents
│   ├── infrastructure/        # System-level utilities
│   │   ├── features.py        # Feature registry
│   │   └── errors.py          # Error categorization
│   ├── llm_backends/          # LLM provider adapters
│   │   ├── anthropic_backend.py
│   │   └── openai_backend.py
│   ├── orchestrator/          # Workflow coordination
│   │   ├── orchestrator.py    # Main orchestrator
│   │   └── dag_orchestrator.py# Pure DAG executor
│   ├── security/              # Access control
│   │   ├── path_guard.py      # File operation validation
│   │   └── sandbox.py         # Execution isolation
│   ├── storage/               # Persistence
│   │   └── transcript.py      # Human-readable logs
│   ├── tools/                 # Agent capabilities
│   │   ├── base.py            # Tool ABC with prepare/invoke
│   │   ├── code_editor.py     # Code manipulation
│   │   ├── filesystem.py      # File operations
│   │   ├── terminal.py        # Shell execution
│   │   ├── arxiv.py           # ArXiv search
│   │   ├── pubmed.py          # PubMed search
│   │   ├── web_search.py      # Web search
│   │   ├── edit_session.py    # Atomic file edits
│   │   ├── response_stream.py # Typed progress
│   │   └── tool_selector.py   # Tool enablement
│   └── workflows/             # Analysis modules
│       ├── consultation.py    # User intent → TaskGraph
│       ├── literature_review.py
│       ├── execute_analysis.py
│       ├── writeup_results.py
│       └── critical_review.py
├── scripts/
│   └── minilab.py             # CLI entry point
├── Sandbox/                   # Project outputs (gitignored)
├── ReadData/                  # Input datasets
├── minilab_config.yaml        # System configuration
├── ARCHITECTURE.md            # Detailed architecture docs
└── environment.yml            # Conda environment spec
```

### Core Components

#### TokenAccount
Centralized, real-time token tracking with taxonomy-based attribution:
- Per-agent, per-tool, per-operation granularity
- Integrates with BudgetHistory for Bayesian learning
- Provides usage summaries and cost estimates

#### TaskGraph
DAG-based execution planning:
- Generated by Consultation workflow
- Defines tasks, dependencies, and agent assignments
- Orchestrator respects dependencies for execution order

#### PathGuard
Code-enforced security (not prompt-based):
- Validates all file operations before execution
- Agent-specific write permissions
- Audit logging of all access attempts

#### EditSession
VS Code-style atomic file editing:
- Stage multiple edits before committing
- Preview changes with diffs
- Rollback on failure

---

## Configuration

MiniLab is configured via `minilab_config.yaml`:

```yaml
# Token budgets
budget:
  default_budget: 500000
  phase_allocations:
    discovery: 0.05
    planning: 0.15
    execution: 0.60
    synthesis: 0.15
    review: 0.05
  learning: true  # Enable Bayesian adaptation

# Feature requirements
features:
  pdf_generation:
    required: true
  prompt_caching:
    required: true
  rag_retrieval:
    required: false

# Error handling policies
error_handling:
  missing_required_feature: fatal
  network_timeout: retry
  optional_feature_missing: skip
```

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional
MINILAB_SANDBOX=/path/to/sandbox  # Output directory
MINILAB_BUDGET=500000             # Default token budget
MINILAB_LOG_LEVEL=INFO            # Logging verbosity
```

---

## Agents

### Communication Patterns

Agents communicate through structured consultations:

```python
# Agent consulting a colleague
response = await self.consult_colleague(
    colleague_id="hinton",
    question="What ML approach would you recommend for this classification problem?",
    mode="focused",  # quick, focused, or detailed
)
```

### Tool Access

Each agent has specific tool permissions enforced by PathGuard:

| Agent | File Write Access | Special Capabilities |
|:------|:------------------|:---------------------|
| Bohr | All (coordinator) | Project planning |
| Gould | `.md`, `.txt`, `.bib` | Literature synthesis |
| Hinton | `.py`, `.json` | ML modeling |
| Dayhoff | `.py`, `.csv`, `.fasta` | Bioinformatics |
| Bayes | `.py`, `.json` | Statistical analysis |

### Budget Isolation

Colleague consultations receive isolated budget slices:

```python
# Automatic budget isolation in consultations
# Colleague gets proportional allocation, not shared pool
await self.consult_colleague(
    colleague_id="bayes",
    question="Is this correlation statistically significant?",
    budget_isolation=True,  # Default
)
```

---

## Workflows

### Available Workflows

| Workflow | Purpose | Key Outputs |
|:---------|:--------|:------------|
| **Consultation** | Understand user intent | TaskGraph |
| **Literature Review** | Background research | Nature PDF, bibliography |
| **Planning Committee** | Multi-agent deliberation | Detailed analysis plan |
| **Execute Analysis** | Run analysis code | Results, figures |
| **Writeup Results** | Documentation | Reports, summaries |
| **Critical Review** | Quality assurance | Review comments |

### Custom Workflows

Extend `WorkflowModule` to create new workflows:

```python
from MiniLab.workflows import WorkflowModule, WorkflowResult

class MyWorkflow(WorkflowModule):
    name = "my_workflow"
    
    async def execute(self, context: dict) -> WorkflowResult:
        # Your workflow logic
        return WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            summary="Workflow completed successfully",
            artifacts=["output.md"],
        )
```

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Detailed system architecture and design decisions
- **[Examples](examples/)**: Sample analysis scripts

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=MiniLab --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format MiniLab/

# Lint
ruff check MiniLab/

# Type checking
mypy MiniLab/
```

---

## Citation

If you use MiniLab in your research, please cite:

```bibtex
@software{minilab2026,
  title={MiniLab: Autonomous Multi-Agent Scientific Research Platform},
  author={Patton, Robert},
  year={2026},
  url={https://github.com/yourusername/MiniLab},
  note={DAG-driven multi-agent system for scientific analysis}
}
```

---

## License

MiniLab is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

MiniLab builds on ideas from:

- **CellVoyager** (Stanford Zhou Lab) - Autonomous biological analysis
- **VirtualLab** (Stanford) - Multi-agent collaborative research
- **VS Code Agent Infrastructure** - Tool patterns and guardrails
- **Apache Airflow** - DAG-based orchestration patterns

---

<p align="center">
  <strong>MiniLab</strong>: Professional scientific research through collaborative AI agents
</p>
