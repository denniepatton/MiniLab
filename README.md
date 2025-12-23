# MiniLab

**MiniLab** is a multi-agent scientific research assistant that combines autonomous analysis capabilities with collaborative agent workflows. Inspired by [CellVoyager](https://www.biorxiv.org/content/10.1101/2025.06.03.657517v1) for autonomous biological data analysis, [VirtualLab](https://www.nature.com/articles/s41586-025-09442-9) for multi-agent scientific collaboration, and modern agentic coding paradigms, MiniLab provides an integrated environment for conducting state-of-the-art research workflows.

## Overview

MiniLab creates a team of specialized AI agents that work together to assist researchers with:

- **Literature Synthesis**: Comprehensive searches across PubMed and arXiv with critical assessment
- **Data Exploration**: Autonomous exploration and characterization of datasets
- **Hypothesis Generation**: Multi-agent deliberation to develop and refine research questions
- **Analysis Execution**: End-to-end implementation from planning through statistical validation
- **Documentation**: Automated report generation with proper citations and figure legends

The system employs a ReAct-style execution loop where agents autonomously use tools, consult colleagues, and iterate toward solutions—while maintaining human oversight at key decision points.

## Key Features

### Multi-Agent Architecture
- **Nine specialized agents** organized into three guilds (Synthesis, Theory, Implementation)
- **Cross-agent consultation** with visible dialogue for transparency
- **Dynamic delegation** based on task requirements and agent expertise

### Autonomous Execution
- **ReAct-style loops** enabling agents to reason, act, and observe iteratively
- **Tool integration** for file operations, code execution, web search, and literature access
- **Checkpoint/resume capability** for long-running analyses

### Flexible Workflow System
- **Six composable mini-workflows** that can be combined into larger pipelines
- **Token budget management** with dynamic allocation across workflow phases
- **Tiered modes** (Quick vs. Comprehensive) adapting to resource constraints

### Security and Safety
- **PathGuard access control** enforcing read-only data directories and sandboxed outputs
- **Agent-specific permissions** limiting tool access by role
- **Audit logging** for all file operations

### User Experience
- **Narrative-style communication** from the orchestrator (Bohr)
- **Visible agent consultations** showing inter-agent dialogue
- **Graceful interruption** with progress saving via Ctrl+C
- **Comprehensive transcripts** capturing all session activity

## Architecture

```
MiniLab/
├── agents/                    # Agent system
│   ├── base.py               # Agent with ReAct loop, colleague consultation
│   ├── prompts.py            # Structured 5-part prompting schema
│   └── registry.py           # Agent creation and colleague relationships
├── config/
│   ├── agents.yaml           # Agent personas and tool assignments
│   └── loader.py             # YAML configuration loader
├── context/                   # RAG-based context management
│   ├── context_manager.py    # Context orchestration with token budgets
│   ├── embeddings.py         # Sentence-transformers integration
│   ├── vector_store.py       # FAISS vector store for retrieval
│   └── state_objects.py      # ProjectState, TaskState definitions
├── llm_backends/             # LLM integrations
│   ├── anthropic_backend.py  # Claude API with prompt caching
│   └── openai_backend.py     # OpenAI API support
├── orchestrators/
│   └── bohr_orchestrator.py  # Workflow coordination, session management
├── security/
│   └── path_guard.py         # File access control and audit logging
├── storage/
│   ├── state_store.py        # Persistent state management
│   └── transcript.py         # Session transcript logging
├── tools/                    # Typed tool system
│   ├── base.py               # Tool, ToolInput, ToolOutput base classes
│   ├── filesystem.py         # File read/write/list operations
│   ├── code_editor.py        # Code creation and editing
│   ├── terminal.py           # Shell command execution
│   ├── environment.py        # Package management
│   ├── web_search.py         # Tavily web search integration
│   ├── pubmed.py             # NCBI E-utilities for literature
│   ├── arxiv.py              # arXiv paper search
│   ├── citation.py           # Bibliography management
│   ├── user_input.py         # User interaction tool
│   └── tool_factory.py       # Agent-specific tool instantiation
├── utils/
│   ├── __init__.py           # Console formatting, spinners
│   └── timing.py             # Performance timing utilities
└── workflows/                # Modular workflow components
    ├── base.py               # WorkflowModule abstract base class
    ├── consultation.py       # User goal clarification (Bohr)
    ├── literature_review.py  # Background research (Gould)
    ├── planning_committee.py # Multi-agent deliberation
    ├── execute_analysis.py   # Implementation loop (Dayhoff→Hinton→Bayes)
    ├── writeup_results.py    # Documentation (Gould)
    └── critical_review.py    # Quality assessment (Farber)
```

## Agent Team

All agents use Claude Sonnet 4 via the Anthropic API with structured role-specific prompting:

| Agent | Guild | Role | Specialty |
|-------|-------|------|-----------|
| **Bohr** | Synthesis | Project Manager | Orchestration, user interaction, workflow selection |
| **Gould** | Synthesis | Librarian Writer | Literature review, citations, scientific writing |
| **Farber** | Synthesis | Clinician Critic | Critical review, clinical relevance, quality control |
| **Feynman** | Theory | Curious Physicist | Creative problem-solving, analogies, naive questions |
| **Shannon** | Theory | Information Theorist | Experimental design, methodology, analytical rigor |
| **Greider** | Theory | Molecular Biologist | Biological mechanisms, pathway interpretation |
| **Dayhoff** | Implementation | Bioinformatician | Workflow design, data pipelines, execution planning |
| **Hinton** | Implementation | CS Engineer | Code development, debugging, script execution |
| **Bayes** | Implementation | Statistician | Statistical validation, uncertainty quantification |

## Installation

### Prerequisites

- macOS or Linux
- Python 3.11 or higher
- micromamba, conda, or mamba for environment management
- Anthropic API key (required)
- Tavily API key (optional, for web search)

### Setup

```bash
# Clone repository
git clone https://github.com/denniepatton/MiniLab.git
cd MiniLab

# Create environment
micromamba env create -f environment.yml
micromamba activate minilab

# Install in development mode
pip install -e .

# Configure environment variables
cp example.env .env
# Edit .env with your API keys

# Verify installation
python -c "from MiniLab import run_minilab; print('MiniLab ready')"
```

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional - Web Search
TAVILY_API_KEY=tvly-...

# Optional - PubMed (higher rate limits)
NCBI_EMAIL=your@email.com
NCBI_API_KEY=...

# Optional - Timing/Debug
MINILAB_TIMING=1  # Enable timing reports
```

## Usage

### Command Line Interface

```bash
# Start a new analysis project
python scripts/minilab.py "Analyze the Pluvicto genomic data for treatment response predictors"

# Quick literature review
python scripts/minilab.py "What is the state of the art in cfDNA methylation analysis?"

# Resume an existing project
python scripts/minilab.py --resume Sandbox/pluvicto_analysis

# List existing projects
python scripts/minilab.py --list-projects
```

### Python API

```python
import asyncio
from MiniLab import run_minilab

async def main():
    results = await run_minilab(
        request="Analyze genomic features predictive of Pluvicto response",
        project_name="pluvicto_analysis",
    )
    print(results["final_summary"])

asyncio.run(main())
```

### Interactive Session

During execution, you can interrupt with `Ctrl+C` to access options:
1. **Provide guidance** - Give direction to the current workflow
2. **Skip to next phase** - Move past the current workflow step
3. **Save and exit** - Preserve progress for later resumption
4. **Continue** - Cancel the interrupt and proceed

## Workflows

### Major Workflows

| Workflow | Description | Token Budget Guidance |
|----------|-------------|----------------------|
| `brainstorming` | Explore ideas and hypotheses | Quick (~100K) |
| `literature_review` | Background research and synthesis | Thorough (~500K) |
| `start_project` | Full analysis pipeline | Comprehensive (~1M) |
| `explore_dataset` | Data characterization and EDA | Thorough (~500K) |

### Token Budget Tiers

During consultation, you can select a budget tier:

| Tier | Tokens | Estimated Cost | Use Case |
|------|--------|----------------|----------|
| Quick | ~100K | ~$0.50 | Fast exploration, simple queries |
| Thorough | ~500K | ~$2.50 | Full analysis with figures |
| Comprehensive | ~1M | ~$5.00 | Deep dive with extensive literature review |
| Custom | User-specified | Varies | Fine-grained control |

Cost estimates are based on empirical usage averaging approximately $5 per million tokens (input and output combined).

### Mini-Workflow Modules

1. **Consultation** - User discussion, goal clarification, budget selection
2. **Literature Review** - PubMed/arXiv search with critical assessment (Quick or Comprehensive mode)
3. **Planning Committee** - Multi-agent deliberation on methodology
4. **Execute Analysis** - Dayhoff→Hinton→Bayes implementation loop
5. **Write-up Results** - Documentation and report generation
6. **Critical Review** - Quality assessment and recommendations

## Security Model

MiniLab enforces strict file access control:

| Directory | Access | Purpose |
|-----------|--------|---------|
| `ReadData/` | Read-only | Protected input data |
| `Sandbox/` | Read-write | Project outputs and intermediate files |
| Other paths | Blocked | No access outside workspace |

Additional protections:
- Path traversal attacks are blocked
- Agent-specific write permissions by file type
- Comprehensive audit logging

## Project Output Structure

All outputs are organized within `Sandbox/{project_name}/`:

```
{project_name}/
├── project_specification.md    # Goals and scope from consultation
├── data_manifest.md           # Summary of input data
├── literature/
│   ├── references.md          # Bibliography
│   └── literature_summary.md  # Narrative synthesis
├── analysis/
│   ├── exploratory/          # EDA scripts and outputs
│   └── modeling/             # Statistical models
├── figures/                   # Generated visualizations
├── outputs/
│   ├── summary_report.md     # Final findings
│   └── tables/               # Result tables
└── checkpoints/              # Workflow state for resumption
```

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=MiniLab --cov-report=html
```

### Import Verification

```python
from MiniLab import (
    run_minilab,
    BohrOrchestrator,
    PathGuard,
    Agent,
    WorkflowModule,
    console,
)
print("All imports successful")
```

## Best Practices

1. **Trust the agents** - Allow the ReAct loop to iterate; avoid micromanaging
2. **Prepare your data** - Ensure data files exist in `ReadData/` before starting
3. **Use descriptive project names** - Facilitates organization and resumption
4. **Start with exploration** - Use `brainstorming` or `literature_review` to understand scope
5. **Review transcripts** - Stored in `Transcripts/` for debugging and auditing
6. **Set appropriate budgets** - Match token allocation to task complexity

## Limitations

- Agents may produce hallucinations if not properly grounded with tool use
- Long-running computations may require timeout adjustments
- API costs accumulate with complex, multi-phase analyses
- Requires active API keys for full functionality
- Currently optimized for biomedical and computational biology research

## Data Security Notice

MiniLab sends data to external APIs (Anthropic, Tavily, NCBI). Users should not process protected health information (PHI) without:
- Institutional Review Board (IRB) approval
- Business Associate Agreement (BAA) with API providers
- Appropriate de-identification procedures

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

MiniLab is inspired by and builds upon ideas from:
- [CellVoyager](https://www.biorxiv.org/content/10.1101/2025.06.03.657517v1) - Autonomous biological data analysis
- [VirtualLab](https://www.nature.com/articles/s41586-025-09442-9) - Multi-agent scientific collaboration
- Modern agentic coding assistants and ReAct-style agent architectures

## Changelog

### Version 0.3.0 (December 2025)
- Redesigned token budget system with Quick/Thorough/Comprehensive tiers and custom input
- Narrative-style orchestrator communication
- Visible cross-agent consultations
- Tiered literature review (Quick 3-step vs. Comprehensive 7-step)
- Immediate graceful exit with agent interruption propagation
- Consolidated output file structure (single living documents)
- Enhanced transcript system as single source of truth
- Agent signature guidelines ("MiniLab Agent [Name]")
- Timestamp utilities to prevent date hallucination
- Post-consultation summary showing confirmed scope and budget

### Version 0.3.2 (December 2025)
- **Intelligent Budget Allocation**: Bohr reserves 10% buffer for graceful completion, never exceeds budget
- **Contextual Autonomy**: Natural language user preferences flow through to agent tools (no hardcoded levels)
- **Budget Typo Handling**: Fixes common typos like "200l" → "200k", warns on ambiguous input
- **Hard Budget Enforcement**: `BudgetExceededError` exception and agent ReAct loop budget checks
- **User Preference Propagation**: Consultation captures "best judgment"/"without consulting" preferences
- **Auto-proceed in Autonomous Mode**: `user_input` tool respects user's autonomy preferences
- **Graceful Completion**: Always finishes cleanly, skips to writeup when budget is low

### Version 0.3.1 (December 2025)
- **TokenAccount**: Real-time token budget tracking with warnings at 60/80/95% thresholds
- **ProjectWriter**: Centralized output management preventing duplicate files
- **Complete Transcript System**: Full lab notebook capturing all agent conversations, reasoning, and tool use
- **Date Injection**: Current session date injected into all agent prompts (fixes date hallucination)
- **Conditional data_manifest.md**: Only created when data files are present
- **Single session_summary.md**: Prevented duplicate file creation by agents
- **Output Guidelines**: Agents instructed not to create redundant files (executive_summary.md, etc.)
- Budget warnings displayed to agents as they approach token limits

### Version 0.2.0 (December 2025)
- Complete architecture refactor
- PathGuard security system
- Structured 5-part agent prompting
- RAG context management with FAISS
- Modular workflow system
- Tavily web search integration
- PubMed and arXiv literature tools
- Bohr orchestrator for workflow coordination
- Console utilities for styled output
- Prompt caching for cost reduction
