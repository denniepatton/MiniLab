# MiniLab

**MiniLab** is a multi-agent scientific research assistant inspired by [VirtualLab](https://www.nature.com/articles/s41586-025-09442-9) and [CellVoyager](https://www.biorxiv.org/content/10.1101/2025.06.03.657517v1). It creates a collaborative team of AI agents with specialized expertise to assist with scientific research—including literature synthesis, experimental design, coding, statistical analysis, and report generation.

## Key Features

- **9 Specialized Agents** with distinct personas and SOTA role-specific prompting
- **TRUE Agentic Execution**: Agents use tools autonomously in a ReAct-style loop
- **Modular Workflow System**: 6 composable mini-workflows for flexible pipelines
- **RAG-Based Context**: FAISS vector store with semantic + recency retrieval
- **Security-First**: PathGuard enforces read-only data, write-only sandbox
- **Web & Literature Search**: Tavily web search, PubMed, and arXiv integration
- **Cross-Agent Collaboration**: Open dialogue protocol for multi-agent deliberation
- **Comprehensive Logging**: Full transcripts with timestamps and token tracking

## Architecture (v0.2)

```
MiniLab/
├── security/              # PathGuard access control
│   └── path_guard.py      # Read/write validation, agent permissions
├── tools/                 # Typed tool system
│   ├── base.py            # Tool, ToolInput, ToolOutput with Pydantic
│   ├── filesystem.py      # File operations with security
│   ├── code_editor.py     # Code creation and editing
│   ├── terminal.py        # Shell command execution
│   ├── web_search.py      # Tavily API integration
│   ├── pubmed.py          # NCBI E-utilities search
│   ├── arxiv.py           # arXiv paper search
│   └── citation.py        # Bibliography management
├── context/               # RAG-based context management
│   ├── context_manager.py # Orchestrates context building
│   ├── embeddings.py      # sentence-transformers integration
│   ├── vector_store.py    # FAISS vector store
│   └── state_objects.py   # ProjectState, TaskState, etc.
├── agents/                # SOTA agent system
│   ├── base.py            # Agent with ReAct loop
│   ├── prompts.py         # 5-part prompt schema
│   └── registry.py        # Agent creation and lookup
├── workflows/             # Modular workflow components
│   ├── base.py            # WorkflowModule ABC
│   ├── consultation.py    # User goal clarification
│   ├── literature_review.py  # Background research
│   ├── planning_committee.py # Multi-agent deliberation
│   ├── execute_analysis.py   # Dayhoff→Hinton→Bayes loop
│   ├── writeup_results.py    # Documentation
│   └── critical_review.py    # Quality assessment
├── orchestrators/         # High-level orchestration
│   └── bohr_orchestrator.py  # Workflow selection & coordination
├── llm_backends/          # LLM integrations
│   └── anthropic_backend.py  # Claude API
├── utils/                 # Utilities
│   └── __init__.py        # Console output formatting
└── config/                # Configuration
    └── agents.yaml        # Agent personas
```

## Agent Team

All agents use **Claude Sonnet 4** via Anthropic API with SOTA role-specific prompting:

| Agent | Guild | Specialty |
|-------|-------|-----------|
| **Bohr** | Orchestration | Project coordination, user interaction, workflow selection |
| **Gould** | Synthesis | Literature review, citations, documentation, writing |
| **Farber** | Synthesis | Critical review, clinical relevance, quality assessment |
| **Feynman** | Theory | Creative problem-solving, naive questions, analogies |
| **Shannon** | Theory | Experimental design, information theory, methodology |
| **Greider** | Theory | Biological mechanisms, molecular interpretation |
| **Dayhoff** | Implementation | Bioinformatics workflows, data preparation |
| **Hinton** | Implementation | Code development, debugging, execution |
| **Bayes** | Implementation | Statistical validation, uncertainty quantification |

## Installation

### Prerequisites

- **macOS** (or Linux)
- **micromamba** (or conda/mamba)
- **Python 3.11+**
- **API Keys**: Anthropic (required), Tavily (optional), NCBI (optional)

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

# Configure environment
cat > .env << EOF
ANTHROPIC_API_KEY=your_key_here
TAVILY_API_KEY=optional_for_web_search
NCBI_EMAIL=your_email_for_pubmed
NCBI_API_KEY=optional_for_higher_rate_limits
EOF

# Verify installation
python -c "from MiniLab import run_minilab; print('✓ MiniLab ready')"
```

## Quick Start

### Command Line Interface

```bash
# Start a new analysis
python scripts/minilab.py "Analyze the Pluvicto genomic data for treatment response predictors"

# Specify workflow explicitly
python scripts/minilab.py "What is the state of the art in cfDNA analysis?" --workflow literature_review

# Resume an existing project
python scripts/minilab.py --resume Sandbox/pluvicto_analysis

# Interactive mode
python scripts/minilab.py --interactive

# List existing projects
python scripts/minilab.py --list-projects
```

### Python API

```python
import asyncio
from MiniLab import run_minilab

async def main():
    results = await run_minilab(
        request="Analyze the genomic features predictive of Pluvicto response",
        project_name="pluvicto_analysis",
        workflow="start_project",  # Optional: auto-detected if omitted
    )
    print(results["final_summary"])

asyncio.run(main())
```

## Workflows

### Major Workflows (User-Facing)

| Workflow | Description | Mini-Workflows Used |
|----------|-------------|---------------------|
| `brainstorming` | Explore ideas and approaches | Consultation → Planning Committee |
| `literature_review` | Background research | Consultation → Literature Review |
| `start_project` | Full analysis pipeline | All 6 modules in sequence |
| `work_on_existing` | Continue existing project | Consultation → Planning → Execute → Writeup → Review |
| `explore_dataset` | Data exploration focus | Consultation → Execute → Writeup |

### Mini-Workflow Modules

1. **CONSULTATION** - User discussion and requirement gathering (Bohr lead)
2. **LITERATURE REVIEW** - PubMed/arXiv search and synthesis (Gould lead)
3. **PLANNING COMMITTEE** - Multi-agent deliberation on approach (Open dialogue)
4. **EXECUTE ANALYSIS** - Dayhoff→Hinton→Bayes implementation loop
5. **WRITE-UP RESULTS** - Documentation and reporting (Gould lead)
6. **CRITICAL REVIEW** - Quality assessment and recommendations (Farber lead)

## Security Model

MiniLab enforces strict file access control via **PathGuard**:

| Directory | Access | Purpose |
|-----------|--------|---------|
| `ReadData/` | **Read-only** | Protected input data (no writes allowed) |
| `Sandbox/` | **Read-write** | Project outputs, scripts, results |
| Other paths | **Blocked** | Cannot access files outside workspace |

Additional protections:
- Path traversal (`../`) blocked
- Agent-specific permission levels
- Audit logging for all file operations

## Configuration

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional - Web Search
TAVILY_API_KEY=tvly-...

# Optional - PubMed (higher rate limits)
NCBI_EMAIL=your@email.com
NCBI_API_KEY=...

# Optional - Custom paths
MINILAB_SANDBOX=/path/to/custom/sandbox
```

### pyproject.toml Dependencies

```toml
dependencies = [
    "anthropic>=0.50.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
    "pydantic>=2.0.0",
    "sentence-transformers>=2.2.0",
    "faiss-cpu>=1.7.0",
    "tavily-python>=0.3.0",
    "aiofiles>=23.0.0",
]
```

## Development

### Running Tests

```bash
# Run all tests
micromamba run -n minilab python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_smoke.py -v

# Run with coverage
python -m pytest tests/ --cov=MiniLab --cov-report=html
```

### Project Structure Verification

```python
from MiniLab import (
    run_minilab,
    BohrOrchestrator,
    PathGuard,
    Agent,
    WorkflowModule,
    console,
)
print("✓ All imports successful")
```

## Context System

MiniLab uses a structured context approach:

1. **Static Header** - Agent persona, role, objective, tools documentation
2. **Rolling Task State** - ~1000 tokens, compressed via summarization
3. **RAG Retrieval** - Semantic + recency weighted chunks from FAISS
4. **Canonical State Objects** - Structured project/task state

```python
from MiniLab import ContextManager, ProjectState

manager = ContextManager(project_root="./", project_name="my_project")
context = manager.build_context(agent_id="hinton", task_state=current_task)
prompt = context.to_prompt()  # Ready for LLM
```

## Best Practices

1. **Let agents work** - Trust the ReAct loop; don't micromanage
2. **Check ReadData/** - Verify data exists before starting analysis
3. **Use project names** - Helps with resuming and organization
4. **Review transcripts** - Stored in Sandbox/project_name/ for debugging
5. **Start simple** - Use `brainstorming` or `literature_review` to explore first

## Limitations

- Agents may hallucinate if not grounded with tool use
- Long-running scripts may timeout (configurable limit)
- API costs can accumulate with complex analyses
- Requires API keys for full functionality

## IRB and Data Security

⚠️ **MiniLab sends data to external APIs (Anthropic, Tavily, NCBI)**

Do NOT use with protected health information (PHI) without:
- IRB approval
- BAA with Anthropic
- Proper de-identification

## License

MIT License

## Acknowledgments

Inspired by:
- [VirtualLab](https://www.nature.com/articles/s41586-025-09442-9) (Nature, 2025)
- [CellVoyager](https://www.biorxiv.org/content/10.1101/2025.06.03.657517v1) (bioRxiv, 2025)

## Changelog

### v0.2.0 (December 2025)
- Complete architecture refactor
- Added PathGuard security system
- Implemented SOTA 5-part prompting
- Added RAG context with FAISS
- Created modular workflow system
- Integrated Tavily web search
- Added PubMed/arXiv literature tools
- New BohrOrchestrator for workflow coordination
- Console utility for styled output
- Comprehensive test suite
