# MiniLab

**MiniLab** is a multi-agent scientific research assistant inspired by [VirtualLab](https://www.nature.com/articles/s41586-025-09442-9) and [CellVoyager](https://www.biorxiv.org/content/10.1101/2025.06.03.657517v1). It creates a collaborative team of AI agents with specialized expertise to assist with scientific research—including literature synthesis, experimental design, coding, statistical analysis, and report generation.

## Key Features

- **9 Specialized Agents** with distinct personas but equal capabilities
- **TRUE Agentic Execution**: Agents use tools autonomously in a ReAct-style loop
- **Universal Tool Access**: All agents can read/write files, edit code, search the web, run terminal commands
- **Cross-Agent Collaboration**: Any agent can consult any other agent in real-time
- **Dual-Mode Filesystem**: ReadData/ (read-only) + Sandbox/ (read-write) for safe data handling
- **Comprehensive Logging**: Full transcripts with timestamps and token tracking

## Agent Team

All agents use **Claude Sonnet 4** via Anthropic API and share the same tool capabilities. They differ only in their personas and specialized roles:

| Agent | Role | Specialty |
|-------|------|-----------|
| **Bohr** | Project Lead | Coordination, integration, decision-making |
| **Farber** | Critical Reviewer | Clinical relevance, feasibility, catching errors |
| **Gould** | Librarian & Writer | Literature review, citations, figure legends, summaries |
| **Feynman** | Creative Theorist | Naive questions, unconventional approaches, physics analogies |
| **Shannon** | Methodologist | Experimental design, causality, statistical methods |
| **Greider** | Biological Expert | Molecular mechanisms, biological plausibility |
| **Bayes** | Statistician | Statistical analysis, clinical trial design |
| **Hinton** | Primary Coder | Script development, debugging, code execution |
| **Dayhoff** | Analysis Architect | Translating plans into executable analyses |

## Installation

### Prerequisites

- **macOS** (or Linux)
- **micromamba** (or conda/mamba)
- **Python 3.11+**
- **Anthropic API Key**

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

# Configure API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Verify installation
python -c "from MiniLab import load_agents; print('✓ MiniLab ready')"
```

## Quick Start

### Run Single Analysis Workflow

```bash
python scripts/run_single_analysis.py
```

This launches the comprehensive 7-stage research workflow:

1. **Stage 0**: Confirm files and project naming
2. **Stage 1**: Build project structure, summarize data
3. **Stage 2**: Plan analysis (Synthesis → Theory → Implementation cores)
4. **Stage 3**: Exploratory execution (if needed)
5. **Stage 4**: Complete execution with iterative debugging
6. **Stage 5**: Write-up (legends, summary with citations)
7. **Stage 6**: Critical review

**Primary Outputs** (saved to `Sandbox/ProjectName/`):
- `ProjectName_figures.pdf` - 4-6 panel Nature-style figure
- `ProjectName_legends.md` - Figure legends
- `ProjectName_summary.md` - Mini-paper with citations

### Interactive Mode

```bash
python scripts/minilab.py
```

Select from menu options for different interaction modes.

## Architecture

```
MiniLab/
├── agents/
│   ├── base.py           # Agent class with agentic_execute() ReAct loop
│   └── registry.py       # Loads agents from YAML config
├── config/
│   └── agents.yaml       # Agent personas and shared tool definitions
├── orchestrators/
│   ├── single_analysis.py  # 7-stage research workflow
│   └── meetings.py         # PI-coordinated team meetings
├── tools/
│   ├── filesystem_dual.py  # ReadData (RO) + Sandbox (RW)
│   ├── code_editor.py      # Incremental code building (10 actions)
│   ├── web_search.py       # Web, PubMed, arXiv search
│   ├── environment.py      # Package management with approval
│   └── system_tools.py     # Terminal and Git access
├── storage/
│   ├── state_store.py      # Project persistence
│   └── transcript.py       # Conversation logging
└── llm_backends/
    └── anthropic_backend.py  # Claude API integration
```

## Core Concepts

### Agentic Execution

Agents operate in a **ReAct-style loop** (`agentic_execute`):

1. **Think** about the task
2. **Use a tool** (filesystem, code_editor, web_search, etc.)
3. **Observe** the result
4. **Continue** until task is complete or ask a colleague for help

```
# Tool call format (triple-backtick blocks)
```tool
{"tool": "filesystem", "action": "list", "params": {"path": "ReadData/"}}
```

# Colleague consultation format
```colleague
{"colleague": "hinton", "question": "Can you write a script to load this data?"}
```

# Completion signal
```done
{"result": "Analysis complete", "outputs": ["analysis.pdf"]}
```
```

### Shared Context

All agents receive comprehensive context including:
- Project name and research question
- Data file inventory
- Current working plan
- Execution plan
- Previous results

This prevents hallucination by ensuring agents know what actually exists.

### Tool Capabilities

All agents have access to:

| Tool | Description |
|------|-------------|
| `filesystem` | Read/write files, list directories, create folders |
| `code_editor` | Create, view, edit, run Python scripts incrementally |
| `web_search` | Search the web for information |
| `terminal` | Run shell commands |
| `environment` | Check/install packages (with user approval) |

## Configuration

### agents.yaml

All agents share the same tools via YAML anchors:

```yaml
default_tools: &default_tools
  - filesystem
  - code_editor
  - web_search
  - terminal
  - environment

agents:
  bohr:
    backend: "anthropic:claude-sonnet-4-5"
    tools: *default_tools
    persona: |
      You are Bohr, the project lead...
  
  hinton:
    backend: "anthropic:claude-sonnet-4-5"
    tools: *default_tools
    persona: |
      You are Hinton, the primary coder...
```

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional
TAVILY_API_KEY=...       # Enhanced web search
NCBI_EMAIL=...           # PubMed API
```

## Data Directories

| Directory | Access | Purpose |
|-----------|--------|---------|
| `ReadData/` | Read-only | Protected input data |
| `Sandbox/` | Read-write | Project outputs, scripts, scratch files |
| `Transcripts/` | Auto-generated | Conversation logs |

## Security

- **Filesystem sandboxing**: Agents cannot access files outside workspace
- **ReadData protection**: Input data is read-only
- **Package approval**: Non-common packages require user confirmation
- **Path validation**: No directory traversal (`../`) allowed

## Development

### Running Tests

```bash
micromamba run -n minilab python -m pytest tests/
```

### Code Structure Verification

```bash
python -c "
from MiniLab.orchestrators.single_analysis import run_single_analysis
from MiniLab.agents.base import Agent
print('agentic_execute:', hasattr(Agent, 'agentic_execute'))
print('All systems operational')
"
```

## Best Practices

1. **Use git** for version control—no need for v2 suffixes in filenames
2. **Trust the agents** to use their tools; don't micromanage in prompts
3. **Check ReadData/** before running analysis to confirm data exists
4. **Review transcripts** for debugging and understanding agent decisions
5. **Keep prompts focused** on what you want, not how to do it

## Limitations

- Agents may occasionally hallucinate if not grounded with tool use
- Long-running scripts may timeout (5 minute limit)
- Vision capabilities (PDF viewing) are experimental
- API costs can accumulate with complex analyses

## IRB and Data Security

⚠️ **MiniLab sends data to Anthropic's API**

Do NOT use with protected health information (PHI) without:
- IRB approval
- BAA with Anthropic
- De-identification pipelines

## License

MIT License

## Acknowledgments

Inspired by:
- [VirtualLab](https://www.nature.com/articles/s41586-025-09442-9) (Nature, 2025)
- [CellVoyager](https://www.biorxiv.org/content/10.1101/2025.06.03.657517v1) (bioRxiv, 2025)
