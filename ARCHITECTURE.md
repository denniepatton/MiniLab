# MiniLab Architecture

## Implementation Status

All planned PRs have been implemented with **175 passing tests**:

| PR | Component | Status | Tests |
|----|-----------|--------|-------|
| PR-0 | Repo hardening (pyproject.toml, Makefile, CI) | ✅ | 13 |
| PR-1 | Orchestrator runtime (taskgraph, runlog, meetings) | ✅ | 26 |
| PR-2 | Context/memory + budget enforcement | ✅ | 26 |
| PR-3 | Tool gateway + MCP adapters | ✅ | 25 |
| PR-4 | Security/policy engine + sandbox | ✅ | 28 |
| PR-5 | Scientific workflow library (artifacts, patterns) | ✅ | 29 |
| PR-6 | Evaluation harness (MiniBench) | ✅ | 28 |

## Overview

MiniLab is a multi-agent scientific lab assistant that orchestrates specialized AI agents to perform research tasks. The system follows an **artifact-first, event-sourced** design where all operations produce reproducible outputs with full provenance tracking.

## Core Principles

1. **Artifact-First**: Every run produces artifacts in `outputs/<run_id>/` with provenance metadata
2. **Event-Sourced**: All operations emit events to a RunLog for auditability and debugging
3. **Deny-by-Default Security**: Tools require explicit policy approval; file access is jailed to workspace
4. **Token Budget Enforcement**: Hard limits on prompt sizes and tool outputs (not prompt discipline)
5. **Separation of Concerns**: Personas reason; OrchestratorRuntime delegates; ToolGateway executes

## System Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI / API                               │
│                   (minilab run --goal ...)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OrchestratorRuntime                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   plan()    │  │run_meeting()│  │  verify()   │             │
│  │ TaskGraph   │  │  Minutes    │  │  Reports    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
           │                │                │
           ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Persona Agents                             │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │  Bohr  │ │ Gould  │ │ Turing │ │ Feynman│ │  ...   │       │
│  │ (coord)│ │ (lit)  │ │ (code) │ │(theory)│ │        │       │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘       │
│                                                                 │
│  NOTE: Personas do NOT invoke tools directly.                   │
│        They produce reasoning + tool requests.                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ToolGateway                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │PolicyEngine │  │ ToolRegistry│  │  RunLog     │             │
│  │ (deny-by-   │  │ (metadata,  │  │ (events,    │             │
│  │  default)   │  │  schemas)   │  │  artifacts) │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Tool Implementations                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │filesystem│  │ terminal │  │ pubmed   │  │  arxiv   │       │
│  │ (jailed) │  │(sandboxed│  │ (search) │  │ (search) │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
MiniLab/
├── agents/              # Persona agent definitions
│   ├── base.py          # Agent base class with ReAct loop
│   ├── registry.py      # Agent factory + configuration
│   └── prompts.py       # System prompt builder
│
├── runtime/             # Orchestration runtime (PR-1) ✓
│   ├── orchestrator.py  # OrchestratorRuntime (plan/run/verify)
│   ├── taskgraph.py     # TaskNode + TaskGraph models
│   ├── meeting.py       # Team/1:1 meeting protocols
│   ├── verification.py  # Schema/File/Code verifiers
│   └── runlog.py        # Event-sourced RunLog (20+ event types)
│
├── context/             # Context management (PR-2) ✓
│   ├── budget_enforcer.py  # Hierarchical budget enforcement
│   ├── memory_manager.py   # Rolling memory with compression
│   ├── context_manager.py  # RAG retrieval
│   ├── embeddings.py       # Vector embeddings
│   └── state_objects.py    # State serialization
│
├── tools/               # Tool system (PR-3) ✓
│   ├── gateway.py       # ToolGateway (policy-checked invocation)
│   ├── mcp_adapter.py   # MCP protocol adapters
│   ├── base.py          # Tool base class
│   └── [tool modules]   # filesystem, terminal, pubmed, arxiv, etc.
│
├── security/            # Security layer (PR-4) ✓
│   ├── policy_engine.py # PolicyEngine (deny-by-default)
│   ├── sandbox.py       # Subprocess sandboxing with risk levels
│   └── path_guard.py    # File system access control
│
├── workflows/           # Scientific workflows (PR-5) ✓
│   ├── base.py          # Workflow base class
│   ├── artifacts.py     # Artifact types, store, provenance
│   ├── patterns.py      # Analysis patterns (survival, classification)
│   └── [workflow modules]
│
├── evaluation/          # Evaluation harness (PR-6) ✓
│   └── __init__.py      # MiniBench (BenchCase, BenchResult, validators)
│
├── llm_backends/        # LLM backend implementations
│   ├── base.py          # LLM backend interface
│   ├── anthropic_backend.py
│   └── openai_backend.py
│
├── core/                # Core utilities
│   ├── token_account.py # Token accounting
│   └── project_writer.py
│
├── config/              # Configuration
│   ├── loader.py        # Config loading
│   └── agents.yaml      # Agent definitions
│
├── storage/             # Persistence
│   ├── state_store.py   # State persistence
│   └── transcript.py    # Session transcripts
│
└── utils/               # Utilities
    └── timing.py        # Timing utilities
```

## Key Components

### OrchestratorRuntime

The central coordination layer that:
- Plans: Converts user goals into TaskGraphs
- Delegates: Assigns TaskNodes to persona agents
- Verifies: Checks outputs against schemas
- Recovers: Retries or replans on failure

```python
runtime = OrchestratorRuntime(config)
graph = await runtime.plan(goal="Review literature on CHIP mutations")
result = await runtime.run(graph)
```

### TaskGraph

A DAG of TaskNodes representing the work plan:
- Nodes have owners, dependencies, and output schemas
- Topological ordering ensures correct execution
- Status tracking enables checkpointing and resume

### ToolGateway

The **only** path for tool execution:
- Consults PolicyEngine before execution
- Enforces timeouts and output truncation
- Emits events to RunLog
- Stores large outputs as artifacts

### RunLog

Event-sourced audit trail:
- MessageEvent, ToolCallEvent, ToolResultEvent, ArtifactEvent
- Queryable for context retrieval
- Exported as JSONL for reproducibility

## Security Model

### Policy Scopes
- `READ_FILE`: Only under workspace root
- `WRITE_FILE`: Only under `outputs/`, `artifacts/`, `scratch/`
- `RUN_COMMAND`: Allowlisted executables only
- `NETWORK`: Disabled by default; allowlist domains when enabled

### Sandbox Constraints
- Fixed environment variables
- Working directory jailed to workspace
- Timeouts enforced
- Output truncation with artifact storage

## Token Budget Enforcement

Hard limits (not prompt discipline):
- `MAX_PROMPT_TOKENS_PER_STEP`: 8192
- `MAX_TOOL_OUTPUT_CHARS`: 4000
- `MAX_CONTEXT_ITEMS`: 20
- `SUMMARY_EVERY_N_EVENTS`: 10
- `RETRIEVAL_TOP_K`: 5

Context assembly truncates/summarizes to stay within budget.

## Artifact Provenance

Every run produces:
```
outputs/<run_id>/
├── provenance.json      # Full run metadata
├── summary.md           # Human-readable summary
├── runlog.jsonl         # Event stream
└── <workflow artifacts>
```

Provenance includes:
- run_id, timestamps
- input parameters
- tool calls (by event id)
- environment (Python version, deps hash)

## Workflow Patterns

### Literature Review
1. Plan: Define search strategy
2. Team Meeting: Agents propose search terms
3. Execute: Run PubMed/arXiv searches
4. Synthesize: Produce literature summary
5. Verify: Check citations and coverage

### Data Exploration
1. Plan: Define analysis objectives
2. Execute: Load data, compute statistics
3. Generate: Create Jupyter notebook skeleton
4. Verify: Check outputs against schema

## Extension Points

### Adding a New Tool
1. Implement tool class in `tools/local/`
2. Register in ToolRegistry with metadata
3. Define required policy scopes
4. Add tests

### Adding a New Workflow
1. Implement workflow class in `workflows/`
2. Define `build_taskgraph()` method
3. Specify artifact schemas
4. Add example + tests

### Adding a New Agent Persona
1. Define in `config/team.yaml`
2. Specify guild, tools, colleagues
3. Create system prompt template
4. Add to AgentRegistry
