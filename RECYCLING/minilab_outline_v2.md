# MiniLab Outline and Project Overview

## Project Goals
MiniLab provides a single entry point to a coordinated “team” of scientific agents with complementary expertise. The team collaborates to generate and refine ideas, delegate work to specialists, and critically review outputs for correctness, completeness, and reproducibility. MiniLab targets professional-grade “dry lab” outputs: executable code, structured project trees, and polished documents/figures with correctly linked citations (default formatting follows Nature-style conventions; see `MiniLab/config/formatting_rubric.md`).

Typical project types include (non-exhaustive):
- High-level literature summaries; ultra-deep reviews synthesized into review-article-like narratives
- Teaching/briefing grounded in recent peer-reviewed sources
- Hypothesis brainstorming informed by prior work and/or provided data
- Exploratory data analysis (EDA), variable auditing, plotting, interpretation
- End-to-end analysis pipeline generation (cleaning → transforms → plots → stats → interpretation → report)
- Bioinformatics + ML pipeline development or refactoring of existing codebases

## Project Design
MiniLab integrates current best practices in agentic scientific and coding workflows, inspired by multi-agent scientific systems (e.g., Virtual Lab / CellVoyager) and by modern agentic coding patterns (e.g., VS Code-style tool-mediated iteration). The core architectural commitments are:

- A **hard-coded orchestrator** mediates all tool use (filesystem, terminal, search, document generation), enforces guardrails, logs provenance, and maintains typed I/O boundaries.
- A **Plan–Act architecture**: Bohr serves as the primary planner (produces/updates structured plans and DAGs); other agents execute bounded steps and return observations.
- **Artifact-centric execution**: substantive progress is expressed as file artifacts in `Sandbox/{project}/...`, not only as chat.
- **Verification loops** are first-class (pre-mortem → execute → verify → reflect/update).
- **Tiered memory** (static → working → long-term retrieval) supports long projects while controlling token usage.

### Design Principles
- **Autonomy with accountability**: agents can propose/refine plans and delegate work, but acceptance criteria and verification are required before marking tasks complete.
- **Agent experts & team dynamics**: each persona has a defined role; cross-discipline discussion and critique is expected.
- **Plan–Act execution**: ReAct-compatible loops operate *within* Plan–Act. Bohr plans; executors act; observations update the plan and artifacts.
- **Verification loops**: every project uses systematic review cycles:
  - (a) pre-mortem / failure anticipation
  - (b) execution
  - (c) verification (tests, sanity checks, citation validation, formatting checks)
  - (d) reflection + memory/budget update
- **Task DAGs (project-level)**: tasks are causally ordered to enforce correct information flow (e.g., ingest → transform → plot → interpret → report → review).
- **Artifact-centric SSOT**: the SSOT is the evolving set of project artifacts (plan, DAG, evidence, manifests, outputs). Agents should reference artifacts, not rely on recollection.
- **Tiered memory (token-efficient context)**:
  - **Static memory (cached)**: persona, tool schemas, constraints/policies (rarely changes)
  - **Working memory (adaptive cache)**: current plan + DAG + SSOT pointers + immediate working set
  - **Long-term memory (retrieval/RAG)**: on-demand retrieval of grounded artifacts (project notes, evidence packets, codebase docs, literature sources). Retrieval is “pull,” must be justified, and must write back into artifacts.
- **Guardrails + flexibility**:
  - **Strict** at tool boundaries: capability-gated tools, typed interfaces, path allowlists, permission checks, write-through artifact gateways, provenance logging
  - **Flexible** at the plan boundary: DAG shape may adapt; delegation is dynamic; tasks may expand into subgraphs when needed (with rationale logged and budget impact tracked)
- **Budgeting & token learning (living, compact)**: token usage is tracked by task/module/tool. Learnings are periodically compacted into structured priors/posteriors, enabling better future estimation without runaway growth.
- **Future-facing extensibility**: new LLM backbones, tools, and modules can be added without rewriting the framework.

### The MiniLab Team
- Bohr: project manager/planner (primary planner; owns DAG and acceptance criteria)
- Farber: clinician critic (clinical realism, claims discipline, translational critique)
- Gould: librarian writer (literature synthesis and narrative quality)
- Feynman: curious physicist (creative reframing, “why” probing)
- Shannon: information theorist (measurement, data abstractions, evaluation framing)
- Greider: molecular biologist (mechanistic plausibility, biological context)
- Bayes: Bayesian statistician (statistical rigor, uncertainty, calibration)
- Hinton: CS engineer (robust code, tooling, pipeline construction)
- Dayhoff: bioinformatician (data processing, workflows, domain pipelines)
- Orchestrator: non-persona controller that mediates tools, permissions, logging, schemas, and I/O between filesystem + agents

## Project Flow

### ENTRY_POINT (TASK)
**Goal:** converge quickly on project name, scope, outputs, acceptance criteria, and budget.

- STEP 1 (MODULE: CONSULTATION)
  - Bohr summarizes the user goal, proposes a project name + scope; user can iterate until correct.
- STEP 2 (MODULE: TEAM_DISCUSSION)
  - Bohr runs a structured discussion with agents. Bohr must provide the current draft plan/scope to all participants. Agents critique within their expertise and propose risks, dependencies, and verification needs.
- STEP 3 (MODULE: PLANNING)
  - Bohr produces a concrete plan and initializes the SSOT artifacts + task DAG + directories.
  - The plan must include:
    - Numbered phases that map to DAG task nodes
    - Delegation (who does what, and why)
    - Token budget per phase + total (with an explicit policy for re-allocation)
    - Exact final outputs (filenames, formats)
    - Input data/pipeline summaries (what exists, where, how it will be used)
  - User approves plan or requests changes; iterate until approved.

After approval, all agents receive the working-memory cache: plan, DAG, SSOT pointers, and project constraints.

### EXECUTION (adaptive, multi-workflow)
Bohr manages DAG execution autonomously until:
- a user decision is required (permission, scope/budget changes, major plan change)
- an agent needs additional user context
- the project completes (EXIT_POINT) or fails (FAIL_POINT)

### EXIT_POINT (TASK)
Bohr verifies outputs, summarizes what was done, and writes a pickup artifact for future continuation:
- `project_pickup.md` (agent-friendly state + next steps)
- token usage summary vs estimate
- known limitations and recommended follow-ons

### FAIL_POINT (TASK)
If the project fails (error, budget exhaustion, blocked dependency), Bohr logs:
- failure cause, reproduction steps, last-known-good artifacts
- recovery attempts performed
- a minimal pickup path for the next run

## Project Outputs

### General input/output structure
- Input data lives in `ReadData/` and is **read-only** for agents.
- All working space and outputs live in `Sandbox/{project}/`.

#### Standard skeleton
```

┌─ ReadData/ (read-only)
│   └─ {user-provided data; any structure}
│
├─ Sandbox/{project}/ (read-write)
│   ├─ artifacts/                # SSOT artifacts (plans, decisions, evidence, specs)
│   │   ├─ plan.md
│   │   ├─ evidence.md
│   │   ├─ decisions.md
│   │   └─ acceptance_checks.md
│   ├─ planning/
│   │   ├─ task_dag.dot
│   │   └─ task_dag.png
│   ├─ transcripts/
│   │   └─ transcript_{date_time}.md
│   ├─ logs/
│   │   └─ {tool logs, checkpoints, run metadata}
│   ├─ data/
│   │   ├─ raw/                  # immutable copies or pointers (no edits)
│   │   ├─ interim/              # intermediate transforms
│   │   └─ processed/            # analysis-ready datasets
│   ├─ scripts/                  # generated code; runnable end-to-end when possible
│   ├─ results/                  # figures/tables/data outputs (not narrative prose)
│   ├─ reports/                  # user-facing narrative outputs (pdf/docx/html)
│   │   └─ methods.docx
│   ├─ env/                      # reproducibility captures (versions, locks, seeds)
│   ├─ eval/                     # checks: tests, sanity reports, citation audits
│   ├─ memory/                   # long-term memory store (retrieval targets)
│   │   ├─ notes/
│   │   ├─ sources/
│   │   └─ index/
│   ├─ cache/                    # ephemeral retrieval cache (safe to delete)
│   └─ project_pickup.md
│
└─ All other paths are access-denied

```

### Always produced
- `transcripts/transcript_{date_time}.md` — complete transcript including agent discussion summaries and tool call logs/pointers
- `planning/task_dag.dot` and `planning/task_dag.png` — current DAG (agent + user view)
- `artifacts/plan.md` — SSOT plan (phases, tasks, acceptance checks)
- `project_pickup.md` — resumable state summary

### Commonly produced
- `artifacts/evidence.md` — evidence packets + citation list + “what supports what”
- `artifacts/decisions.md` — decisions + rationale + tradeoffs + budget impacts
- `data_manifest.md` (location flexible, typically `artifacts/`) — input paths + schemas + notes
- `scripts/*` — scripts to reproduce analysis and figures
- `eval/*` — sanity check outputs, unit tests, formatting/citation audit logs
- `reports/methods.docx` — methods narrative (Nature-style by default)

## Task / Module / Tool Taxonomy

### Definitions
- **Task**: a project-DAG node representing a user-meaningful milestone with clear inputs/outputs and acceptance checks. A task may expand into an internal subgraph of modules.
- **Module**: a reusable procedure that composes tools (and possibly multiple agents) to achieve a bounded subgoal. Modules can be **linear** or **small subgraphs** with retries/verification.
- **Tool**: an atomic, side-effectful capability with a typed interface enforced by the orchestrator (filesystem read/write, terminal execution, search, rendering, etc.). Tools never “decide”; they execute.

### TASKS (project DAG nodes)
Core lifecycle:
- ENTRY_POINT
- EXIT_POINT
- FAIL_POINT
- ERROR_RECOVERY (attempt recovery before FAIL_POINT)

Common scientific work:
- LITERATURE_REVIEW
- HYPOTHESIS_BRAINSTORM
- DATA_EXPLORATION
- DATA_ACQUISITION (permission-gated)
- ANALYSIS_EXECUTION
- PIPELINE_GENERATION
- FIGURE_GENERATION
- DOCUMENT_GENERATION
- RESULTS_INTERPRETATION

Quality gates:
- CRITICAL_REVIEW
- CITATION_AUDIT
- FORMATTING_AUDIT
- REPRODUCIBILITY_AUDIT

Custom:
- {CUSTOM_TASK} — permitted with explicit acceptance checks + budget impact logged

### MODULES (linear or small subgraphs)
Coordination:
- CONSULTATION
- TEAM_DISCUSSION
- ONE_ON_ONE
- PLANNING
- CORE_INPUT (Synthesis/Theory/Implementation cores)

Evidence & writing:
- EVIDENCE_GATHERING (uses search tools; writes `artifacts/evidence.md`)
- WRITE_ARTIFACT (single write gateway; enforces provenance + formatting rules)
- BUILD_REPORT (assemble narrative outputs from artifacts + results)

Execution & verification:
- GENERATE_CODE (writes runnable scripts + minimal usage instructions)
- RUN_CHECKS (tests, lint/type checks where applicable)
- SANITY_CHECK_DATA (schema/missingness/distribution checks)
- INTERPRET_STATS
- INTERPRET_PLOT
- CITATION_CHECK
- FORMATTING_CHECK

**Ephemeral external expert (on-demand):**
- CONSULT_EXTERNAL_EXPERT
  - Orchestrator spawns a short-lived specialist agent with strict contracts:
    - tool-limited (typically no filesystem writes; no installs)
    - context-limited (only a brief + relevant evidence packet)
    - output-constrained (assumptions, recommended actions, failure modes, key citations)
  - Output is treated as advice and must pass CRITICAL_REVIEW before it changes the plan.

### Project Templates (canonical starter DAGs)
Templates are “starter DAGs” that can be adapted; they are not rigid workflows.

#### 1) Literature Review
```

ENTRY_POINT
└─ LITERATURE_REVIEW
├─ (MODULE) EVIDENCE_GATHERING  -> artifacts/evidence.md
├─ (MODULE) BUILD_REPORT        -> reports/review.pdf (or .docx)
└─ (TASK)  CITATION_AUDIT
└─ CRITICAL_REVIEW
└─ EXIT_POINT

```

#### 2) Hypothesis Brainstorming
```

ENTRY_POINT
└─ HYPOTHESIS_BRAINSTORM
├─ (MODULE) TEAM_DISCUSSION
├─ (MODULE) CONSULT_EXTERNAL_EXPERT (optional)
├─ (MODULE) WRITE_ARTIFACT -> artifacts/hypotheses.md (ranked + tests)
└─ (TASK)  CRITICAL_REVIEW
└─ EXIT_POINT

```

#### 3) Data Exploration (EDA)
```

ENTRY_POINT
└─ DATA_EXPLORATION
├─ (MODULE) SANITY_CHECK_DATA  -> eval/data_sanity.md
├─ (MODULE) GENERATE_CODE      -> scripts/eda.py (or notebook)
├─ (TASK)  ANALYSIS_EXECUTION  -> results/figures/, results/tables/
└─ (TASK)  RESULTS_INTERPRETATION -> artifacts/interpretation.md
└─ REPRODUCIBILITY_AUDIT -> env/*
└─ CRITICAL_REVIEW
└─ EXIT_POINT

```

## Guardrails, Permissions, and “Living” Budget Learning

### Tool guardrails (enforced by orchestrator)
- Path allowlists:
  - Read-only: `ReadData/`
  - Read-write: `Sandbox/{project}/`
  - Everything else: access denied
- Permission-gated operations (require explicit user approval):
  - installing packages / modifying environments
  - downloading external datasets
  - large compute jobs or long-running workflows
  - changing total budget beyond agreed policy
- Write-through policy:
  - Substantive file writes must go through `WRITE_ARTIFACT` (or a controlled writer path) so provenance, formatting, and logs are consistent.

### Flexibility (permitted and expected)
- DAG shape can evolve when new information emerges, as long as:
  - rationale is written to `artifacts/decisions.md`
  - acceptance checks are updated
  - budget deltas are tracked

### Token/budget learning (compact, living)
MiniLab tracks token usage by task/module/tool to improve estimation. Instead of an infinite append-only doc:
- Maintain a compact structured file (e.g., `MiniLab/config/token_model.yaml`) containing:
  - per-task priors/posteriors (mean/variance or distribution params)
  - recency-weighted updates (decay/forgetting)
  - last-update timestamp and sample counts
- Maintain a short rolling log (e.g., last N runs) for debugging only (e.g., `MiniLab/config/token_runs_recent.jsonl`).
- Orchestrator periodically compacts/refreshes the model (e.g., after each project).

### Self-extension (proposals-based, controlled)
Agents may propose new modules/tools mid-project by writing:
- `Sandbox/{project}/artifacts/proposals/{timestamp}_{name}.md` (motivation, schema, tests, risks)
- stub implementation under `Sandbox/{project}/scripts/` (or repo path if permitted)
Activation requires orchestrator policy + user approval (PR-like gating).

## Tools (typed interfaces; orchestrator-enforced)

Tools are atomic capabilities with typed I/O. The orchestrator:
- validates schemas
- enforces permissions and path policies
- logs tool calls (inputs/outputs metadata)
- attaches outputs to artifacts when required (e.g., evidence packets)

### Core tool families (representative)
Filesystem:
- `fs.read_text(path) -> str`
- `fs.write_text(path, content, mode={overwrite|append}) -> ok`
- `fs.list_dir(path) -> [paths]`
- `fs.exists(path) -> bool`

Terminal / execution:
- `terminal.run(cmd, cwd, timeout) -> {stdout, stderr, exit_code}`
- `python.run(script_path, args) -> {stdout, stderr, artifacts_written}` (optional wrapper)

Search / retrieval (evidence-producing):
- `search.pubmed(query, filters) -> results[]`
- `search.biorxiv(query, filters) -> results[]`
- `search.web(query, recency, domains) -> results[]`
Policy: results must be summarized into `artifacts/evidence.md` with traceable citations/identifiers.

Document/figure handling (as available in environment):
- `docx.write(path, sections/spec) -> ok`
- `pdf.render(path) -> images[]` (for review/interpretation)
- `figure.compose(panels, layout_spec) -> figure_path`

Permissions:
- `permission.request(action, rationale) -> {approved|denied}`

(Additional tools can be registered via orchestrator schemas as MiniLab evolves.)
```

