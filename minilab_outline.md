# MiniLab Outline and Project Overview

## Project Goals

MiniLab provides a single entry point to a coordinated “team” of scientific agents with complementary expertise. The team collaborates to generate and refine ideas, delegate work to specialists, and critically review outputs for correctness, completeness, and reproducibility.

MiniLab targets professional-grade “dry lab” outputs:

* Executable code (Python/R/bash, etc.)
* Structured project trees with clear provenance
* Polished documents/figures with correctly linked citations
  (default formatting follows Nature-style conventions; see `MiniLab/config/formatting_rubric.md`)

Typical project types include (non-exhaustive):

* High-level literature summaries; ultra-deep reviews synthesized into review-article-like narratives
* Teaching/briefing grounded in recent peer-reviewed sources
* Hypothesis brainstorming informed by prior work and/or provided data
* Exploratory data analysis (EDA), variable auditing, plotting, interpretation
* End-to-end analysis pipeline generation (cleaning → transforms → plots → stats → interpretation → report)
* Bioinformatics + ML pipeline development or refactoring of existing codebases

---

## Project Design

MiniLab integrates current best practices in agentic scientific and coding workflows, inspired by:

* Multi-agent scientific systems (e.g., Virtual Lab / CellVoyager)
* Modern agentic coding patterns (e.g., VS Code-style tool-mediated iteration)

Core architectural commitments:

* A **hard-coded orchestrator** mediates all tool use, enforces guardrails, logs provenance, and maintains typed I/O boundaries.
* A **Plan–Act architecture**: Bohr serves as the primary planner (produces/updates structured plans and DAGs); other agents execute bounded steps and return observations.
* **Artifact-centric execution**: substantive progress is expressed as file artifacts in `Sandbox/{project}/...`, not only as chat.
* **Verification loops** are first-class (pre-mortem → execute → verify → reflect/update).
* **Tiered memory** (static → working → long-term retrieval) supports long projects while controlling token usage.

### Design Principles

* **Autonomy with accountability**

  * Agents can propose/refine plans and delegate work, but completion requires acceptance checks, verification, and (when relevant) audits.
* **Agent experts & team dynamics**

  * Each persona has a defined role; cross-discipline critique is expected.
* **Plan–Act execution**

  * ReAct-compatible loops operate *within* Plan–Act.
  * Bohr plans; executors act; observations update plan + artifacts.
* **Verification loops**

  * Every project uses systematic cycles:

    * (a) pre-mortem / failure anticipation
    * (b) execution
    * (c) verification (tests, sanity checks, citation validation, formatting checks)
    * (d) reflection + memory/budget update
* **Task DAGs (project-level)**

  * Tasks are causally ordered to enforce correct information flow (e.g., ingest → transform → plot → interpret → report → review).
* **Artifact-centric SSOT**

  * SSOT = evolving set of project artifacts (plan, DAG, evidence, manifests, outputs).
  * Agents should reference artifacts, not rely on recollection.
* **Tiered memory (token-efficient context)**

  * **Static memory (cached):** persona, tool schemas, constraints/policies (rarely changes)
  * **Working memory (adaptive cache):** current plan + DAG + SSOT pointers + immediate working set
  * **Long-term memory (retrieval/RAG):** on-demand retrieval of grounded artifacts (project notes, evidence packets, codebase docs, literature sources). Retrieval is “pull,” must be justified, and must write back into artifacts.
* **Guardrails + flexibility (deliberate balance)**

  * **Strict** at tool boundaries: capability-gated tools, typed interfaces, path allowlists, permission checks, write-through artifact gateways, provenance logging
  * **Flexible** at the plan boundary: DAG shape may adapt; delegation is dynamic; tasks may expand into subgraphs (with rationale logged and budget impact tracked)
* **Budgeting & token learning (living, compact)**

  * Token usage is tracked by task/module/tool and periodically compacted into a structured model (not an infinite append-only doc).
* **Future-facing extensibility**

  * New LLM backbones, tools, and modules can be added without rewriting the framework.

### The MiniLab Team

Persona agents:

* **Bohr**: project manager/planner (primary planner; owns DAG and acceptance criteria)
* **Farber**: clinician critic (clinical realism, claims discipline, translational critique)
* **Gould**: librarian writer (literature synthesis, narrative quality, citation discipline)
* **Feynman**: curious physicist (creative reframing, “why” probing, conceptual stress-testing)
* **Shannon**: information theorist (measurement, data abstractions, evaluation framing)
* **Greider**: molecular biologist (mechanistic plausibility, biological context)
* **Bayes**: Bayesian statistician (statistical rigor, uncertainty, calibration)
* **Hinton**: CS engineer (robust code, tooling, pipeline construction)
* **Dayhoff**: bioinformatician (data processing, workflows, domain pipelines)

System agent:

* **Orchestrator** (non-persona controller): mediates tools, permissions, logging, schemas, and I/O between filesystem + agents.

---

## Project Flow

### ENTRY_POINT (TASK)

**Goal:** converge quickly on project name, scope, outputs, acceptance criteria, and budget.

**Step 1 — (MODULE: CONSULTATION)**

* Bohr summarizes the user goal, proposes a project name + scope.
* Iterates until the user confirms the scope and desired outputs.

**Step 2 — (MODULE: TEAM_DISCUSSION)**

* Bohr runs a structured discussion with relevant agents.
* Bohr must provide the current draft scope/plan to all participants.
* Outputs:

  * risks/unknowns
  * dependency notes
  * verification and audit requirements
  * recommended task graph structure (high-level)

**Step 3 — (MODULE: PLANNING)**
Bohr produces a concrete plan and initializes SSOT artifacts + task DAG + directories.

Plan must include:

* Numbered phases mapped to DAG task nodes
* Delegation (who does what, and why)
* Token budget per phase + total, with a policy for re-allocation
* Exact final outputs (filenames, formats)
* Input data/pipeline summaries (what exists, where, how it will be used)
* Acceptance checks (what “done” means per deliverable)

After approval, all agents receive the working-memory cache: plan, DAG, SSOT pointers, and project constraints.

### EXECUTION (adaptive, multi-workflow)

Bohr manages DAG execution autonomously until:

* a user decision is required (permissions, scope/budget changes, major plan change)
* an agent needs additional user context
* the project completes (EXIT_POINT) or fails (FAIL_POINT)

### EXIT_POINT (TASK)

Bohr verifies outputs, summarizes what was done, and writes a pickup artifact for future continuation:

* `project_pickup.md` (agent-friendly state + next steps)
* token usage summary vs estimate
* known limitations and recommended follow-ons

### FAIL_POINT (TASK)

If the project fails (error, budget exhaustion, blocked dependency), Bohr logs:

* failure cause, reproduction steps, last-known-good artifacts
* recovery attempts performed
* a minimal pickup path for the next run

---

## Project Outputs

### General input/output structure

* Input data lives in `ReadData/` and is **read-only** for agents.
* All working space and outputs live in `Sandbox/{project}/`.

### Standard skeleton

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

* `transcripts/transcript_{date_time}.md`

  * complete transcript including agent discussion summaries and tool call pointers
* `planning/task_dag.dot` and `planning/task_dag.png`

  * current DAG (agent + user view)
* `artifacts/plan.md`

  * SSOT plan (phases, tasks, acceptance checks)
* `project_pickup.md`

  * resumable state summary + next steps

### Commonly produced

* `artifacts/evidence.md`

  * evidence packets + citation list + “what supports what”
* `artifacts/decisions.md`

  * decisions + rationale + tradeoffs + budget impacts
* `artifacts/data_manifest.md`

  * input paths + schemas + notes (or a pointer to `ReadData/` structure)
* `scripts/*`

  * scripts to reproduce analysis and figures
* `eval/*`

  * sanity check outputs, unit tests, formatting/citation audit logs
* `reports/methods.docx`

  * methods narrative (Nature-style by default)

---

## Task / Module / Tool Taxonomy

### Definitions

* **Task**

  * A project-DAG node representing a user-meaningful milestone with:

    * clear inputs and outputs
    * explicit acceptance checks
    * a defined “done” condition
  * A task may expand into an internal subgraph of modules.

* **Module**

  * A reusable procedure that composes tools (and possibly multiple agents) to achieve a bounded subgoal.
  * Modules can be:

    * **Linear:** a fixed ordered sequence of steps
    * **Small subgraphs:** includes retries, branching, and verification hooks

* **Tool**

  * An atomic, side-effectful capability with typed I/O enforced by the orchestrator.
  * Tools never “decide”; they execute.
  * Tools are capability-gated and permission-controlled where needed.

---

## TASKS (Project DAG Nodes) — Detailed

Each task definition includes: **Purpose**, **Inputs**, **Outputs**, **Acceptance checks**, and common **Modules** used.

### Core lifecycle tasks

#### ENTRY_POINT

* **Purpose:** initialize project; align on scope, outputs, acceptance criteria, and budget.
* **Inputs:** user prompt; optional existing `project_pickup.md`; optional `ReadData/` paths.
* **Outputs:** `artifacts/plan.md`, `planning/task_dag.*`, directory skeleton, updated caches.
* **Acceptance checks:** user approval of plan and outputs; budget policy defined.
* **Typical modules:** CONSULTATION → TEAM_DISCUSSION → PLANNING.

#### EXIT_POINT

* **Purpose:** finalize; ensure outputs meet acceptance checks; write pickup state.
* **Inputs:** completed artifacts/results/reports; eval logs.
* **Outputs:** `project_pickup.md`, final summary, stable artifact references.
* **Acceptance checks:** pass CRITICAL_REVIEW and required audits; outputs exist and open correctly.
* **Typical modules:** RUN_CHECKS, CITATION_CHECK, FORMATTING_CHECK, WRITE_ARTIFACT.

#### ERROR_RECOVERY

* **Purpose:** structured recovery attempts prior to FAIL_POINT.
* **Inputs:** error logs; reproduction steps; last-known-good artifact pointers.
* **Outputs:** recovery report; patched scripts/configs; updated plan or escalation.
* **Acceptance checks:** error resolved or clearly bounded; next step defined.
* **Typical modules:** RUN_CHECKS, GENERATE_CODE, SANITY_CHECK_DATA.

#### FAIL_POINT

* **Purpose:** produce an actionable failure record and stable pickup for the next run.
* **Inputs:** error context; logs; partial outputs.
* **Outputs:** failure summary inside `project_pickup.md`; `artifacts/decisions.md` updated.
* **Acceptance checks:** reproduction steps captured; last-known-good artifacts identified.

---

### Common scientific work tasks

#### LITERATURE_REVIEW

* **Purpose:** gather, filter, and synthesize literature into an evidence-backed narrative.
* **Inputs:** question(s), scope constraints, optional seed papers.
* **Outputs:** `artifacts/evidence.md`, `reports/review.(docx|pdf)`, optional `artifacts/taxonomy.md`.
* **Acceptance checks:** traceable citations; claims supported; coverage matches scope.
* **Typical modules:** EVIDENCE_GATHERING → BUILD_REPORT → CITATION_CHECK.

#### HYPOTHESIS_BRAINSTORM

* **Purpose:** generate and refine hypotheses with test plans and failure modes.
* **Inputs:** domain context; constraints; optionally prior results or preliminary data.
* **Outputs:** `artifacts/hypotheses.md` with ranked hypotheses, testability, and expected readouts.
* **Acceptance checks:** each hypothesis has: assumptions, predicted observations, disconfirming tests, priority rationale.
* **Typical modules:** TEAM_DISCUSSION, CORE_INPUT, CONSULT_EXTERNAL_EXPERT (optional), WRITE_ARTIFACT.

#### DATA_EXPLORATION

* **Purpose:** quickly characterize dataset(s), structure, quality, and feasible analyses.
* **Inputs:** `ReadData/` paths; data dictionary if available.
* **Outputs:** `artifacts/data_manifest.md`, `eval/data_sanity.md`, initial plots/tables in `results/`.
* **Acceptance checks:** schemas summarized; missingness/obvious anomalies noted; reproducible EDA script produced if needed.
* **Typical modules:** SANITY_CHECK_DATA → GENERATE_CODE → ANALYSIS_EXECUTION → RESULTS_INTERPRETATION.

#### DATA_ACQUISITION (permission-gated)

* **Purpose:** obtain external datasets or references not already present.
* **Inputs:** target dataset specs; source URLs/APIs; licensing constraints.
* **Outputs:** raw data or pointers in `data/raw/`, provenance in `artifacts/decisions.md`.
* **Acceptance checks:** permission granted; provenance captured; licensing/compliance noted.
* **Typical modules:** permission.request → search/web retrieval tools → WRITE_ARTIFACT.

#### ANALYSIS_EXECUTION

* **Purpose:** execute scripts/pipelines, producing results with logged provenance.
* **Inputs:** scripts in `scripts/`; config; data under `ReadData/` or `data/`.
* **Outputs:** figures/tables in `results/`; logs in `logs/`; run metadata in `env/`.
* **Acceptance checks:** non-zero outputs; clean exit status; results correspond to plan.
* **Typical modules:** GENERATE_CODE (if needed), terminal.run, RUN_CHECKS.

#### PIPELINE_GENERATION

* **Purpose:** generate an end-to-end pipeline with reproducibility and tests.
* **Inputs:** requirements; I/O formats; computational constraints; target environment.
* **Outputs:** scripts; README/run instructions; minimal tests; manifests.
* **Acceptance checks:** pipeline runs on sample input; outputs match spec; dependencies captured.
* **Typical modules:** PLANNING → GENERATE_CODE → RUN_CHECKS → REPRODUCIBILITY_AUDIT.

#### FIGURE_GENERATION

* **Purpose:** produce publication-quality figures with formatting rules and provenance.
* **Inputs:** processed data; plot specs; journal style constraints.
* **Outputs:** `results/figures/*`; optional multi-panel composition outputs.
* **Acceptance checks:** legible; correct labels/units; consistent fonts; traceable data provenance.
* **Typical modules:** GENERATE_CODE, INTERPRET_PLOT, FORMATTING_CHECK.

#### DOCUMENT_GENERATION

* **Purpose:** produce polished narrative documents grounded in artifacts/evidence.
* **Inputs:** `artifacts/plan.md`, `artifacts/evidence.md`, results figures/tables.
* **Outputs:** `reports/*` (docx/pdf/html), `reports/methods.docx` if applicable.
* **Acceptance checks:** matches formatting rubric; citations valid; no unsupported claims.
* **Typical modules:** BUILD_REPORT, WRITE_ARTIFACT, CITATION_CHECK, FORMATTING_CHECK.

#### RESULTS_INTERPRETATION

* **Purpose:** interpret outputs (plots/stats) and produce structured takeaways and limitations.
* **Inputs:** figures/tables; test outputs; model diagnostics.
* **Outputs:** `artifacts/interpretation.md` (claims, uncertainty, caveats, next tests).
* **Acceptance checks:** interpretations align with observed results; limitations stated.
* **Typical modules:** INTERPRET_STATS, INTERPRET_PLOT, WRITE_ARTIFACT.

---

### Quality gate tasks

#### CRITICAL_REVIEW

* **Purpose:** “peer review”-style scrutiny of all major deliverables.
* **Inputs:** candidate outputs + evidence mapping.
* **Outputs:** `eval/critical_review.md` + required changes list; updated artifacts.
* **Acceptance checks:** major claims supported; no internal contradictions; reproducibility acceptable.

#### CITATION_AUDIT

* **Purpose:** confirm references are real, correctly cited, correctly linked, and appropriately scoped.
* **Inputs:** report drafts; bib entries; evidence packets.
* **Outputs:** `eval/citation_audit.md` + fixes.
* **Acceptance checks:** citations resolve; no hallucinated sources; traceability preserved.

#### FORMATTING_AUDIT

* **Purpose:** ensure formatting meets rubric (Nature-style by default).
* **Inputs:** docx/pdf/figures; rubric config.
* **Outputs:** `eval/formatting_audit.md`; patched report/figure outputs.
* **Acceptance checks:** compliance with rubric or documented exceptions.

#### REPRODUCIBILITY_AUDIT

* **Purpose:** capture environment, versions, seeds, and minimal reproduction steps.
* **Inputs:** scripts; runtime environment; logs.
* **Outputs:** `env/*`, `eval/reproducibility_audit.md`.
* **Acceptance checks:** “fresh run” instructions exist; dependencies pinned; key outputs reproducible.

---

## MODULES (Linear or Small Subgraphs) — Detailed

### Coordination modules

#### CONSULTATION (linear)

* **Goal:** confirm scope, inputs, constraints, and outputs quickly.
* **Outputs:** short scope statement + proposed project name; open questions list.
* **Notes:** should remain conversational; stop once scope is stable.

#### TEAM_DISCUSSION (linear with optional repeats)

* **Goal:** structured multi-agent feedback.
* **Inputs:** draft plan/scope; relevant artifacts.
* **Outputs:** risks, dependencies, verification requirements, task suggestions.
* **Policy:** Bohr must pass the current scope/plan to all participants.

#### ONE_ON_ONE (linear)

* **Goal:** deep dive with a specific expert agent.
* **Outputs:** targeted recommendations, pitfalls, and next actions.

#### PLANNING (subgraph)

* **Goal:** produce `artifacts/plan.md` + DAG + directory structure + acceptance checks.
* **Subgraph pattern:**

  1. outline phases and dependencies
  2. define deliverables and acceptance checks
  3. estimate token budget (with re-allocation policy)
  4. write plan + decisions artifacts
  5. initialize DAG and skeleton

#### CORE_INPUT (small subgraph)

* **Goal:** get a single coherent answer from a core subgroup.
* **Cores:**

  * Synthesis Core: Bohr + Farber + Gould (feasibility, claims discipline, narrative)
  * Theory Core: Feynman + Shannon + Greider (concepts, abstractions, mechanism)
  * Implementation Core: Dayhoff + Hinton + Bayes (pipelines, compute, stats)

### Evidence & writing modules

#### EVIDENCE_GATHERING (subgraph)

* **Goal:** consistent searching + evidence packet creation.
* **Subgraph pattern:**

  1. generate search plan (queries + inclusion/exclusion)
  2. run search tools
  3. triage results (relevance, recency, quality)
  4. write `artifacts/evidence.md` (claims → supporting sources)
  5. create/update bibliography records in `memory/sources/`

#### WRITE_ARTIFACT (linear; mandatory gateway)

* **Goal:** enforce style/provenance and keep SSOT stable.
* **Policy:** substantive file writes flow through this module (or an equivalent orchestrator-enforced writer).
* **Outputs:** written file + log pointer + optional checksum.

#### BUILD_REPORT (subgraph)

* **Goal:** assemble narrative outputs grounded in artifacts + results.
* **Subgraph pattern:** outline → draft → cite → format → audit → finalize.

### Execution & verification modules

#### GENERATE_CODE (subgraph)

* **Goal:** produce runnable scripts with minimal “how to run” instructions.
* **Outputs:** `scripts/*`, plus `eval/*` checks where feasible.
* **Subgraph pattern:** draft → run minimal test → fix → annotate run instructions.

#### RUN_CHECKS (linear)

* **Goal:** run tests/lint/type checks where applicable; always run minimal smoke checks.
* **Outputs:** `eval/run_checks.md` and terminal logs.

#### SANITY_CHECK_DATA (linear)

* **Goal:** data schema validation, missingness, distribution checks, obvious anomalies.
* **Outputs:** `eval/data_sanity.md` + suggested mitigations.

#### INTERPRET_STATS (linear)

* **Goal:** interpret statistical outputs for validity and meaning.
* **Outputs:** a structured interpretation section for `artifacts/interpretation.md`.

#### INTERPRET_PLOT (linear)

* **Goal:** interpret plots visually and identify anomalies or misrepresentations.
* **Outputs:** structured plot notes (trend, scale, caveats, next plots to run).

#### CITATION_CHECK (linear)

* **Goal:** verify citation integrity and mapping from claims to sources.
* **Outputs:** citation fix list + updated bibliography artifacts.

#### FORMATTING_CHECK (linear)

* **Goal:** verify compliance to formatting rubric.
* **Outputs:** formatting fix list + patched outputs.

### Ephemeral external expert module

#### CONSULT_EXTERNAL_EXPERT (strict-contract subgraph)

* **Goal:** on-demand specialist consultation without contaminating the project with uncontrolled tool use.
* **Contract:**

  * **tool-limited** (typically no filesystem writes; no installs)
  * **context-limited** (only a brief + relevant evidence packet)
  * **output-constrained**: assumptions, recommended actions, failure modes, key citations
* **Policy:** output is advisory; must pass CRITICAL_REVIEW before changing plan.

---

## Project Templates (Canonical Starter DAGs)

Templates are “starter DAGs” that can be adapted; they are not rigid workflows.

### 1) Literature Review

```
ENTRY_POINT
  └─ LITERATURE_REVIEW
      ├─ (MODULE) EVIDENCE_GATHERING  -> artifacts/evidence.md
      ├─ (MODULE) BUILD_REPORT        -> reports/review.(docx|pdf)
      └─ (TASK)   CITATION_AUDIT      -> eval/citation_audit.md
  └─ (TASK) CRITICAL_REVIEW           -> eval/critical_review.md
  └─ EXIT_POINT
```

### 2) Hypothesis Brainstorming

```
ENTRY_POINT
  └─ HYPOTHESIS_BRAINSTORM
      ├─ (MODULE) TEAM_DISCUSSION
      ├─ (MODULE) CONSULT_EXTERNAL_EXPERT (optional)
      ├─ (MODULE) WRITE_ARTIFACT -> artifacts/hypotheses.md (ranked + tests)
      └─ (TASK)   CRITICAL_REVIEW
  └─ EXIT_POINT
```

### 3) Data Exploration (EDA)

```
ENTRY_POINT
  └─ DATA_EXPLORATION
      ├─ (MODULE) SANITY_CHECK_DATA   -> eval/data_sanity.md
      ├─ (MODULE) GENERATE_CODE       -> scripts/eda.py (or notebook)
      ├─ (TASK)   ANALYSIS_EXECUTION  -> results/figures/, results/tables/
      └─ (TASK)   RESULTS_INTERPRETATION -> artifacts/interpretation.md
  └─ (TASK) REPRODUCIBILITY_AUDIT     -> env/* + eval/reproducibility_audit.md
  └─ (TASK) CRITICAL_REVIEW
  └─ EXIT_POINT
```

---

## Guardrails, Permissions, and “Living” Budget Learning

### Tool guardrails (enforced by orchestrator)

* **Path allowlists**

  * Read-only: `ReadData/`
  * Read-write: `Sandbox/{project}/`
  * Everything else: access denied
* **Permission-gated operations (explicit user approval required)**

  * installing packages / modifying environments
  * downloading external datasets
  * large compute jobs or long-running workflows
  * changing total budget beyond agreed policy
* **Write-through policy**

  * Substantive file writes must go through `WRITE_ARTIFACT` (or orchestrator-controlled writer paths) so provenance, formatting, and logs are consistent.
* **Provenance logging**

  * Every tool invocation is logged with:

    * inputs (or hashes for large inputs)
    * outputs (or hashes)
    * exit status and timestamps
    * environment metadata (where feasible)
    * pointers to written artifacts

### Flexibility (permitted and expected)

* DAG shape can evolve when new information emerges, as long as:

  * rationale is written to `artifacts/decisions.md`
  * acceptance checks are updated
  * budget deltas are tracked and reported

### Token/budget learning (compact, living)

MiniLab tracks token usage by task/module/tool to improve estimation:

* Maintain a compact structured file (`MiniLab/config/token_model.md`) containing:

  * per-task priors/posteriors (e.g., mean/variance or distribution params)
  * recency-weighted updates (decay/forgetting)
  * last-update timestamp and sample counts
  * model version
* Maintain a short rolling log for debugging only (e.g., last N runs): `MiniLab/config/token_runs_recent.jsonl`.
* Orchestrator periodically compacts/refreshes the model after each project (or after major phases).

### Self-extension (proposals-based, controlled)

Agents may propose new modules/tools mid-project by writing:

* `Sandbox/{project}/artifacts/proposals/{timestamp}_{name}.md` including:

  * motivation/use-case
  * typed schema (inputs/outputs)
  * tests/verification approach
  * safety/guardrail implications
  * expected token and compute impact
* a stub implementation under `Sandbox/{project}/scripts/` (or repo path if permitted)

Activation requires orchestrator policy + user approval (PR-like gating).

---

## Tools (Typed Interfaces; Orchestrator-Enforced)

Tools are atomic capabilities with typed I/O. The orchestrator:

* validates schemas
* enforces permissions and path policies
* logs tool calls (inputs/outputs metadata)
* attaches outputs to artifacts when required (e.g., evidence packets)
* prevents “tool drift” (agents calling tools outside allowed contexts)

### Tool families (representative typed API)

#### Filesystem (`fs.*`)

Purpose: read/write/list within allowlisted paths.

* `fs.read_text(path: str) -> str`
* `fs.write_text(path: str, content: str, mode: "overwrite"|"append") -> {"ok": bool}`
* `fs.list_dir(path: str) -> {"paths": list[str]}`
* `fs.exists(path: str) -> {"exists": bool}`
* `fs.read_bytes(path: str) -> bytes` (optional)
* `fs.write_bytes(path: str, blob: bytes, mode) -> {"ok": bool}` (optional)

**Guardrails:** deny any path outside allowlists; enforce max file sizes as configured.

#### Terminal execution (`terminal.*`)

Purpose: execute commands reproducibly and capture output.

* `terminal.run(cmd: list[str], cwd: str|None, timeout_s: int|None) -> {"stdout": str, "stderr": str, "exit_code": int}`

**Guardrails:** allowlisted working directories; timeout by default; disallow destructive commands per policy.

#### Python runner (`py.*`) (optional wrapper)

Purpose: consistent execution of Python scripts with metadata capture.

* `py.run(script_path: str, args: list[str], cwd: str|None) -> {"stdout": str, "stderr": str, "exit_code": int, "artifacts_written": list[str]}`

#### Search and retrieval (`search.*`) — evidence-producing

Purpose: retrieve information when outside the working set; must write back into `artifacts/evidence.md`.

* `search.pubmed(query: str, filters: dict) -> {"results": list[Result]}`
* `search.biorxiv(query: str, filters: dict) -> {"results": list[Result]}`
* `search.web(query: str, recency_days: int|None, domains: list[str]|None) -> {"results": list[Result]}`

`Result` minimally includes:

* `title`, `authors` (if available), `year`, `venue`
* stable identifier (`doi`, `pmid`, `url`, etc.)
* snippet/abstract (if available)

**Policy:** search results must be triaged and summarized into `artifacts/evidence.md`, with identifiers recorded in `memory/sources/`.

#### Document generation (`doc.*`)

Purpose: produce reports with consistent structure and formatting.

* `doc.write_docx(path: str, spec: dict) -> {"ok": bool}`
* `doc.write_pdf(path: str, spec: dict) -> {"ok": bool}` (optional)
* `doc.convert(input_path: str, output_path: str, format: str) -> {"ok": bool}` (optional)

**Policy:** documents should be assembled from SSOT artifacts; formatting audit is recommended for final outputs.

#### Figure tooling (`fig.*`)

Purpose: compose publication-quality figures.

* `fig.save_plot(path: str, meta: dict) -> {"ok": bool}`
* `fig.compose_panels(panels: list[str], layout_spec: dict, out_path: str) -> {"ok": bool}`

**Policy:** figure provenance should reference data sources and scripts used.

#### Rendering/inspection (`render.*`)

Purpose: enable visual inspection/review.

* `render.pdf_to_images(path: str) -> {"images": list[str]}` (paths)
* `render.image_info(path: str) -> {"width": int, "height": int, ...}`

#### Permissions (`permission.*`)

Purpose: request explicit user approval for restricted operations.

* `permission.request(action: str, rationale: str, impact: dict|None) -> {"approved": bool, "notes": str|None}`

Examples of actions:

* `install_dependency`, `download_external_data`, `run_large_compute`, `increase_budget`, `write_outside_sandbox` (typically forbidden)

---

## Self-Correctness and Consistency Notes (internal alignment)

This outline is designed to be internally consistent along these axes:

* **Hierarchy clarity:** tasks are DAG milestones; modules are reusable procedures (linear or subgraph); tools are typed atomic actions enforced by orchestrator.
* **SSOT coherence:** `artifacts/` is the authoritative record; plans/evidence/decisions/acceptance checks are explicitly stored.
* **Memory correctness:** tiered memory is explicitly defined and operationalized via retrieval writing back into artifacts.
* **Guardrails vs flexibility:** strict at execution boundaries; flexible at planning/DAG shaping with mandatory rationale and budget tracking.
* **Budget learning scalability:** living compact model + short rolling log prevents unbounded growth.
* **Ephemeral expert safety:** strict contracts prevent uncontrolled tool usage and keep advice auditable through CRITICAL_REVIEW.
