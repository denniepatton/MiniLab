# MiniLab Outline and Project Overview

## Project Goals
The aim of the MiniLab project is to provide single entry point, all-in-one access to a “team” of scientific agents with unique skills and expertise spanning complementary disciplines. This highly flexible and autonomous team works together, consulting one another for idea generation and refinement, delegating questions and tasks to specific agent experts, and reviewing one another’s work for completeness and scientific merit. MiniLab is focused on professional-level, insightful, and fully reproducible scientific work and may help with tasks such as: generating high level, topical literature summaries; performing ultra-deep literature reviews and synthesizing information into “Review Article”-like narratives; teaching the user about specific science topics while grounding them in recent, peer-reviewed research; brainstorming and hypothesis generation, based on available data and/or prior work in a field; exploratory data analysis to better understand a data set(s), trends, how to use different variables, and so on; generating complete data analysis pipelines rooted in best practices, including cleaning, sorting, transformations, plotting, and interpretation; developing bioinformatic and ML pipelines de novo; refining existing code bases or pipelines; and any other typical “dry lab” scientific work. Outputs may include conversations, complex project trees and directory structures, immediately executable code in multiple languages, as well as polished documents with correctly linked citations and high-quality figures (MiniLab defaults to generating documents and figures in Nature Journal style, following their labeling and formatting schema: see MiniLab/config/formatting_rubric.md).

## Project Design
MiniLab seeks to integrate and improve on the most state-of-the-art tools for agentic prompting and team guidance for scientific work. This project was inspired by work in the Zhou lab at Stanford: CellVoyager (https://www.biorxiv.org/content/10.1101/2025.06.03.657517v1) and the Virtual Lab (https://www.biorxiv.org/content/10.1101/2024.11.11.623004v1) with the goal of combining these ideas of autonomous agentic analysis and team-based hypothesis generation and research. MiniLab further aims to use the  most advanced agentic coding and prompting infrastructure, in emulation of the success of VS Code agent integration. This means a hard-coded orchestrator to translate agentic commands to the system with clear guard rails for data security, self-aware token usage for improved planning and project scoping while staying within a budget and converging on the desired output, DAG task graphs to ensure causally linked completion in complex workflows, and adaptive token caching for personas, abilities, and project plans to maintain long contexts while staying efficient. MiniLab further takes advantage of pertinent psychology and team building research, enforcing clear roles for agents, grouping them into specialization and role-specific subgroups/cores, encouraging mutual ownership, and demanding critical self and team review.

### Design Principles
- Autonomy and flexibility: agents have considerable autonomy to propose and refine ideas, plan, and use tools; the framework is built with flexibility in mind, in order to adapt to novel workflows and projects while generating new insights.
- Agent experts: each agent has their own expertise and skillset, encouraging cross-discipline discussions and task delegation.
- Team dynamics: individual agents take on a unique role in the team, and their rubric for interactions is based on what works best in real-world teams based on human psychology.
- Plan-Act architecture: agents use a ReAct (Reasoning + Acting) -compatible execution loop within a Plan-Act architecture to alternate between thinking (reasoning) and acting (using tools) in a loop (Thought -> Action -> Observation) to solve complex tasks and communicate within a larger planning framework. Within the Plan-Act framework, Bohr serves as the primary planner (produces/updates a structured plan) while the other agents serve as executors, performing bounded steps and reporting observations.
- Verification loops: MiniLab is designed to perform professional scientific work while following best scientific and ethical practices; in every project workflows and outputs are critically reviewed for completeness, truth, consistency, style, and alignment with scientific reporting guidelines through systematic review loops: (a) pre-mortem / failture anticipation, (b) execution, (c) verification (tests, sanity checks, citation validation), (d) reflection/memory update which promote reflective episodic learning.
- Task DAG: Project-level Directed Acyclic Graphs (DAGs) are used to model and manage workflows where tasks (nodes) have dependencies, flowing in a specific direction (directed) without looping back (acyclic). They define the order tasks run in and ensure causal execution, like data extraction -> transformation -> plotting -> interpretation -> write-up -> review.
- Artifact-centric SSOT: all substantive steps and agent tasks produce or update files in Sandbox/{project}/... (manifests, intermediate datasets, figure specs, bibliography, etc.) which couple with a Single Source of Truth (SSOT) framework shared across agents, ensuring all agents have up-to-date information and context for the project, including the project directory and file structure, execution plan, data and contents, and final outputs.
- Static caching: agents' static personas (personality, expertise, MiniLab role, available tools and how to use them, etc.) are cached at the start of each project to efficiently maintain identity throughout long projects while maintaining style across projects.
- Caching and RAG: in order to maintain long context across entire projects and work in concert while efficiently using tokens, individual agents also have an adaptive context cache which handles project-agent-specific roles and tasks, the overall execution plan, DAG, SSOT, and any other shared resources or context needed to complete their work. Retrievals occur only when a task requires facts beyond the working set and must be attached to artifacts. This is accomplished through 3 layers of context control:
  - Static caching: persona, tool schemas, and constraints
  - Adaptive caching: current project working set (plan + DAG + SSOT pointers)
  - Retrieval (RAG): on-demand fetching of grounded artifacts, which may include project artifacts (manifests, prior analyses, transcripts), codebase docs and API references, and literature (papers/notes with citations)
- Token budgeting: to refine the scope of individual projects and enable transparent pricing when performing analyses, MiniLab keeps a running record of the tokens used for workflows/tasks/modules/tools and uses a Bayesian framework to iteratively improve token-use heuristics which are used to estimate cost and scope for different kinds of projects (maintained in MiniLab/config/token_usage_learnings.md.
- Guardrails: while agents may use most tools freely, tools are also capability-gated and they must ask permission to install into the MiniLab venv, download data, perform large compute jobs etc. and are programmatically constrained to only read from and write to pre-existing folders (ReadData/ and Sandbox/, respectively).
- Future facing: ultimately MiniLab represents a framework for enabling and guiding agent interaction and execution for scientific analysis; new LLM backbones and additional tools and modules can be dropped in, and prompting rubrics and guidelines are easily adaptable.

### The MiniLab Team
- Bohr: project manager, who oversees project development and completion
- Farber: clinician critic, who reviews the correctness and completeness of MiniLab's work
- Gould: librarian writer, who reads and writes scientific literature
- Feynman: curious physicist, who provides outside-the-box insights
- Shannon: information theorist, who develops data and analytical frameworks
- Greider: molecular biologist, who adds expertise in biological mechanisms
- Bayes: bayesian statistician, who ensures statistical rigor and best practices
- Hinton: CS engineer, who is an expert coder and pipeline builder
- Dayhoff: expert bioinformatician, who constructs complex data workflows
- Orchestrator: non-persona agent for translating persona agent inputs and outputs within the MiniLab framework to enable tool use and communication between the filesystem and LLMs

## Project Flow
Once MiniLab has started there is a general (automated/handled by the orchestrator) prompt for the user to input their question, research topic, data directories, etc. which may be as specific or open-ended as the user wishes. A single MiniLab workflow then follows this general flow:

### ENTRY_POINT (TASK)
- STEP 1 (MODULE: CONSULTATION): Acting alone, Bohr (project manager) will quickly summarize the user’s goal and present a project name for approval. At this point the user may (in plain language) accept Bohr’s name and scope, or ask for another name/suggest a different name, or ask Bohr to change his understanding of the project goals, and so on. This is a very fast, iterative process with Bohr to confirm a correct understanding of the desired project (and project name). It should be conversational and can take as many iterations as needed.
- STEP 2 (MODULE: DISCUSSION): Only after Bohr has explicit, plain language approval from the user regarding both the name and the scope, he will then orchestrate a team discussion. This is also meant to be a fast, efficient process where Bohr seeks expert and cross-disciplinary opinions, and is given guidance on specifics from other agents. For this DISCUSSION which is a part of the ENTRY_POINT workflow, Bohr MUST pass the current proposed plan, as it exists after CONSULTATION with the user, to all other agents. Each agent is prompted to provide feedback based on their specific expertise which, critically, must lie within the scope of the project. Bohr will then synthesize this into a detailed plan. While creating the plan, Bohr may perform additional DISCUSSIONs with one or more subsets of agents if he believes it is needed, or perform any number of ONE-ON-ONEs, until he feels his plan fully addresses the user’s request. Individual agents are free to use tools during their ONE-ON-ONEs or DISCUSSIONs to gain context, for instance if the user describes an analysis of data in ReadData/InputData, then DAYHOFF may want to briefly explore what that data looks like and provide additional data-specific context. Final plans produced by Bohr must include:
    - Clear, numbered phases (“Phase 1: Data Exploration” etc.) which directly correspond to tasks/nodes on the task DAG
    - Delegation of specific tasks and which agents will help with/be involved in what processes
    - Estimates for the tokens which will be used in each phase, as well as the total (the TOTAL token estimate must be adhered to within 10%, but Bohr may re-allocate that budget at any time)
    - The exact, named files which will constitute the final output for the user (e.g. “NarrativeReviewSummary.pdf” or “Figure1_ResponseMeasures.png” and “Figure2_BiomarkerCorrelates.png” etc.)
    - Any context-specific summaries or information which are relevant: if the user supplied data, what those files are and what they include/cover; if the user supplied a pipeline or code, the current state of them and how they function, and so on
- STEP 3 (MODULE: PLANNING): As with the consultation, the user may NOT approve the plan Bohr produces in STEP 2, in which case Bohr should take that feedback – whatever it is – and perform another discussion/follow-up until the user accepts the plan and approves beginning. The user may wish for different, specific outputs; to change the scope (use MORE or FEWER tokens); to do something else entirely; to add plots or scripts, and so on. Once approved, Bohr will then create the SSOT project plan and task DAG, initial project directories, and any other files needed to establish the project. At this point all agents should have (cached as adaptive context) the complete project plan, scope, context, DAG, structure, and SSOT framework.

### EXECUTION (adaptive, multi-workflow)
The user may interrupt to add context/suggestions or ask questions at any time, but the system should now be fully autonomous and managed by Bohr until:
- An agent needs guidance from the user, and prompts it
- An agent needs permission, e.g. to install a package, modify the budget, or alter the current plan
- The project finishes successfully and proceeds to EXIT_POINT
- The project finishes unsuccessfully (errors or runs out of tokens before completion) and proceeds to FAIL_POINT

### EXIT_POINT (TASK)
If final desired outputs detailed by Bohr exist and are correct (checked for completeness, critical review, visually interpreted, etc.) then Bohr will give a summary of what was accomplished, any changes, the final outputs, what we learned, token usage stats etc. Bohr will also summarize this info in an agent-friendly project_pickup.md file to allow for additional iteration on the project with minimal overhead. The MiniLab instance will remain active in case the user wants to perform additional work on the project immediately, or the user may exit without issues.

## Project Outputs

### General input/output structure
Input data may be formatted and organized in any way as long as it lives in ReadData/. Output data (and working space) will always live in Sandbox/{project}/ where {project} is the name agreed on with Bohr during consultation. The following represents the skeleton of how those directories are organized; depending on the request and project, Bohr may adapt the structure and outputs to meet the user's needs.
```
┌─ ReadData/ (read-only for all agents)
│   └─ {optional user provided data with any structure}
│
├─ Sandbox/{project}/ (read-write for all agents)
│   ├─ logs/
│   │   └─ {all agent-generated logs and checkpoints}
│   ├─ memory/ (agent scratch space)
│   │   ├─ notes/ (zettelkasten-style atomic notes)
│   │   ├─ sources/ (PDfs / links / bib records)
│   │   └─ index/ (vector + keyword index)
│   ├─ transcripts/
│   │   └─ transcript_{date_and_time}.md
│   ├─ planning/
│   │   ├─ {agent-facing SSOT files for planning and execution}
│   │   ├─ task_dag.dot
│   │   └─ task_dag.png
│   ├─ scripts/ (contextual; not always generated)
│   │   └─ {all agent-generated scripts and code}
│   ├─ results/
│   │   ├─ {all user-facing output files, possibly with nested structure}
│   │   └─ methods.docx
│   └─ project_pickup.md
│
└─ All other paths are access-denied
```

### Always produced
 - {project}/transcripts/transcript_{date_and_time}.md > a complete transcript of the MiniLab instance, including agent-to-agent conversations and tool calls
 - {project}/planning/task_dag.dot > the SSOT task graph, which may be updated during the project (.dot for agents)
 - {project}/planning/task_dag.png > the SSOT task graph, which may be updated during the project (.png for users)
 - {project}/results/methods.docx > Nature-style methods section describing the work done in the project to produce the results
 - {project}/project_pickup.md > summary of the project to-date and notes to allow a new instance of MiniLab to resume the project immediately

### Sometimes produced (non-exhaustive)
 - {project}/planning/data_manifest.md > a summary with pertinent notes of all user provided input data, including paths and contents
 - {project}/scripts/(data_exploration.py, generate_figures.py, generate_stats.py, gsea.R, etc.) > scripts for completing the project workflow
 - {project}/data/ > additional space for agents to store transformed or downloaded data
 - {project}/results/figures/ > intelligently structured results subdirectories containing relevant outputs

## Task DAG Hierarchy and Structure
MiniLab projects are built up of several levels, with the Task Graph (DAG) defining the broadest view. Individual nodes of the Task Graph each represent distinct tasks, or multi-module/step processes, which are internally flexible but must be completed in a specific order to ensure correct information flow and execution order. Each Task Graph will always originate with the task ENTRY_POINT and end with the task EXIT_POINT, but the workflows connecting them will vary by the project and scope. Bohr oversees the graph construction during ENTRY_POINT and monitors adherence throughout, which may include parallel execution of nodes as possible.

### TASKS
A DAG node representing a user-meaningful milestone with clear inputs/ouputs and acceptance checks; tasks may expand into a subgraph of modules
 - ENTRY_POINT: Bohr oversees entry into a MiniLab instance, picking up a project or starting fresh
 - EXIT_POINT: Bohr double-checks outputs, summarizes the project, and awaits further instructions
 - FAIL_POINT: entered when another workflow fails for any reason to log the issues and enable pickup in the future; ends the current MiniLab instance
 - LIT_REVIEW: perform a literature review of varying scope, updating the context for other agents and assembling citations
 - EXPLORE_DATA: assess and summarize data, including paths and contents, using basic command line coding and return the data context
 - RETRIEVE_DATA: download data from the web or another 3rd party source (REQUIRES PERMISSION)
 - GENERATE_DOCUMENTS: generate correctly-formatted user-facing documents of any kind
 - GENERATE_FIGURES: generate correctly-formatted user-facing figures (multi-panel figure documents) using existing plots
 - GENERATE_PIPELINE: generate a complete bioinformatics pipeline to achieve a specific goal
 - EXECUTE_ANALYSIS: run existing scripts or pipelines
 - ERROR_RECOVERY: attempt to recover before moving to FAIL_POINT
 - INTERPRET_RESULTS: interpret any novel output qualitatively, e.g. stats, p-vals, shape of a plotted curve, goodness of fit, and so on
 - CRITICAL_REVIEW: critically review documents and/or figures for completeness, scientific merit, and mistakes (think peer review at Nature)
 - CITATIONS_CHECK: ensure all citations are real, correctly ordered and formatted, and correctly pointed-to throughout a document
 - FORMATTING_CHECK: ensure formatting of documents and/or figures meet formatting specifications
 - {CUSTOM}: agents may also generate any new workflow (task node) ad hoc for a specific project in order to meet its goals

### MODULES
Reusable procedure that composes tools 9and posisbly multiple agents) to accomplish a bounded subgoal; modules may also represent subgraphs
- CONSULTATION: consult with the user to determine project goals and scope
- DISCUSSION: discuss a topic with two or more agents, meeting style
- ONE-ON-ONE: get targeted advice or expertise from a single agent, conversation-style
- PLANNING: create a detailed plan for an entire project's workflow, including the task graph, budgeting, and assignments
- CORE_INPUT: ask a *core* of agents for targeted input or advice; agents within the core will briefly discuss before coming back with a single answer
  - Synthesis Core (for critical review and feasibility insights): Bohr (representative), Farber, and Gould
  - Theory Core (for cross-discipline input and creative insights): Feynman (representative), Shannon, and Grider
  - Implementation Core (for analysis and execution insights): Dayhoff (representative), Hinton, and Bayes
- INTERPRET_STATS: interpret what stats mean, if they are reasonable, what we can infer
- INTERPRET_PLOT: visually ingest and interpret a plot, what the trends are, what the shape tells us
- GENERATE_CODE: generate a complete script, ensuring it can run successfully and produce the desired outputs
- EVIDENCE GATHERING: calls search tools consistently and write an evidence artifact
- WRITE_ARTIFACT: single gateway for file writes; enforces style/provenance
- RUN_CHECKS: run unit tests, lint, type checks, reproducibility checks
- SANITY_CHECK_DATA: validate schema, missingness, distribution

### TOOLS
Atomic, side-effectual capability with a typed interface (read file, run command, web search, render PDF, write file, etc.)
- SEARCH_PUBMED: perform PubMed search(es) across one or more topics
- SEARCH_BIORXIV: perform bioRxiv search(es) across one or more topics
- SEARCH_WEB: perform web search(es) across one or more topics
- GET_PERMISSION
- CODE_EDITOR
- FILESYSTEM
- TERMINAL
- ...
