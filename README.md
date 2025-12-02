# MiniLab

**MiniLab** is a multi-agent scientific research assistant inspired by [VirtualLab](https://www.nature.com/articles/s41586-025-09442-9) and [CellVoyager](https://www.biorxiv.org/content/10.1101/2025.06.03.657517v1). It creates a collaborative team of AI agents, each with specialized expertise, to assist with scientific research including literature synthesis, experimental design, coding, statistical analysis, and idea generation.

## Overview

MiniLab provides:

- **Specialized Agent Team**: 9 expert agents across directional, theory, and implementation guilds
- **Living Bibliography**: Automatic citation tracking, knowledge graphs, and literature connections
- **Project State Management**: Persistent storage of ideas, decisions, and meeting history
- **Multi-LLM Support**: OpenAI, Anthropic Claude, and extensible to other providers
- **Tool Integration**: Web search, Zotero, PubMed, arXiv, terminal, git, and filesystem access
- **Daily Literature Digest**: Automated paper recommendations with connection analysis

### Agent Team

**Directional Guild** (Vision & Domain Expertise):
- **Franklin** (PI): Project synthesizer, computational oncologist, deep learning expert
- **Watson** (Clinical Reviewer): Adversarial critic focused on feasibility
- **Carroll** (Librarian): Literature management, Zotero integration, knowledge graphs

**Theory Guild** (Conceptual Development):
- **Feynman** (Physicist): Conceptual clarity, back-of-the-envelope checks
- **Shannon** (Information Theorist): Causal design, identifiability, information content
- **Greider** (Molecular Biologist): Mechanistic grounding, experimental viability

**Implementation Guild** (Execution):
- **Bayes** (Statistician): Bayesian inference, uncertainty quantification
- **Lee** (CS Engineer): Code quality, infrastructure, reproducibility
- **Dayhoff** (Bioinformatician): Data pipelines, synthesis of theory and practice

## Installation

### Prerequisites

- **macOS** (or Linux/Windows with micromamba/conda)
- **micromamba** (or conda/mamba)
- **Python 3.10+**
- **API Keys**: OpenAI and/or Anthropic (see `.env.example`)

### Setup with Micromamba

1. **Clone the repository** (or if you already have it locally, navigate to it):
   ```bash
   cd /Users/robertpatton/MiniLab
   ```

2. **Create micromamba environment**:
   ```bash
   micromamba env create -f environment.yml
   micromamba activate minilab
   ```

3. **Install MiniLab in development mode**:
   ```bash
   pip install -e .
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (use nano, vim, or any text editor)
   ```

5. **Verify installation**:
   ```bash
   python -c "from MiniLab import load_agents; print('✓ MiniLab installed successfully')"
   ```

### Required API Keys

Edit `.env` with your credentials:

- `OPENAI_API_KEY`: For GPT-4 models (Franklin, Watson, Carroll, Bayes, Lee, Dayhoff)
- `ANTHROPIC_API_KEY`: For Claude models (Feynman, Shannon)
- `ZOTERO_API_KEY` & `ZOTERO_USER_ID`: (Optional) For Carroll's library management
- `TAVILY_API_KEY`: (Optional) For enhanced web search
- `NCBI_EMAIL`: (Optional) For PubMed API rate limit improvements

## Quick Start

### Basic Usage

```python
import asyncio
from MiniLab import load_agents
from MiniLab.orchestrators.meetings import run_user_team_meeting

async def main():
    # Load all agents
    agents = load_agents()
    
    # Ask the team a question
    responses = await run_user_team_meeting(
        agents,
        user_prompt="What are the latest advances in deep learning for cancer genomics?",
        project_context=None,
    )
    
    # Print responses
    for agent_id, response in responses.items():
        print(f"\n[{agents[agent_id].display_name}]")
        print(response)
        print("-" * 60)

asyncio.run(main())
```

### Running Example Scripts

```bash
# Interactive team meeting
python scripts/run_user_meeting.py

# Daily literature digest (example script to be created)
python scripts/daily_digest.py

# Project management (example script to be created)
python scripts/manage_project.py
```

## Architecture

### Core Components

```
MiniLab/
├── agents/          # Agent definitions and registry
├── llm_backends/    # LLM provider interfaces (OpenAI, Anthropic)
├── orchestrators/   # Meeting coordination and workflows
├── storage/         # Project state, citations, knowledge graphs
├── tools/           # Agent capabilities (search, Zotero, terminal, etc.)
├── bibliography/    # Literature scanning and recommendations
└── config/          # Agent configurations and personas
```

### Key Abstractions

- **Agent**: Represents a team member with expertise, persona, and LLM backend
- **Tool**: Reusable capability (web search, Zotero, terminal commands, etc.)
- **ProjectState**: Persistent storage of citations, ideas, decisions, and graphs
- **Meeting**: Orchestration pattern for agent collaboration

### State Management

MiniLab stores project data in `~/.minilab/projects/` with:
- **Citations**: Full bibliography with metadata and abstracts
- **Knowledge Graph**: Concept links between papers and ideas
- **Agent Notes**: Per-agent memory and observations
- **Meeting History**: Summaries of discussions and decisions
- **Ideas**: Tracked hypotheses with status and citations

## Usage Patterns

### 1. Literature Review

```python
from MiniLab import load_agents
from MiniLab.storage.state_store import StateStore

# Create a project
store = StateStore()
project = store.create_project(
    project_id="cancer_dl_review",
    name="Deep Learning in Cancer Genomics",
    description="Survey of deep learning methods for cancer prediction"
)

# Ask Carroll to find relevant papers
agents = load_agents()
carroll = agents["carroll"]
response = await carroll.arespond(
    "Find recent papers on graph neural networks for cancer genomics"
)
```

### 2. Team Discussion

```python
from MiniLab.orchestrators.meetings import run_internal_team_meeting

# Theory guild discusses a research idea
history = await run_internal_team_meeting(
    agents={k: agents[k] for k in ["feynman", "shannon", "greider"]},
    agenda="Evaluate the identifiability of our proposed causal model",
    project_context="We're modeling gene regulatory networks...",
    rounds=3,
)
```

### 3. Daily Literature Monitoring

```python
from MiniLab.bibliography import DailyDigest

digest = DailyDigest(
    state_store=store,
    topics=["deep learning cancer", "single-cell genomics", "causal inference"]
)

recommendations = await digest.generate_daily_recommendations(
    project_id="cancer_dl_review",
    num_papers=3
)

print(digest.format_recommendation_email(recommendations))
```

## Integration with Zotero

MiniLab can integrate with your Zotero library for literature management:

1. Get your Zotero API credentials:
   - Visit https://www.zotero.org/settings/keys
   - Create a new private key with read/write access
   - Note your User ID (shown on the page)

2. Add to `.env`:
   ```
   ZOTERO_API_KEY=your_key_here
   ZOTERO_USER_ID=your_user_id
   ```

3. Use Zotero tools:
   ```python
   from MiniLab.tools.zotero import ZoteroTool
   
   zotero = ZoteroTool()
   result = await zotero.execute(action="search", query="deep learning")
   ```

## Future Development

### Near-term (v0.2)
- [ ] HPC integration for computational tasks
- [ ] Enhanced tool calling with function definitions
- [ ] Jupyter notebook interface
- [ ] Automated experiment logging

### Mid-term (v0.3)
- [ ] RAG-based agent memory
- [ ] Semantic paper embeddings for better recommendations
- [ ] IRB-compliant data handling framework
- [ ] Multi-project workspace management

### Long-term
- [ ] Local LLM support (Ollama, LM Studio)
- [ ] Agent fine-tuning on domain knowledge
- [ ] Collaborative editing of manuscripts
- [ ] Integration with lab notebooks and ELNs

## Best Practices

1. **Always Ground Claims**: Agents are instructed to cite sources for scientific claims
2. **Version Control**: Use git to track all code and configuration changes
3. **Project Scoping**: Create separate projects for distinct research questions
4. **Regular Backups**: MiniLab stores data locally; back up `~/.minilab/` regularly
5. **API Cost Management**: Monitor API usage; use cheaper models for routine tasks

## IRB and Data Security

**IMPORTANT**: MiniLab currently sends data to third-party LLM APIs (OpenAI, Anthropic). 

⚠️ **Do NOT use with protected health information (PHI) or identifiable human subjects data** without:
- IRB approval
- Business Associate Agreements (BAAs) with LLM providers
- De-identification pipelines
- Secure, local LLM deployment

For HPC integration with sensitive data, plan to deploy local models or use approved secure API gateways.

## Citation

If you use MiniLab in your research, please cite the foundational papers:

```bibtex
@article{zhou2025virtuallab,
  title={VirtualLab: AI agents as virtual research assistants},
  author={Zhou, et al.},
  journal={Nature},
  year={2025},
  doi={10.1038/s41586-025-09442-9}
}

@article{cellvoyager2025,
  title={CellVoyager: Interactive single-cell data analysis with AI agents},
  author={Zhou, et al.},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.06.03.657517v1}
}
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

[Add your license here - e.g., MIT, Apache 2.0]

## Support

- **Issues**: https://github.com/yourusername/MiniLab/issues
- **Discussions**: https://github.com/yourusername/MiniLab/discussions

## Acknowledgments

MiniLab is inspired by and builds upon concepts from:
- VirtualLab (Nature, 2025)
- CellVoyager (bioRxiv, 2025)

Developed for scientific research in deep learning, genomics, and computational biology.
