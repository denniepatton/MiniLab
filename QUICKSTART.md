# MiniLab Quick Reference

## Installation TL;DR

```bash
cd /Users/robertpatton/MiniLab
micromamba env create -f environment.yml
micromamba activate minilab
pip install -e .
cp .env.example .env
# Edit .env with your API keys
```

## Essential Commands

### Environment
```bash
micromamba activate minilab          # Activate environment
micromamba deactivate                # Deactivate environment
```

### Running MiniLab
```bash
python scripts/run_user_meeting.py      # Interactive team meeting
python scripts/run_triad_meeting.py     # Focused triad discussion
python scripts/daily_digest.py          # Literature recommendations
python scripts/manage_project.py        # Project management
```

### Git Workflow
```bash
git status                           # Check changes
git add .                            # Stage all changes
git commit -m "Description"          # Commit changes
git push                             # Push to GitHub
git pull                             # Pull latest changes
```

## Agent Reference

| Agent | Role | Guild | Model | Key Expertise |
|-------|------|-------|-------|---------------|
| **Franklin** | PI & Synthesizer | Directional | GPT-4o | Deep learning, oncology, project vision |
| **Watson** | Clinical Critic | Directional | GPT-4o-mini | Feasibility, clinical translation |
| **Carroll** | Librarian | Directional | GPT-4o-mini | Literature, Zotero, knowledge graphs |
| **Feynman** | Physicist | Theory | Claude-3.5 | Conceptual clarity, first principles |
| **Shannon** | Info Theorist | Theory | Claude-3.5 | Causal inference, identifiability |
| **Greider** | Molecular Biologist | Theory | GPT-4o | Mechanisms, experimental design |
| **Bayes** | Statistician | Implementation | GPT-4o | Bayesian inference, uncertainty |
| **Lee** | CS Engineer | Implementation | GPT-4o | Infrastructure, reproducibility |
| **Dayhoff** | Bioinformatician | Implementation | GPT-4o-mini | Data pipelines, integration |

## Triad Configurations

- **directional_core**: Franklin, Watson, Carroll (vision & domain)
- **theory_modeling**: Feynman, Shannon, Greider (conceptual development)
- **implementation_data**: Bayes, Lee, Dayhoff (execution & analysis)

## API Keys Required

- `OPENAI_API_KEY`: https://platform.openai.com/api-keys
- `ANTHROPIC_API_KEY`: https://console.anthropic.com/settings/keys
- `ZOTERO_API_KEY` (optional): https://www.zotero.org/settings/keys
- `ZOTERO_USER_ID` (optional): https://www.zotero.org/settings/keys

## Project Structure

```
MiniLab/
├── MiniLab/                 # Core package
│   ├── agents/              # Agent system
│   ├── llm_backends/        # LLM providers
│   ├── orchestrators/       # Meeting coordination
│   ├── storage/             # State & citations
│   ├── tools/               # Agent capabilities
│   ├── bibliography/        # Literature scanning
│   └── config/              # Agent configurations
├── scripts/                 # Executable scripts
├── environment.yml          # Conda/micromamba environment
├── pyproject.toml          # Python package config
├── .env                    # API keys (DO NOT COMMIT)
├── .gitignore              # Git ignore patterns
├── README.md               # Full documentation
└── SETUP.md                # Setup instructions
```

## Python API Quick Start

### Basic Team Meeting
```python
import asyncio
from MiniLab import load_agents
from MiniLab.orchestrators.meetings import run_user_team_meeting

async def main():
    agents = load_agents()
    responses = await run_user_team_meeting(
        agents,
        user_prompt="Your question here",
        project_context="Optional context"
    )
    for agent_id, response in responses.items():
        print(f"\n[{agents[agent_id].display_name}]\n{response}")

asyncio.run(main())
```

### Project Management
```python
from MiniLab.storage.state_store import StateStore, Citation

store = StateStore()
project = store.create_project(
    project_id="my_project",
    name="My Research Project",
    description="Description here"
)

# Add citation
citation = Citation(
    key="Smith2020",
    title="Paper Title",
    authors=["Smith, J.", "Doe, J."],
    year=2020
)
project.add_citation(citation)
store.save_project(project)
```

### Daily Digest
```python
from MiniLab.bibliography import DailyDigest

digest = DailyDigest(
    state_store=store,
    topics=["deep learning", "genomics"]
)
recs = await digest.generate_daily_recommendations(num_papers=3)
print(digest.format_recommendation_email(recs))
```

## Configuration Files

### agents.yaml
Location: `MiniLab/config/agents.yaml`
- Agent personas and roles
- LLM backend assignments
- Tool assignments

### .env
Location: `.env` (root directory)
- API keys and credentials
- **NEVER commit this file**

## Common Tasks

### Change Agent Model
Edit `MiniLab/config/agents.yaml`:
```yaml
franklin:
  backend: "openai:gpt-4o-mini"  # Change to cheaper model
```

### Add New Agent
1. Add to `MiniLab/config/agents.yaml`
2. Define persona, role, backend, tools
3. Reload agents with `load_agents()`

### Create Custom Tool
1. Create file in `MiniLab/tools/`
2. Inherit from `Tool` base class
3. Implement `execute()` method

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found: minilab" | `pip install -e .` |
| "API key not set" | Check `.env` file exists and has keys |
| Rate limit errors | Use cheaper models or add delays |
| Import errors | `micromamba activate minilab` |
| Git conflicts | `git pull --rebase` |

## Storage Locations

- **Projects**: `~/.minilab/projects/*.json`
- **Global Bibliography**: `~/.minilab/global_bibliography.json`
- **Logs**: Terminal output (redirect to file if needed)

## Best Practices

1. ✅ Always activate environment before running
2. ✅ Commit changes regularly with descriptive messages
3. ✅ Test with cheap models first (gpt-4o-mini)
4. ✅ Keep `.env` backed up securely (not in git)
5. ✅ Create separate projects for different research questions
6. ❌ Never commit API keys
7. ❌ Don't use with PHI without IRB approval
8. ❌ Don't skip backups of `~/.minilab/`

## Quick Links

- [Full README](README.md) - Complete documentation
- [Setup Guide](SETUP.md) - Detailed installation
- [VirtualLab Paper](https://www.nature.com/articles/s41586-025-09442-9)
- [CellVoyager Paper](https://www.biorxiv.org/content/10.1101/2025.06.03.657517v1)

## Getting Help

1. Check this quick reference
2. Read [SETUP.md](SETUP.md) for detailed instructions
3. Review [README.md](README.md) for architecture
4. Check terminal error messages
5. Open GitHub issue if stuck

---

**Version**: 0.1.0  
**Last Updated**: December 2025
