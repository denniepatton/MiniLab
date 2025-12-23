# Contributing to MiniLab

Thank you for your interest in contributing to MiniLab! This document provides guidelines and instructions for development.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- micromamba or conda (recommended) or pip
- Git

### Environment Setup

#### Option 1: Using micromamba (recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/MiniLab.git
cd MiniLab

# Create environment from environment.yml
micromamba create -f environment.yml
micromamba activate minilab

# Install in development mode
pip install -e ".[dev]"
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/your-org/MiniLab.git
cd MiniLab

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Run the test suite
make test

# Run linting
make lint

# Run type checking
make typecheck

# Run all checks
make check
```

## Development Workflow

### Code Quality

We use the following tools to maintain code quality:

- **ruff**: Linting and formatting
- **mypy**: Type checking
- **pytest**: Testing

Run all checks before submitting a PR:

```bash
make all  # Runs fmt + lint + typecheck + test
```

### Testing

Write tests for all new functionality:

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_specific.py -v

# Run specific test
pytest tests/test_specific.py::test_function -v
```

Test files should be placed in `tests/` and follow the naming convention `test_*.py`.

### Code Style

- Follow PEP 8 guidelines (enforced by ruff)
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Keep functions focused and small

Example:

```python
from typing import Optional

def process_data(input_path: str, output_dir: Optional[str] = None) -> dict:
    """
    Process data from input file.
    
    Args:
        input_path: Path to input file
        output_dir: Optional output directory (defaults to outputs/)
    
    Returns:
        Dictionary containing processing results
    
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    ...
```

## Project Structure

```
MiniLab/
├── agents/          # Agent definitions and base classes
├── config/          # Configuration files and loaders
├── context/         # Context management and RAG
├── core/            # Core utilities (token accounting, project writer)
├── llm_backends/    # LLM provider implementations
├── orchestrators/   # Orchestration logic
├── runtime/         # New orchestrator runtime (PR-1+)
├── security/        # Security and policy enforcement
├── storage/         # Transcript and state storage
├── tools/           # Tool implementations
├── utils/           # Utility functions
└── workflows/       # Workflow modules

tests/               # Test files
examples/            # Runnable examples
outputs/             # Run outputs (gitignored)
```

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with appropriate tests

3. **Run checks locally**:
   ```bash
   make all
   ```

4. **Commit with clear messages**:
   ```bash
   git commit -m "feat: add new feature X"
   ```
   
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation
   - `test:` Tests
   - `refactor:` Code refactoring
   - `chore:` Maintenance

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **PR Requirements**:
   - All CI checks must pass
   - At least one approving review
   - No merge conflicts
   - Documentation updated if needed

## Architecture Guidelines

### Core Principles

1. **Personas don't invoke tools directly** - All tool use goes through ToolGateway
2. **Deny-by-default security** - Explicit policy approval required
3. **Token budgets are hard limits** - Enforced in code, not by prompts
4. **Artifact-first outputs** - Every run produces reproducible artifacts
5. **Event-sourced logging** - All operations emit events to RunLog

### Adding New Components

#### New Tool

1. Create implementation in `MiniLab/tools/local/`
2. Register in ToolRegistry with metadata
3. Define required policy scopes
4. Add tests in `tests/test_tools.py`

#### New Workflow

1. Create workflow in `MiniLab/workflows/`
2. Implement `build_taskgraph()` method
3. Define artifact schemas
4. Add example in `examples/`
5. Add tests

#### New Agent Persona

1. Add to `MiniLab/config/team.yaml`
2. Define guild, available tools, colleagues
3. Create system prompt template
4. Register in AgentRegistry

## Running Examples

```bash
# Literature review example
python examples/lit_review.py --goal "Review CHIP mutations in cancer"

# Data exploration example
python examples/data_explore.py --data ReadData/Pluvicto/

# Dry run (no API calls)
python examples/lit_review.py --dry-run
```

Examples produce outputs in `outputs/<run_id>/` with:
- `provenance.json` - Run metadata
- `summary.md` - Human-readable summary
- `runlog.jsonl` - Event stream

## Getting Help

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Use discussions for questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
