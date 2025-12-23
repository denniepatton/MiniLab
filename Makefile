# MiniLab Development Makefile
# ============================================================
# Standard targets for development workflow

.PHONY: help install install-dev lint typecheck test fmt check all clean

# Default target
help:
	@echo "MiniLab Development Commands"
	@echo "============================================================"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make lint         Run ruff linter"
	@echo "  make typecheck    Run mypy type checker"
	@echo "  make test         Run pytest test suite"
	@echo "  make fmt          Format code with ruff"
	@echo "  make check        Run lint + typecheck + test"
	@echo "  make all          Full CI pipeline (fmt + check)"
	@echo "  make clean        Remove build artifacts"
	@echo ""

# ============================================================
# Installation
# ============================================================
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# ============================================================
# Quality checks
# ============================================================
lint:
	ruff check MiniLab/ tests/ examples/ scripts/

typecheck:
	mypy MiniLab/

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=MiniLab --cov-report=term-missing

# ============================================================
# Formatting
# ============================================================
fmt:
	ruff format MiniLab/ tests/ examples/ scripts/
	ruff check --fix MiniLab/ tests/ examples/ scripts/

# ============================================================
# Combined targets
# ============================================================
check: lint typecheck test

all: fmt check

# ============================================================
# Cleanup
# ============================================================
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ============================================================
# Examples
# ============================================================
run-lit-review:
	python examples/lit_review.py

run-data-explore:
	python examples/data_explore.py
