# MiniLab Sandbox

This directory is the workspace for MiniLab agents to create, edit, and test files.

## Purpose
- Agents can safely create and modify files here without affecting the main codebase
- All code generation, testing, and experimentation happens in this space
- Acts as a secure boundary - agents cannot modify files upstream of this directory

## Structure
Feel free to organize subdirectories as needed for different projects:
- `experiments/` - Quick tests and explorations
- `projects/` - Longer-term work
- `data/` - Generated or processed data files
- `scripts/` - Utility scripts created by agents
- `notebooks/` - Jupyter notebooks for analysis

## Notes
This folder is included in `.gitignore` by default to keep experimental work local.
