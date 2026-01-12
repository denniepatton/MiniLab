"""
MiniLab Orchestrators.

Infrastructure for coordinating analysis sessions.

Terminology (aligned with minilab_outline.md):
- Task: A project-DAG node representing a user-meaningful milestone
- Module: A reusable procedure that composes tools and possibly agents
- Tool: An atomic, side-effectful capability with typed I/O

Components:
- MiniLabOrchestrator: Primary orchestrator with TaskGraph-based execution
- MiniLabSession: Session state container

NOTE: BohrOrchestrator is a backwards-compatible alias for MiniLabOrchestrator.
The orchestrator is INFRASTRUCTURE, not an AI agent. The Bohr AI agent
is a separate persona defined in agents.yaml.
"""

from .orchestrator import MiniLabOrchestrator, MiniLabSession, run_minilab

# Backwards-compatible alias
BohrOrchestrator = MiniLabOrchestrator

__all__ = [
    "MiniLabOrchestrator",
    "BohrOrchestrator",  # Backwards-compatible alias
    "MiniLabSession",
    "run_minilab",
]
