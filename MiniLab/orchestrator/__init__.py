"""
MiniLab Orchestrators.

Infrastructure for coordinating analysis sessions.

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
