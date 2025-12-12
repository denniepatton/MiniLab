"""
MiniLab Agents Module

Refactored agent system with:
- Structured role-specific prompting
- ReAct loop with typed tools
- Persistent state per task
- Interrupt/pause capabilities
"""

from .base import Agent, AgentState, AgentResponse
from .prompts import AgentPrompt, PromptBuilder
from .registry import AgentRegistry, create_agents

__all__ = [
    "Agent",
    "AgentState",
    "AgentResponse",
    "AgentPrompt",
    "PromptBuilder",
    "AgentRegistry",
    "create_agents",
]
