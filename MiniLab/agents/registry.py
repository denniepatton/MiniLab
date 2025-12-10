"""
Agent Registry - Creates and manages agent instances.

Handles:
- Loading agent configurations
- Creating LLM backends
- Creating tool instances per agent
- Setting up colleague relationships
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

from ..llm_backends import AnthropicBackend, OpenAIBackend
from ..context import ContextManager
from ..tools.tool_factory import ToolFactory
from .base import Agent
from .prompts import PromptBuilder


# Guild definitions
GUILDS = {
    "synthesis": ["bohr", "gould", "farber"],
    "theory": ["feynman", "shannon", "greider"],
    "implementation": ["dayhoff", "hinton", "bayes"],
}

# Default LLM model
DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AgentRegistry:
    """
    Registry for creating and managing agents.
    
    Creates agents with:
    - SOTA prompts
    - Typed tools with permissions
    - LLM backends
    - Colleague relationships
    """
    
    def __init__(
        self,
        workspace_root: Path,
        context_manager: ContextManager,
        tool_factory: ToolFactory,
        default_model: str = DEFAULT_MODEL,
    ):
        """
        Initialize registry.
        
        Args:
            workspace_root: Root of the workspace
            context_manager: Shared context manager
            tool_factory: Factory for creating tools
            default_model: Default LLM model to use
        """
        self.workspace_root = workspace_root
        self.context_manager = context_manager
        self.tool_factory = tool_factory
        self.default_model = default_model
        
        self._agents: dict[str, Agent] = {}
        self._llm_backends: dict[str, Any] = {}
    
    def _create_llm_backend(self, model: str) -> Any:
        """Create LLM backend for a model string."""
        if model in self._llm_backends:
            return self._llm_backends[model]
        
        # Parse model string (e.g., "anthropic:claude-sonnet-4-20250514")
        if ":" in model:
            provider, model_name = model.split(":", 1)
        else:
            # Assume Anthropic for Claude models
            if "claude" in model.lower():
                provider = "anthropic"
                model_name = model
            else:
                provider = "openai"
                model_name = model
        
        if provider == "anthropic":
            backend = AnthropicBackend(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                model=model_name,
            )
        elif provider == "openai":
            backend = OpenAIBackend(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=model_name,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        self._llm_backends[model] = backend
        return backend
    
    def create_agent(
        self,
        agent_id: str,
        model: Optional[str] = None,
    ) -> Agent:
        """
        Create a single agent.
        
        Args:
            agent_id: Agent identifier (e.g., 'bohr', 'gould')
            model: LLM model to use (default: DEFAULT_MODEL)
            
        Returns:
            Configured Agent instance
        """
        if agent_id in self._agents:
            return self._agents[agent_id]
        
        model = model or self.default_model
        
        # Get SOTA prompt
        prompts = PromptBuilder.build_all_prompts()
        if agent_id not in prompts:
            raise ValueError(f"Unknown agent: {agent_id}")
        
        prompt = prompts[agent_id]
        
        # Get tools for this agent
        tools = self.tool_factory.create_tools_for_agent(agent_id)
        
        # Get tool documentation
        tools_doc = self.tool_factory.get_tool_documentation(agent_id)
        
        # Build full system prompt
        system_prompt = prompt.format_system_prompt(tools_doc)
        
        # Create LLM backend
        llm = self._create_llm_backend(model)
        
        # Determine guild
        guild = None
        for g, members in GUILDS.items():
            if agent_id in members:
                guild = g
                break
        
        # Create agent
        agent = Agent(
            agent_id=agent_id,
            name=prompt.name,
            guild=guild or "unknown",
            system_prompt=system_prompt,
            llm_backend=llm,
            tools=tools,
            context_manager=self.context_manager,
            max_iterations=prompt.max_iterations,
        )
        
        self._agents[agent_id] = agent
        return agent
    
    def create_all_agents(self, model: Optional[str] = None) -> dict[str, Agent]:
        """
        Create all agents.
        
        Args:
            model: LLM model to use for all agents
            
        Returns:
            Dict mapping agent_id to Agent
        """
        all_agent_ids = []
        for members in GUILDS.values():
            all_agent_ids.extend(members)
        
        for agent_id in all_agent_ids:
            self.create_agent(agent_id, model)
        
        # Set up colleague relationships
        self._setup_colleagues()
        
        return self._agents
    
    def _setup_colleagues(self) -> None:
        """Set up colleague relationships between agents."""
        for agent_id, agent in self._agents.items():
            colleagues = {
                aid: a for aid, a in self._agents.items()
                if aid != agent_id
            }
            agent.set_colleagues(colleagues)
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def get_guild_agents(self, guild: str) -> dict[str, Agent]:
        """Get all agents in a guild."""
        guild_members = GUILDS.get(guild, [])
        return {
            aid: self._agents[aid]
            for aid in guild_members
            if aid in self._agents
        }
    
    def get_all_agents(self) -> dict[str, Agent]:
        """Get all registered agents."""
        return self._agents


def create_agents(
    workspace_root: Path,
    input_callback: Optional[Callable[[str, Optional[list[str]]], str]] = None,
    permission_callback: Optional[Callable[[str], bool]] = None,
    model: str = DEFAULT_MODEL,
) -> tuple[dict[str, Agent], ContextManager, ToolFactory]:
    """
    Convenience function to create all agents with full setup.
    
    Args:
        workspace_root: Root of the workspace
        input_callback: Callback for user input
        permission_callback: Callback for permission requests
        model: LLM model to use
        
    Returns:
        Tuple of (agents dict, context_manager, tool_factory)
    """
    # Create context manager
    context_manager = ContextManager(workspace_root)
    
    # Create tool factory
    tool_factory = ToolFactory(
        workspace_root=workspace_root,
        input_callback=input_callback,
        permission_callback=permission_callback,
    )
    
    # Create registry and agents
    registry = AgentRegistry(
        workspace_root=workspace_root,
        context_manager=context_manager,
        tool_factory=tool_factory,
        default_model=model,
    )
    
    agents = registry.create_all_agents(model)
    
    return agents, context_manager, tool_factory
