"""
YAML Configuration Loader for MiniLab.

Loads agent personas and static configurations from agents.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import yaml


@dataclass
class AgentConfig:
    """Static configuration for an agent loaded from YAML."""
    agent_id: str
    display_name: str
    guild: str
    role: str
    backend: str
    tools: list[str]
    persona: str
    
    # Additional fields computed or extended
    expertise: list[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, agent_id: str, data: dict[str, Any]) -> "AgentConfig":
        """Create AgentConfig from YAML dict."""
        return cls(
            agent_id=agent_id,
            display_name=str(data.get("display_name", agent_id.title())),
            guild=str(data.get("guild", "Unknown")),
            role=str(data.get("role", "Agent")),
            backend=str(data.get("backend", "anthropic:claude-sonnet-4-5")),
            tools=list(data.get("tools", [])),
            persona=str(data.get("persona", "")),
        )


@dataclass
class TriadConfig:
    """Configuration for an agent triad."""
    name: str
    members: list[str]


class ConfigLoader:
    """
    Loads and provides access to MiniLab configuration from YAML.
    
    Singleton pattern - call ConfigLoader.get_instance() for shared access.
    """
    
    _instance: Optional["ConfigLoader"] = None
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to agents.yaml (default: MiniLab/config/agents.yaml)
        """
        if config_path is None:
            config_path = Path(__file__).parent / "agents.yaml"
        
        self.config_path = config_path
        self._raw_config: dict[str, Any] = {}
        self._agents: dict[str, AgentConfig] = {}
        self._triads: dict[str, TriadConfig] = {}
        self._default_tools: list[str] = []
        self._loaded = False
    
    @classmethod
    def get_instance(cls, config_path: Optional[Path] = None) -> "ConfigLoader":
        """Get singleton instance of ConfigLoader."""
        if cls._instance is None:
            cls._instance = cls(config_path)
            cls._instance.load()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
    
    def load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path) as f:
            self._raw_config = yaml.safe_load(f)
        
        # Parse default tools
        self._default_tools = self._raw_config.get("default_tools", [])
        
        # Parse agents
        agents_data = self._raw_config.get("agents", {})
        for agent_id, agent_data in agents_data.items():
            self._agents[agent_id] = AgentConfig.from_dict(agent_id, agent_data)
        
        # Parse triads
        triads_data = self._raw_config.get("triads", {})
        for triad_name, triad_data in triads_data.items():
            self._triads[triad_name] = TriadConfig(
                name=triad_name,
                members=triad_data.get("members", []),
            )
        
        self._loaded = True
    
    def get_agent(self, agent_id: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent."""
        if not self._loaded:
            self.load()
        return self._agents.get(agent_id)
    
    def get_all_agents(self) -> dict[str, AgentConfig]:
        """Get all agent configurations."""
        if not self._loaded:
            self.load()
        return self._agents.copy()
    
    def get_agent_ids(self) -> list[str]:
        """Get list of all agent IDs."""
        if not self._loaded:
            self.load()
        return list(self._agents.keys())
    
    def get_triads(self) -> dict[str, TriadConfig]:
        """Get all triad configurations."""
        if not self._loaded:
            self.load()
        return self._triads.copy()
    
    def get_triad_for_agent(self, agent_id: str) -> Optional[str]:
        """Get the triad name for an agent."""
        if not self._loaded:
            self.load()
        
        for triad_name, triad in self._triads.items():
            if agent_id in triad.members:
                return triad_name
        return None
    
    def get_guild_for_agent(self, agent_id: str) -> Optional[str]:
        """Get the guild for an agent."""
        agent = self.get_agent(agent_id)
        return agent.guild if agent else None
    
    def get_default_tools(self) -> list[str]:
        """Get the default tool list."""
        if not self._loaded:
            self.load()
        return self._default_tools.copy()
    
    def get_agents_by_guild(self, guild: str) -> list[AgentConfig]:
        """Get all agents in a guild."""
        if not self._loaded:
            self.load()
        return [a for a in self._agents.values() if a.guild.lower() == guild.lower()]
    
    def get_raw_config(self) -> dict[str, Any]:
        """Get raw YAML config dict."""
        if not self._loaded:
            self.load()
        return self._raw_config.copy()


# Convenience functions
def load_agent_config(agent_id: str) -> Optional[AgentConfig]:
    """Load configuration for a specific agent."""
    return ConfigLoader.get_instance().get_agent(agent_id)


def load_all_agents() -> dict[str, AgentConfig]:
    """Load all agent configurations."""
    return ConfigLoader.get_instance().get_all_agents()


def get_agent_persona(agent_id: str) -> str:
    """Get the persona string for an agent."""
    config = load_agent_config(agent_id)
    return config.persona if config else ""


def get_agent_guild(agent_id: str) -> str:
    """Get the guild for an agent."""
    config = load_agent_config(agent_id)
    return config.guild if config else "Unknown"
