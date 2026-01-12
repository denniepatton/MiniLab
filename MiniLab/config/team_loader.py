"""Team configuration loader.

Loads unified agent configuration from agents_unified.yaml (or agents.yaml).
This loader is maintained for backwards compatibility with code that expects
the TeamConfigLoader interface.

Both this loader and loader.py now read from the same unified config file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class TeamSecurityConfig:
    writable_subdirs: list[str] = field(default_factory=lambda: ["*"])
    writable_extensions: list[str] = field(default_factory=lambda: ["*"])
    can_execute_shell: bool = False
    can_modify_environment: bool = False


@dataclass(frozen=True)
class TeamAgentConfig:
    agent_id: str
    display_name: str
    guild: str
    role: str
    backend: str
    tools: list[str] = field(default_factory=list)
    write_tools: list[str] = field(default_factory=list)
    security: TeamSecurityConfig = field(default_factory=TeamSecurityConfig)


class TeamConfigLoader:
    """
    Loads team configuration from unified agents YAML.
    
    This loader maintains backwards compatibility with code expecting
    TeamConfigLoader interface while reading from the unified config.
    """
    _instance: Optional["TeamConfigLoader"] = None

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            # Prefer unified config, fall back to legacy team.yaml
            config_dir = Path(__file__).parent
            unified_path = config_dir / "agents_unified.yaml"
            legacy_path = config_dir / "team.yaml"
            path = unified_path if unified_path.exists() else legacy_path
        
        self.path = path
        self._raw: dict[str, Any] = {}
        self._agents: dict[str, TeamAgentConfig] = {}
        self._triads: dict[str, list[str]] = {}
        self._loaded = False

    @classmethod
    def get_instance(cls, path: Optional[Path] = None) -> "TeamConfigLoader":
        if cls._instance is None:
            cls._instance = cls(path)
            cls._instance.load()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def load(self) -> None:
        if self._loaded:
            return
        if not self.path.exists():
            raise FileNotFoundError(f"Team config not found: {self.path}")
        with open(self.path) as f:
            self._raw = yaml.safe_load(f) or {}

        agents = self._raw.get("agents", {})
        parsed: dict[str, TeamAgentConfig] = {}
        for agent_id, data in agents.items():
            sec = data.get("security", {}) or {}
            security = TeamSecurityConfig(
                writable_subdirs=list(sec.get("writable_subdirs", ["*"])),
                writable_extensions=list(sec.get("writable_extensions", ["*"])),
                can_execute_shell=bool(sec.get("can_execute_shell", False)),
                can_modify_environment=bool(sec.get("can_modify_environment", False)),
            )
            parsed[agent_id.lower()] = TeamAgentConfig(
                agent_id=agent_id.lower(),
                display_name=str(data.get("display_name", agent_id.title())),
                guild=str(data.get("guild", "unknown")).lower(),
                role=str(data.get("role", "Agent")),
                backend=str(data.get("backend", "anthropic:claude-sonnet-4-5")),
                tools=[t.lower() for t in list(data.get("tools", []))],
                write_tools=[t.lower() for t in list(data.get("write_tools", []))],
                security=security,
            )

        self._agents = parsed
        
        # Parse triads - handle both formats:
        # Old format: triads: {name: [members]}
        # New format: triads: {name: {members: [members], description: "..."}}
        triads_raw = self._raw.get("triads", {}) or {}
        for k, v in triads_raw.items():
            if isinstance(v, list):
                # Old format: list of members directly
                self._triads[k] = [a.lower() for a in v]
            elif isinstance(v, dict):
                # New format: dict with 'members' key
                members = v.get("members", [])
                self._triads[k] = [a.lower() for a in members]
            else:
                self._triads[k] = []
        
        self._loaded = True

    def get_agent(self, agent_id: str) -> Optional[TeamAgentConfig]:
        if not self._loaded:
            self.load()
        return self._agents.get(agent_id.lower())

    def get_all_agents(self) -> dict[str, TeamAgentConfig]:
        if not self._loaded:
            self.load()
        return dict(self._agents)

    def get_agent_ids(self) -> list[str]:
        if not self._loaded:
            self.load()
        return list(self._agents.keys())

    def get_triads(self) -> dict[str, list[str]]:
        if not self._loaded:
            self.load()
        return dict(self._triads)


def get_team_config() -> TeamConfigLoader:
    return TeamConfigLoader.get_instance()
