"""Team configuration loader.

Loads a unified team configuration from `MiniLab/config/team.yaml`.
This file is the single source of truth for:
- which agents exist
- which tools they can access
- which tools are write-enabled (prompt labeling)
- security capabilities for PathGuard

Design intent: keep agent autonomy high by moving rigid policy out of code.
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
    _instance: Optional["TeamConfigLoader"] = None

    def __init__(self, path: Optional[Path] = None):
        self.path = path or (Path(__file__).parent / "team.yaml")
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
        self._triads = {k: [a.lower() for a in v] for k, v in (self._raw.get("triads", {}) or {}).items()}
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
