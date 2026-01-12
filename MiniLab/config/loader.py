"""Agent/persona configuration loader.

This module exists for backwards compatibility with code that imports
`MiniLab.config.loader.load_agent_config`.

Current source of truth is `agents_unified.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, cast

import yaml


@dataclass(frozen=True)
class AgentConfig:
    agent_id: str
    display_name: str
    guild: str
    role: str
    backend: str
    persona: str = ""
    tools: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FlexibilityConfig:
    raw: dict[str, Any]


def _default_agents_path() -> Path:
    config_dir = Path(__file__).parent
    unified = config_dir / "agents_unified.yaml"
    legacy = config_dir / "agents.yaml"
    return unified if unified.exists() else legacy


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_agent_config(agent_id: str, *, path: Optional[Path] = None) -> Optional[AgentConfig]:
    """Load a single agent's persona/config from YAML."""
    p = path or _default_agents_path()
    raw = _load_yaml(p)
    agents_any = raw.get("agents", {}) or {}
    agents = cast(dict[str, Any], agents_any)
    data_any = agents.get(str(agent_id).lower())
    data = data_any if isinstance(data_any, dict) else None
    if not isinstance(data, dict):
        return None

    return AgentConfig(
        agent_id=str(agent_id).lower(),
        display_name=str(data.get("display_name", str(agent_id).title())),
        guild=str(data.get("guild", "unknown")).lower(),
        role=str(data.get("role", "Agent")),
        backend=str(data.get("backend", "anthropic:claude-sonnet-4-5")),
        persona=str(data.get("persona", "") or ""),
        tools=[str(t).lower() for t in cast(list[Any], list(data.get("tools", []) or []))],
    )


def load_flexibility_config(*, path: Optional[Path] = None) -> FlexibilityConfig:
    """Load flexibility/autonomy guidance.

    Currently returns the raw YAML structure for maximum compatibility.
    """
    config_dir = Path(__file__).parent
    p = Path(path) if path else (config_dir / "agent_flexibility.yaml")
    return FlexibilityConfig(raw=_load_yaml(p))
