from __future__ import annotations

import pathlib
from typing import Dict

import yaml

from MiniLab.llm_backends.base import parse_backend_name
from MiniLab.llm_backends.openai_backend import OpenAIBackend
from MiniLab.llm_backends.anthropic_backend import AnthropicBackend
from MiniLab.tools.filesystem import FileSystemTool
from .base import Agent


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "agents.yaml"
SANDBOX_PATH = ROOT.parent / "Sandbox"


def _make_backend(backend_str: str):
    provider, model = parse_backend_name(backend_str)
    if provider == "openai":
        return OpenAIBackend(model=model)
    elif provider == "anthropic":
        return AnthropicBackend(model=model)
    # TODO: add google, local, etc.
    raise ValueError(f"Unknown backend provider: {provider}")


def load_agents() -> Dict[str, Agent]:
    """
    Load agent configs from YAML and construct Agent instances.
    Initializes tools for each agent.
    """
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)

    agents_cfg = cfg.get("agents", {})
    agents: Dict[str, Agent] = {}

    for agent_id, a in agents_cfg.items():
        backend = _make_backend(a["backend"])
        
        # Initialize tool instances
        tool_instances = {}
        tool_names = a.get("tools", [])
        
        # Add filesystem tool if requested
        if "filesystem" in tool_names:
            tool_instances["filesystem"] = FileSystemTool(sandbox_root=SANDBOX_PATH)
        
        # Future: add other tools here (web_search, zotero, etc.)
        
        agents[agent_id] = Agent(
            id=agent_id,
            display_name=a["display_name"],
            guild=a["guild"],
            role=a["role"],
            persona=a["persona"],
            backend=backend,
            tools=tool_names,
            tool_instances=tool_instances,
        )

    return agents


def load_triads() -> Dict[str, list[str]]:
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    return cfg.get("triads", {})

