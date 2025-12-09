from __future__ import annotations

import pathlib
from typing import Dict, Callable, Optional

import yaml

from MiniLab.llm_backends.base import parse_backend_name
from MiniLab.llm_backends.openai_backend import OpenAIBackend
from MiniLab.llm_backends.anthropic_backend import AnthropicBackend
from MiniLab.tools.filesystem_dual import DualModeFileSystemTool
from MiniLab.tools.environment import EnvironmentTool
from MiniLab.tools.citation import CitationTool
from MiniLab.tools.system_tools import TerminalTool, GitTool
from MiniLab.tools.web_search import WebSearchTool, PubMedSearchTool, ArxivSearchTool
from MiniLab.tools.code_editor import CodeEditorTool
from MiniLab.tools.user_input import UserInputTool
from .base import Agent


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "agents.yaml"
WORKSPACE_ROOT = ROOT.parent  # MiniLab/ directory
SANDBOX_PATH = WORKSPACE_ROOT / "Sandbox"
READDATA_PATH = WORKSPACE_ROOT / "ReadData"


def _make_backend(backend_str: str):
    provider, model = parse_backend_name(backend_str)
    if provider == "openai":
        return OpenAIBackend(model=model)
    elif provider == "anthropic":
        return AnthropicBackend(model=model)
    # TODO: add google, local, etc.
    raise ValueError(f"Unknown backend provider: {provider}")


def load_agents(
    permission_callback: Optional[Callable] = None,
) -> Dict[str, Agent]:
    """
    Load agent configs from YAML and construct Agent instances.
    Initializes tools for each agent and sets up cross-agent consultation.
    
    Args:
        permission_callback: Optional async function for package install approval.
                           If None, all package installations will fail.
    """
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)

    agents_cfg = cfg.get("agents", {})
    agents: Dict[str, Agent] = {}

    # Create shared filesystem tool (all agents get this for read access)
    shared_filesystem = DualModeFileSystemTool(
        workspace_root=WORKSPACE_ROOT,
        read_only_dirs=["ReadData"],
        read_write_dirs=["Sandbox"],
    )
    
    # Create shared environment tool with permission callback
    shared_environment = EnvironmentTool(
        environment_name="minilab",
        permission_callback=permission_callback,
    )
    
    # Create code editor tool for agents that write code
    shared_code_editor = CodeEditorTool(
        workspace_root=WORKSPACE_ROOT,
        sandbox_dir="Sandbox",
    )
    
    # Create terminal tool with workspace root for path security
    shared_terminal = TerminalTool(workspace_root=WORKSPACE_ROOT)
    
    # Create user input tool for agents to ask user directly
    shared_user_input = UserInputTool()

    for agent_id, a in agents_cfg.items():
        backend = _make_backend(a["backend"])
        
        # ALL agents get ALL core tools - they differ only in persona/role
        # This enables TRUE agentic behavior where any agent can take action
        tool_names = a.get("tools", [])
        
        tool_instances = {
            "filesystem": shared_filesystem,
            "code_editor": shared_code_editor,
            "terminal": shared_terminal,
            "environment": shared_environment,
            "user_input": shared_user_input,
            "web_search": WebSearchTool(),
            "pubmed_search": PubMedSearchTool(),
            "arxiv_search": ArxivSearchTool(),
        }
        
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
    
    # Set up colleague relationships for cross-agent consultation
    for agent in agents.values():
        agent.set_colleagues(agents)

    return agents


def load_triads() -> Dict[str, list[str]]:
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    return cfg.get("triads", {})

