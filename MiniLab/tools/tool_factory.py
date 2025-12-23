"""
Tool Factory: Creates agent-specific tool instances with proper permissions.

This module is the central point for creating tools with:
- PathGuard integration for security
- Agent-specific permissions (write powers)
- Shared callbacks (user input, permission requests)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from ..context import ContextManager

from ..security import PathGuard, AgentPermissions
from ..config.team_loader import get_team_config
from .base import Tool, ToolRegistry
from .filesystem import FileSystemTool
from .code_editor import CodeEditorTool
from .terminal import TerminalTool
from .environment import EnvironmentTool
from .user_input import UserInputTool
from .web_search import WebSearchTool
from .pubmed import PubMedTool
from .arxiv import ArxivTool
from .citation import CitationTool


def _default_agent_permissions() -> dict[str, dict[str, list[str]]]:
    """Backwards-compatible fallback tool permissions."""
    return {
        "bohr": {"tools": ["filesystem", "user_input", "web_search", "pubmed", "arxiv", "citation", "environment"], "write_tools": []},
        "gould": {"tools": ["filesystem", "user_input", "web_search", "pubmed", "arxiv", "citation"], "write_tools": ["filesystem", "citation"]},
        "farber": {"tools": ["filesystem", "user_input", "web_search", "pubmed", "citation"], "write_tools": ["filesystem"]},
        "feynman": {"tools": ["filesystem", "user_input", "web_search", "pubmed", "arxiv"], "write_tools": ["filesystem"]},
        "shannon": {"tools": ["filesystem", "user_input", "web_search", "arxiv"], "write_tools": ["filesystem"]},
        "greider": {"tools": ["filesystem", "user_input", "web_search", "pubmed"], "write_tools": ["filesystem"]},
        "dayhoff": {"tools": ["filesystem", "user_input", "web_search", "pubmed"], "write_tools": ["filesystem"]},
        "hinton": {"tools": ["filesystem", "code_editor", "terminal", "user_input", "environment"], "write_tools": ["filesystem", "code_editor", "terminal"]},
        "bayes": {"tools": ["filesystem", "code_editor", "terminal", "user_input"], "write_tools": ["filesystem", "code_editor", "terminal"]},
    }


class ToolFactory:
    """
    Factory for creating agent-specific tool instances.
    
    Handles:
    - PathGuard initialization
    - Agent-specific tool sets
    - Shared callbacks
    - Tool registry management
    """
    
    def __init__(
        self,
        workspace_root: Path,
        input_callback: Optional[Callable[[str, Optional[list[str]]], str]] = None,
        permission_callback: Optional[Callable[[str], bool]] = None,
        context_manager: Optional[ContextManager] = None,
    ):
        """
        Initialize the tool factory.
        
        Args:
            workspace_root: Root directory of the workspace
            input_callback: Callback for user input
            permission_callback: Callback for permission requests
        """
        self.workspace_root = workspace_root
        self.input_callback = input_callback
        self.permission_callback = permission_callback
        self.context_manager = context_manager

        # Load team config (single source of truth)
        self.team_config = None
        try:
            self.team_config = get_team_config()
        except Exception:
            self.team_config = None
        
        # Initialize PathGuard singleton
        PathGuard.reset()
        self.path_guard = PathGuard(workspace_root)

        # Apply agent security permissions from config
        if self.team_config:
            for agent_id, cfg in self.team_config.get_all_agents().items():
                self.path_guard.set_agent_permissions(
                    agent_id,
                    AgentPermissions(
                        agent_id=agent_id,
                        writable_subdirs=list(cfg.security.writable_subdirs),
                        writable_extensions=list(cfg.security.writable_extensions),
                        can_execute_shell=bool(cfg.security.can_execute_shell),
                        can_modify_environment=bool(cfg.security.can_modify_environment),
                    ),
                )
        
        # Registry for all created tools
        self.registry = ToolRegistry()
    
    def set_input_callback(self, callback: Callable[[str, Optional[list[str]]], str]) -> None:
        """Set the user input callback."""
        self.input_callback = callback
        
        # Update existing user_input tools
        agent_ids = self.team_config.get_agent_ids() if self.team_config else list(_default_agent_permissions().keys())
        for agent_id in agent_ids:
            tool = self.registry.get_tool(agent_id, "user_input")
            if tool and isinstance(tool, UserInputTool):
                tool.set_input_callback(callback)
    
    def set_user_preferences(self, preferences: str) -> None:
        """
        Set user preferences on all user_input tools.
        
        This passes the user's natural language preferences (from consultation)
        to enable contextual autonomy decisions.
        """
        agent_ids = self.team_config.get_agent_ids() if self.team_config else list(_default_agent_permissions().keys())
        for agent_id in agent_ids:
            tool = self.registry.get_tool(agent_id, "user_input")
            if tool and isinstance(tool, UserInputTool):
                tool.set_user_preferences(preferences)
    
    def set_permission_callback(self, callback: Callable[[str], bool]) -> None:
        """Set the permission request callback."""
        self.permission_callback = callback
    
    def _create_tool(self, tool_name: str, agent_id: str) -> Optional[Tool]:
        """Create a single tool instance for an agent."""
        common_kwargs = {
            "agent_id": agent_id,
            "permission_callback": self.permission_callback,
            "context_manager": self.context_manager,
        }
        
        if tool_name == "filesystem":
            return FileSystemTool(
                workspace_root=self.workspace_root,
                **common_kwargs,
            )
        elif tool_name == "code_editor":
            return CodeEditorTool(
                workspace_root=self.workspace_root,
                **common_kwargs,
            )
        elif tool_name == "terminal":
            return TerminalTool(
                workspace_root=self.workspace_root,
                **common_kwargs,
            )
        elif tool_name == "environment":
            return EnvironmentTool(
                workspace_root=self.workspace_root,
                **common_kwargs,
            )
        elif tool_name == "user_input":
            tool = UserInputTool(**common_kwargs)
            if self.input_callback:
                tool.set_input_callback(self.input_callback)
            return tool
        elif tool_name == "web_search":
            return WebSearchTool(**common_kwargs)
        elif tool_name == "pubmed":
            return PubMedTool(**common_kwargs)
        elif tool_name == "arxiv":
            return ArxivTool(**common_kwargs)
        elif tool_name == "citation":
            return CitationTool(**common_kwargs)
        else:
            return None
    
    def create_tools_for_agent(self, agent_id: str) -> dict[str, Tool]:
        """
        Create all tools for a specific agent based on permissions.
        
        Args:
            agent_id: The agent identifier
            
        Returns:
            Dict mapping tool names to Tool instances
        """
        agent_id = agent_id.lower()

        if self.team_config:
            cfg = self.team_config.get_agent(agent_id)
            permissions = {
                "tools": list(cfg.tools) if cfg else [],
                "write_tools": list(cfg.write_tools) if cfg else [],
            }
        else:
            permissions = _default_agent_permissions().get(agent_id, {"tools": [], "write_tools": []})
        
        tools = {}
        for tool_name in permissions["tools"]:
            tool = self._create_tool(tool_name, agent_id)
            if tool:
                tools[tool_name] = tool
                self.registry.register(agent_id, tool)
        
        return tools
    
    def create_all_agent_tools(self) -> dict[str, dict[str, Tool]]:
        """
        Create tools for all agents.
        
        Returns:
            Dict mapping agent_id to dict of tools
        """
        all_tools = {}
        agent_ids = self.team_config.get_agent_ids() if self.team_config else list(_default_agent_permissions().keys())
        for agent_id in agent_ids:
            all_tools[agent_id] = self.create_tools_for_agent(agent_id)
        return all_tools
    
    def get_tool(self, agent_id: str, tool_name: str) -> Optional[Tool]:
        """Get a specific tool for an agent."""
        return self.registry.get_tool(agent_id.lower(), tool_name)
    
    def get_agent_tools(self, agent_id: str) -> dict[str, Tool]:
        """Get all tools for an agent."""
        return self.registry.get_agent_tools(agent_id.lower())
    
    def format_tools_for_prompt(self, agent_id: str) -> str:
        """Format all agent tools for inclusion in prompts."""
        return self.registry.format_tools_for_prompt(agent_id.lower())
    
    def get_tool_documentation(self, agent_id: str) -> str:
        """
        Generate comprehensive tool documentation for an agent.
        
        This is used in the agent's system prompt.
        """
        agent_id = agent_id.lower()
        tools = self.registry.get_agent_tools(agent_id)
        if self.team_config:
            cfg = self.team_config.get_agent(agent_id)
            permissions = {
                "tools": list(cfg.tools) if cfg else [],
                "write_tools": list(cfg.write_tools) if cfg else [],
            }
        else:
            permissions = _default_agent_permissions().get(agent_id, {"tools": [], "write_tools": []})
        
        lines = [
            "## Available Tools",
            "",
            "You have access to the following tools. Use them by outputting a tool call block.",
            "",
        ]
        
        # Document each tool
        for tool_name, tool in sorted(tools.items()):
            is_write_tool = tool_name in permissions.get("write_tools", [])
            
            lines.append(f"### {tool.name}" + (" (WRITE ACCESS)" if is_write_tool else " (READ ONLY)"))
            lines.append(f"{tool.description}")
            lines.append("")
            lines.append("**Actions:**")
            
            for action, desc in tool.get_actions().items():
                lines.append(f"- `{action}`: {desc}")
            
            lines.append("")
        
        # Add tool call format
        lines.extend([
            "## Tool Call Format",
            "",
            "To use a tool, output a JSON block in this format:",
            "",
            "```tool",
            '{"tool": "tool_name", "action": "action_name", "params": {...}}',
            "```",
            "",
            "Wait for the tool result before continuing.",
            "",
        ])
        
        return "\n".join(lines)
