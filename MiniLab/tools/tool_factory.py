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

from ..security import PathGuard
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


# Define which tools each agent can use (read-only tools are shared, write tools vary)
AGENT_TOOL_PERMISSIONS = {
    # Synthesis Guild
    "bohr": {
        "tools": ["filesystem", "user_input", "web_search", "pubmed", "arxiv", "citation", "environment"],
        "write_tools": [],  # Bohr delegates writing to specialists
    },
    "gould": {
        "tools": ["filesystem", "user_input", "web_search", "pubmed", "arxiv", "citation"],
        "write_tools": ["filesystem", "citation"],  # Can write documents, bibliography
    },
    "farber": {
        "tools": ["filesystem", "user_input", "web_search", "pubmed", "citation"],
        "write_tools": ["filesystem"],  # Can write reviews
    },
    
    # Theory Guild
    "feynman": {
        "tools": ["filesystem", "user_input", "web_search", "pubmed", "arxiv"],
        "write_tools": ["filesystem"],  # Can write notes
    },
    "shannon": {
        "tools": ["filesystem", "user_input", "web_search", "arxiv"],
        "write_tools": ["filesystem"],  # Can write notes
    },
    "greider": {
        "tools": ["filesystem", "user_input", "web_search", "pubmed"],
        "write_tools": ["filesystem"],  # Can write notes
    },
    
    # Implementation Guild
    "dayhoff": {
        "tools": ["filesystem", "user_input", "web_search", "pubmed"],
        "write_tools": ["filesystem"],  # Can write execution plans
    },
    "hinton": {
        "tools": ["filesystem", "code_editor", "terminal", "user_input", "environment"],
        "write_tools": ["filesystem", "code_editor", "terminal"],  # Full code access
    },
    "bayes": {
        "tools": ["filesystem", "code_editor", "terminal", "user_input"],
        "write_tools": ["filesystem", "code_editor", "terminal"],  # Statistical code access
    },
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
        
        # Initialize PathGuard singleton
        PathGuard.reset()
        self.path_guard = PathGuard(workspace_root)
        
        # Registry for all created tools
        self.registry = ToolRegistry()
    
    def set_input_callback(self, callback: Callable[[str, Optional[list[str]]], str]) -> None:
        """Set the user input callback."""
        self.input_callback = callback
        
        # Update existing user_input tools
        for agent_id in AGENT_TOOL_PERMISSIONS:
            tool = self.registry.get_tool(agent_id, "user_input")
            if tool and isinstance(tool, UserInputTool):
                tool.set_input_callback(callback)
    
    def set_permission_callback(self, callback: Callable[[str], bool]) -> None:
        """Set the permission request callback."""
        self.permission_callback = callback
    
    def _create_tool(self, tool_name: str, agent_id: str) -> Optional[Tool]:
        """Create a single tool instance for an agent."""
        common_kwargs = {
            "agent_id": agent_id,
            "permission_callback": self.permission_callback,
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
        permissions = AGENT_TOOL_PERMISSIONS.get(agent_id, {"tools": [], "write_tools": []})
        
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
        for agent_id in AGENT_TOOL_PERMISSIONS:
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
        permissions = AGENT_TOOL_PERMISSIONS.get(agent_id, {"tools": [], "write_tools": []})
        
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
