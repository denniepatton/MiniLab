"""
Tool Selector - VS Code-style tool enablement control.

Allows the orchestrator or user to control which tools are available
to agents at runtime. This mirrors VS Code's UserSelectedTools pattern.

Features:
- Enable/disable tools per agent
- Tool presets for different modes
- Runtime tool availability control
- Tool usage tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Tool


class ToolPreset(Enum):
    """Predefined tool presets for common modes."""
    ALL = "all"                     # All tools enabled
    READONLY = "readonly"           # Only read operations
    SAFE = "safe"                   # No destructive operations
    MINIMAL = "minimal"             # Basic set only
    RESEARCH = "research"           # Research-focused tools
    CODING = "coding"               # Code-focused tools
    ANALYSIS = "analysis"           # Data analysis tools
    NONE = "none"                   # No tools enabled


# Tool categories for preset selection
TOOL_CATEGORIES = {
    "readonly": {
        "filesystem": ["read", "head", "list", "stats", "search"],
        "code_editor": ["view", "check_syntax"],
        "web_search": ["search"],
        "pubmed": ["search"],
        "arxiv": ["search"],
        "environment": ["check"],
    },
    "safe": {
        "filesystem": ["read", "head", "list", "stats", "search", "write", "append", "create_dir"],
        "code_editor": ["view", "check_syntax", "create", "insert", "replace", "replace_text", "run"],
        "terminal": ["execute"],  # But with restrictions
        "web_search": ["search"],
        "pubmed": ["search"],
        "arxiv": ["search"],
        "environment": ["check", "install"],
    },
    "research": {
        "filesystem": ["read", "head", "list", "search", "write"],
        "web_search": ["search"],
        "pubmed": ["search"],
        "arxiv": ["search"],
        "citation": ["format", "lookup"],
    },
    "coding": {
        "filesystem": ["read", "head", "list", "stats", "search", "write", "create_dir"],
        "code_editor": ["view", "create", "insert", "replace", "delete_lines", "replace_text", "check_syntax", "run"],
        "terminal": ["execute", "run_script"],
        "environment": ["check", "install"],
    },
    "analysis": {
        "filesystem": ["read", "head", "list", "stats", "search", "write", "append"],
        "code_editor": ["view", "create", "insert", "replace", "check_syntax", "run"],
        "terminal": ["execute"],
        "environment": ["check", "install"],
    },
}


@dataclass
class ToolSelection:
    """
    Selection state for tools.
    
    Tracks which tools and actions are enabled for an agent.
    """
    # Explicit enablement: tool_name -> list of enabled actions (None = all)
    enabled_tools: dict[str, Optional[list[str]]] = field(default_factory=dict)
    
    # Explicit disablement (takes precedence)
    disabled_tools: set[str] = field(default_factory=set)
    disabled_actions: dict[str, set[str]] = field(default_factory=dict)
    
    # Preset to use as base
    preset: ToolPreset = ToolPreset.ALL
    
    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled."""
        if tool_name in self.disabled_tools:
            return False
        
        if self.preset == ToolPreset.NONE:
            return tool_name in self.enabled_tools
        
        if self.preset == ToolPreset.ALL:
            return True
        
        # Check preset category
        preset_tools = TOOL_CATEGORIES.get(self.preset.value, {})
        return tool_name in preset_tools or tool_name in self.enabled_tools
    
    def is_action_enabled(self, tool_name: str, action: str) -> bool:
        """Check if a specific action is enabled."""
        if not self.is_tool_enabled(tool_name):
            return False
        
        # Check explicit disablement
        if tool_name in self.disabled_actions:
            if action in self.disabled_actions[tool_name]:
                return False
        
        # Check explicit enablement
        if tool_name in self.enabled_tools:
            enabled_actions = self.enabled_tools[tool_name]
            if enabled_actions is not None:
                return action in enabled_actions
        
        # Check preset
        if self.preset in (ToolPreset.ALL, ToolPreset.NONE):
            return True
        
        preset_tools = TOOL_CATEGORIES.get(self.preset.value, {})
        if tool_name in preset_tools:
            allowed_actions = preset_tools[tool_name]
            return action in allowed_actions
        
        return True
    
    def enable_tool(self, tool_name: str, actions: Optional[list[str]] = None) -> None:
        """Enable a tool (optionally with specific actions)."""
        self.disabled_tools.discard(tool_name)
        self.enabled_tools[tool_name] = actions
    
    def disable_tool(self, tool_name: str) -> None:
        """Disable a tool entirely."""
        self.disabled_tools.add(tool_name)
        if tool_name in self.enabled_tools:
            del self.enabled_tools[tool_name]
    
    def disable_action(self, tool_name: str, action: str) -> None:
        """Disable a specific action."""
        if tool_name not in self.disabled_actions:
            self.disabled_actions[tool_name] = set()
        self.disabled_actions[tool_name].add(action)
    
    def enable_action(self, tool_name: str, action: str) -> None:
        """Enable a specific action."""
        if tool_name in self.disabled_actions:
            self.disabled_actions[tool_name].discard(action)
    
    def to_dict(self) -> dict:
        return {
            "enabled_tools": self.enabled_tools,
            "disabled_tools": list(self.disabled_tools),
            "disabled_actions": {k: list(v) for k, v in self.disabled_actions.items()},
            "preset": self.preset.value,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> ToolSelection:
        return cls(
            enabled_tools=data.get("enabled_tools", {}),
            disabled_tools=set(data.get("disabled_tools", [])),
            disabled_actions={k: set(v) for k, v in data.get("disabled_actions", {}).items()},
            preset=ToolPreset(data.get("preset", "all")),
        )
    
    @classmethod
    def from_preset(cls, preset: ToolPreset) -> ToolSelection:
        """Create a selection from a preset."""
        return cls(preset=preset)
    
    @classmethod
    def readonly(cls) -> ToolSelection:
        """Create a readonly selection."""
        return cls(preset=ToolPreset.READONLY)
    
    @classmethod
    def safe(cls) -> ToolSelection:
        """Create a safe selection (no destructive ops)."""
        return cls(preset=ToolPreset.SAFE)


class ToolSelector:
    """
    Manages tool availability for agents.
    
    Provides runtime control over which tools agents can use,
    similar to VS Code's UserSelectedTools.
    
    Usage:
        selector = ToolSelector()
        
        # Set per-agent selection
        selector.set_selection("hinton", ToolSelection.from_preset(ToolPreset.ANALYSIS))
        
        # Check if action allowed
        if selector.is_action_allowed("hinton", "filesystem", "delete"):
            # Allow the action
            
        # Get filtered tools for agent
        tools = selector.get_available_tools("hinton", all_tools)
    """
    
    def __init__(self, default_preset: ToolPreset = ToolPreset.ALL):
        self._selections: dict[str, ToolSelection] = {}
        self._default_selection = ToolSelection(preset=default_preset)
        
        # Usage tracking
        self._usage_counts: dict[str, dict[str, int]] = {}  # agent_id -> tool.action -> count
        
        # Callbacks
        self.on_tool_blocked: Optional[Callable[[str, str, str], None]] = None  # (agent_id, tool, action)
    
    def get_selection(self, agent_id: str) -> ToolSelection:
        """Get the tool selection for an agent."""
        return self._selections.get(agent_id, self._default_selection)
    
    def set_selection(self, agent_id: str, selection: ToolSelection) -> None:
        """Set the tool selection for an agent."""
        self._selections[agent_id] = selection
    
    def set_preset(self, agent_id: str, preset: ToolPreset) -> None:
        """Set an agent's selection to a preset."""
        self._selections[agent_id] = ToolSelection.from_preset(preset)
    
    def clear_selection(self, agent_id: str) -> None:
        """Clear an agent's selection (revert to default)."""
        if agent_id in self._selections:
            del self._selections[agent_id]
    
    def is_tool_allowed(self, agent_id: str, tool_name: str) -> bool:
        """Check if a tool is allowed for an agent."""
        selection = self.get_selection(agent_id)
        return selection.is_tool_enabled(tool_name)
    
    def is_action_allowed(self, agent_id: str, tool_name: str, action: str) -> bool:
        """Check if a specific action is allowed for an agent."""
        selection = self.get_selection(agent_id)
        allowed = selection.is_action_enabled(tool_name, action)
        
        if not allowed and self.on_tool_blocked:
            self.on_tool_blocked(agent_id, tool_name, action)
        
        return allowed
    
    def get_available_tools(
        self,
        agent_id: str,
        all_tools: dict[str, Any],  # Tool instances
    ) -> dict[str, Any]:
        """
        Filter tools to only those available for an agent.
        
        Args:
            agent_id: Agent identifier
            all_tools: Dict of all available tools
            
        Returns:
            Filtered dict of allowed tools
        """
        selection = self.get_selection(agent_id)
        available = {}
        
        for tool_name, tool in all_tools.items():
            if selection.is_tool_enabled(tool_name):
                available[tool_name] = tool
        
        return available
    
    def get_available_actions(
        self,
        agent_id: str,
        tool_name: str,
        all_actions: list[str],
    ) -> list[str]:
        """
        Filter actions to only those available for an agent.
        
        Args:
            agent_id: Agent identifier
            tool_name: Tool name
            all_actions: List of all available actions
            
        Returns:
            Filtered list of allowed actions
        """
        selection = self.get_selection(agent_id)
        return [
            action for action in all_actions
            if selection.is_action_enabled(tool_name, action)
        ]
    
    def record_usage(self, agent_id: str, tool_name: str, action: str) -> None:
        """Record tool usage for tracking."""
        if agent_id not in self._usage_counts:
            self._usage_counts[agent_id] = {}
        
        key = f"{tool_name}.{action}"
        self._usage_counts[agent_id][key] = self._usage_counts[agent_id].get(key, 0) + 1
    
    def get_usage_stats(self, agent_id: Optional[str] = None) -> dict:
        """Get usage statistics."""
        if agent_id:
            return self._usage_counts.get(agent_id, {})
        return self._usage_counts.copy()
    
    def enable_tool(
        self,
        agent_id: str,
        tool_name: str,
        actions: Optional[list[str]] = None,
    ) -> None:
        """Enable a tool for an agent."""
        if agent_id not in self._selections:
            self._selections[agent_id] = ToolSelection()
        self._selections[agent_id].enable_tool(tool_name, actions)
    
    def disable_tool(self, agent_id: str, tool_name: str) -> None:
        """Disable a tool for an agent."""
        if agent_id not in self._selections:
            self._selections[agent_id] = ToolSelection()
        self._selections[agent_id].disable_tool(tool_name)
    
    def disable_action(self, agent_id: str, tool_name: str, action: str) -> None:
        """Disable a specific action for an agent."""
        if agent_id not in self._selections:
            self._selections[agent_id] = ToolSelection()
        self._selections[agent_id].disable_action(tool_name, action)
    
    def enable_destructive_for_agent(self, agent_id: str) -> None:
        """Enable destructive operations for an agent."""
        if agent_id not in self._selections:
            self._selections[agent_id] = ToolSelection()
        # Remove filesystem.delete from disabled
        self._selections[agent_id].enable_action("filesystem", "delete")
        self._selections[agent_id].enable_action("filesystem", "move")
        self._selections[agent_id].enable_action("code_editor", "delete_lines")
    
    def disable_destructive_for_agent(self, agent_id: str) -> None:
        """Disable destructive operations for an agent."""
        if agent_id not in self._selections:
            self._selections[agent_id] = ToolSelection()
        self._selections[agent_id].disable_action("filesystem", "delete")
        self._selections[agent_id].disable_action("filesystem", "move")
        self._selections[agent_id].disable_action("code_editor", "delete_lines")
    
    def to_dict(self) -> dict:
        return {
            "selections": {k: v.to_dict() for k, v in self._selections.items()},
            "default_preset": self._default_selection.preset.value,
            "usage_counts": self._usage_counts,
        }


# Global selector instance
_global_selector: Optional[ToolSelector] = None


def get_tool_selector() -> ToolSelector:
    """Get the global tool selector instance."""
    global _global_selector
    if _global_selector is None:
        _global_selector = ToolSelector()
    return _global_selector


def set_tool_selector(selector: ToolSelector) -> None:
    """Set the global tool selector instance."""
    global _global_selector
    _global_selector = selector
