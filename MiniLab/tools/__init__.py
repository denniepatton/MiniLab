from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

__all__ = [
    "Tool", 
    "DualModeFileSystemTool",
    "EnvironmentTool",
    "WebSearchTool",
    "ArxivSearchTool", 
    "PubMedSearchTool",
    "CodeEditorTool",
    "TerminalTool",
    "GitTool",
    "UserInputTool",
]


class Tool(ABC):
    """
    Abstract base class for agent tools.
    Each tool provides a specific capability (filesystem, web search, code editing, etc.)
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with the given parameters.
        Returns a dict with results and any metadata.
        """
        pass

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"


# Import active tools for convenience
from .filesystem_dual import DualModeFileSystemTool
from .environment import EnvironmentTool
from .web_search import WebSearchTool, ArxivSearchTool, PubMedSearchTool
from .code_editor import CodeEditorTool
from .system_tools import TerminalTool, GitTool
from .user_input import UserInputTool
