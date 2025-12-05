from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

__all__ = [
    "Tool", 
    "DualModeFileSystemTool",
    "FileSystemTool",
    "EnvironmentTool",
    "WebSearchTool",
    "ArxivSearchTool", 
    "PubMedSearchTool",
    "CodeEditorTool",
    "TerminalTool",
]


class Tool(ABC):
    """
    Abstract base class for agent tools.
    Each tool provides a specific capability (web search, Zotero access, etc.)
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


# Import tools for convenience
from .filesystem_dual import DualModeFileSystemTool
from .filesystem import FileSystemTool
from .environment import EnvironmentTool
from .web_search import WebSearchTool, ArxivSearchTool, PubMedSearchTool
from .code_editor import CodeEditorTool, TerminalTool
