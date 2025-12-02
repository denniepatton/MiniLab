from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

__all__ = ["Tool", "FileSystemTool"]


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
