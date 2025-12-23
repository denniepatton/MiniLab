"""
MiniLab Tools Module

Typed tool system with Pydantic models for input/output validation.
All tools integrate with PathGuard for security enforcement.

New in 0.4.0:
- ToolGateway: Centralized dispatch with permission checking
- MCP adapters: Integration with Model Context Protocol
- Skill packs: Tool groupings by domain
"""

from .base import Tool, ToolInput, ToolOutput, ToolError
from .filesystem import FileSystemTool
from .code_editor import CodeEditorTool
from .terminal import TerminalTool
from .environment import EnvironmentTool
from .user_input import UserInputTool
from .web_search import WebSearchTool
from .pubmed import PubMedTool
from .arxiv import ArxivTool
from .citation import CitationTool
from .tool_factory import ToolFactory
from .gateway import (
    ToolGateway,
    ToolRegistry,
    ToolScope,
    ToolPermission,
    ToolCall,
    SkillPack,
    SKILL_PACKS,
)
from .mcp_adapter import (
    MCPServer,
    MCPClient,
    MCPToolAdapter,
    MCPToolBridge,
    MCPToolSchema,
    MCPRequest,
    MCPResponse,
    MCPError,
)

__all__ = [
    # Base classes
    "Tool",
    "ToolInput",
    "ToolOutput",
    "ToolError",
    # Tools
    "FileSystemTool",
    "CodeEditorTool",
    "TerminalTool",
    "EnvironmentTool",
    "UserInputTool",
    "WebSearchTool",
    "PubMedTool",
    "ArxivTool",
    "CitationTool",
    "ToolFactory",
    # Gateway
    "ToolGateway",
    "ToolRegistry",
    "ToolScope",
    "ToolPermission",
    "ToolCall",
    "SkillPack",
    "SKILL_PACKS",
    # MCP
    "MCPServer",
    "MCPClient",
    "MCPToolAdapter",
    "MCPToolBridge",
    "MCPToolSchema",
    "MCPRequest",
    "MCPResponse",
    "MCPError",
]
