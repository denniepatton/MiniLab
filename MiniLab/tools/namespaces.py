"""
Tool Namespace Registry.

Provides namespaced access to tools following the minilab_outline.md convention:
- fs.*: File system operations
- search.*: Literature and web search  
- doc.*: Document generation (DOCX/PDF)
- fig.*: Figure generation
- render.*: Markdown rendering to documents
- permission.*: User confirmation prompts
- code.*: Code editing
- user.*: User input
- citation.*: Citation management
- env.*: Environment and terminal

This module creates a unified interface for tool access while maintaining
backward compatibility with the existing tool classes.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from .base import Tool


class ToolNamespace(str, Enum):
    """Tool namespace categories."""
    FS = "fs"
    SEARCH = "search"
    DOC = "doc"
    FIG = "fig"
    RENDER = "render"
    PERMISSION = "permission"
    CODE = "code"
    USER = "user"
    CITATION = "citation"
    ENV = "env"


@dataclass
class NamespacedToolRef:
    """Reference to a tool within a namespace."""
    namespace: ToolNamespace
    action: str
    tool_class: str
    description: str


# Registry mapping namespaced names to tool implementations
TOOL_REGISTRY: dict[str, NamespacedToolRef] = {
    # File system tools (fs.*)
    "fs.read": NamespacedToolRef(
        ToolNamespace.FS, "read", "FileSystemTool",
        "Read file contents"
    ),
    "fs.write": NamespacedToolRef(
        ToolNamespace.FS, "write", "FileSystemTool",
        "Write content to file"
    ),
    "fs.append": NamespacedToolRef(
        ToolNamespace.FS, "append", "FileSystemTool",
        "Append content to file"
    ),
    "fs.list": NamespacedToolRef(
        ToolNamespace.FS, "list", "FileSystemTool",
        "List directory contents"
    ),
    "fs.exists": NamespacedToolRef(
        ToolNamespace.FS, "exists", "FileSystemTool",
        "Check if path exists"
    ),
    "fs.head": NamespacedToolRef(
        ToolNamespace.FS, "head", "FileSystemTool",
        "Read first N lines"
    ),
    "fs.tail": NamespacedToolRef(
        ToolNamespace.FS, "tail", "FileSystemTool",
        "Read last N lines"
    ),
    "fs.search": NamespacedToolRef(
        ToolNamespace.FS, "search", "FileSystemTool",
        "Search within file"
    ),
    "fs.mkdir": NamespacedToolRef(
        ToolNamespace.FS, "create_dir", "FileSystemTool",
        "Create directory"
    ),
    "fs.delete": NamespacedToolRef(
        ToolNamespace.FS, "delete", "FileSystemTool",
        "Delete file or directory"
    ),
    "fs.copy": NamespacedToolRef(
        ToolNamespace.FS, "copy", "FileSystemTool",
        "Copy file"
    ),
    "fs.move": NamespacedToolRef(
        ToolNamespace.FS, "move", "FileSystemTool",
        "Move/rename file"
    ),
    "fs.stats": NamespacedToolRef(
        ToolNamespace.FS, "stats", "FileSystemTool",
        "Get file statistics"
    ),
    
    # Search tools (search.*)
    "search.pubmed": NamespacedToolRef(
        ToolNamespace.SEARCH, "search", "PubMedTool",
        "Search PubMed literature"
    ),
    "search.pubmed_fetch": NamespacedToolRef(
        ToolNamespace.SEARCH, "fetch", "PubMedTool",
        "Fetch PubMed article details"
    ),
    "search.arxiv": NamespacedToolRef(
        ToolNamespace.SEARCH, "search", "ArxivTool",
        "Search arXiv preprints"
    ),
    "search.arxiv_fetch": NamespacedToolRef(
        ToolNamespace.SEARCH, "fetch", "ArxivTool",
        "Fetch arXiv paper details"
    ),
    "search.web": NamespacedToolRef(
        ToolNamespace.SEARCH, "search", "WebSearchTool",
        "Search the web"
    ),
    
    # Document generation tools (doc.*)
    "doc.docx": NamespacedToolRef(
        ToolNamespace.DOC, "create_docx", "DocumentTool",
        "Generate DOCX document"
    ),
    "doc.pdf": NamespacedToolRef(
        ToolNamespace.DOC, "create_pdf", "DocumentTool",
        "Generate PDF document"
    ),
    "doc.to_docx": NamespacedToolRef(
        ToolNamespace.DOC, "markdown_to_docx", "DocumentTool",
        "Convert markdown to DOCX"
    ),
    "doc.to_pdf": NamespacedToolRef(
        ToolNamespace.DOC, "markdown_to_pdf", "DocumentTool",
        "Convert markdown to PDF"
    ),
    
    # Figure tools (fig.*)
    "fig.create": NamespacedToolRef(
        ToolNamespace.FIG, "create", "FigureTool",
        "Create figure from data"
    ),
    "fig.save": NamespacedToolRef(
        ToolNamespace.FIG, "save", "FigureTool",
        "Save figure to file"
    ),
    "fig.render": NamespacedToolRef(
        ToolNamespace.FIG, "render", "FigureTool",
        "Render figure specification"
    ),
    
    # Render tools (render.*)
    "render.markdown": NamespacedToolRef(
        ToolNamespace.RENDER, "render_markdown", "RenderTool",
        "Render markdown"
    ),
    "render.preview": NamespacedToolRef(
        ToolNamespace.RENDER, "preview", "RenderTool",
        "Preview rendered document"
    ),
    
    # Permission tools (permission.*)
    "permission.confirm": NamespacedToolRef(
        ToolNamespace.PERMISSION, "confirm", "PermissionTool",
        "Request user confirmation"
    ),
    "permission.approve": NamespacedToolRef(
        ToolNamespace.PERMISSION, "approve", "PermissionTool",
        "Request approval for action"
    ),
    
    # Code tools (code.*)
    "code.edit": NamespacedToolRef(
        ToolNamespace.CODE, "edit", "CodeEditorTool",
        "Edit code file"
    ),
    "code.insert": NamespacedToolRef(
        ToolNamespace.CODE, "insert", "CodeEditorTool",
        "Insert code at position"
    ),
    "code.replace": NamespacedToolRef(
        ToolNamespace.CODE, "replace", "CodeEditorTool",
        "Replace code section"
    ),
    "code.search": NamespacedToolRef(
        ToolNamespace.CODE, "search", "CodeEditorTool",
        "Search in code files"
    ),
    
    # User input tools (user.*)
    "user.ask": NamespacedToolRef(
        ToolNamespace.USER, "ask", "UserInputTool",
        "Ask user a question"
    ),
    "user.confirm": NamespacedToolRef(
        ToolNamespace.USER, "confirm", "UserInputTool",
        "Get yes/no confirmation"
    ),
    "user.select": NamespacedToolRef(
        ToolNamespace.USER, "select", "UserInputTool",
        "Present options to user"
    ),
    
    # Citation tools (citation.*)
    "citation.add": NamespacedToolRef(
        ToolNamespace.CITATION, "add", "CitationTool",
        "Add citation to bibliography"
    ),
    "citation.format": NamespacedToolRef(
        ToolNamespace.CITATION, "format", "CitationTool",
        "Format citation string"
    ),
    "citation.lookup": NamespacedToolRef(
        ToolNamespace.CITATION, "lookup", "CitationTool",
        "Look up citation details"
    ),
    
    # Environment tools (env.*)
    "env.run": NamespacedToolRef(
        ToolNamespace.ENV, "run", "TerminalTool",
        "Run terminal command"
    ),
    "env.python": NamespacedToolRef(
        ToolNamespace.ENV, "run_python", "TerminalTool",
        "Run Python script"
    ),
    "env.check": NamespacedToolRef(
        ToolNamespace.ENV, "check", "EnvironmentTool",
        "Check environment"
    ),
    "env.install": NamespacedToolRef(
        ToolNamespace.ENV, "install", "EnvironmentTool",
        "Install package"
    ),
}


def get_tool_for_namespace(namespaced_name: str) -> Optional[NamespacedToolRef]:
    """
    Get tool reference for a namespaced tool name.
    
    Args:
        namespaced_name: Name like "fs.read" or "search.pubmed"
        
    Returns:
        NamespacedToolRef if found, None otherwise
    """
    return TOOL_REGISTRY.get(namespaced_name)


def list_namespace_tools(namespace: ToolNamespace) -> list[str]:
    """
    List all tools in a namespace.
    
    Args:
        namespace: The namespace to list
        
    Returns:
        List of tool names in that namespace
    """
    prefix = f"{namespace.value}."
    return [name for name in TOOL_REGISTRY.keys() if name.startswith(prefix)]


def get_all_namespaces() -> list[ToolNamespace]:
    """Get all available namespaces."""
    return list(ToolNamespace)


def resolve_tool_action(namespaced_name: str) -> tuple[str, str]:
    """
    Resolve a namespaced tool name to tool class and action.
    
    Args:
        namespaced_name: Name like "fs.read"
        
    Returns:
        Tuple of (tool_class_name, action_name)
        
    Raises:
        KeyError: If tool not found
    """
    ref = TOOL_REGISTRY.get(namespaced_name)
    if not ref:
        raise KeyError(f"Unknown tool: {namespaced_name}")
    return ref.tool_class, ref.action


class NamespacedToolProxy:
    """
    Proxy object for accessing tools through namespaced interface.
    
    Usage:
        tools = NamespacedToolProxy(agent_id="bohr")
        result = await tools.fs.read(path="file.txt")
    """
    
    def __init__(self, agent_id: str, **kwargs: Any):
        self.agent_id = agent_id
        self.kwargs = kwargs
        self._tool_cache: dict[str, Tool] = {}
    
    def __getattr__(self, namespace: str) -> "_NamespaceAccessor":
        """Get a namespace accessor."""
        try:
            ns = ToolNamespace(namespace)
            return _NamespaceAccessor(self, ns)
        except ValueError:
            raise AttributeError(f"Unknown namespace: {namespace}")
    
    def _get_tool(self, tool_class: str) -> "Tool":
        """Get or create a tool instance."""
        if tool_class not in self._tool_cache:
            from . import (
                FileSystemTool, CodeEditorTool, TerminalTool,
                EnvironmentTool, UserInputTool, WebSearchTool,
                PubMedTool, ArxivTool, CitationTool,
            )
            
            class_map = {
                "FileSystemTool": FileSystemTool,
                "CodeEditorTool": CodeEditorTool,
                "TerminalTool": TerminalTool,
                "EnvironmentTool": EnvironmentTool,
                "UserInputTool": UserInputTool,
                "WebSearchTool": WebSearchTool,
                "PubMedTool": PubMedTool,
                "ArxivTool": ArxivTool,
                "CitationTool": CitationTool,
            }
            
            cls = class_map.get(tool_class)
            if cls:
                self._tool_cache[tool_class] = cls(
                    agent_id=self.agent_id,
                    **self.kwargs
                )
        
        return self._tool_cache.get(tool_class)


class _NamespaceAccessor:
    """Accessor for a specific namespace."""
    
    def __init__(self, proxy: NamespacedToolProxy, namespace: ToolNamespace):
        self._proxy = proxy
        self._namespace = namespace
    
    def __getattr__(self, action: str) -> "_ToolMethod":
        """Get a tool method."""
        full_name = f"{self._namespace.value}.{action}"
        ref = TOOL_REGISTRY.get(full_name)
        if not ref:
            raise AttributeError(f"Unknown tool action: {full_name}")
        return _ToolMethod(self._proxy, ref)


class _ToolMethod:
    """Callable wrapper for a tool action."""
    
    def __init__(self, proxy: NamespacedToolProxy, ref: NamespacedToolRef):
        self._proxy = proxy
        self._ref = ref
    
    async def __call__(self, **kwargs: Any) -> Any:
        """Execute the tool action."""
        tool = self._proxy._get_tool(self._ref.tool_class)
        if tool:
            return await tool.execute(self._ref.action, kwargs)
        raise RuntimeError(f"Tool not available: {self._ref.tool_class}")


# Convenience function to create namespaced tool proxy
def get_tools(agent_id: str, **kwargs: Any) -> NamespacedToolProxy:
    """
    Get a namespaced tool proxy for an agent.
    
    Args:
        agent_id: The agent ID
        **kwargs: Additional arguments passed to tools
        
    Returns:
        NamespacedToolProxy instance
        
    Example:
        tools = get_tools("bohr")
        content = await tools.fs.read(path="data.csv")
        results = await tools.search.pubmed(query="cancer genomics")
    """
    return NamespacedToolProxy(agent_id, **kwargs)
