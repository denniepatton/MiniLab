"""
MiniLab Tools Module

Typed tool system with Pydantic models for input/output validation.
All tools integrate with PathGuard for security enforcement.

Tool Namespaces (aligned with minilab_outline.md):
- fs.*: File system operations
- search.*: Literature and web search
- doc.*: Document generation (DOCX/PDF)
- fig.*: Figure generation
- render.*: Markdown rendering
- permission.*: User confirmation prompts
- code.*: Code editing
- user.*: User input
- citation.*: Citation management
- env.*: Environment and terminal

VS Code-style patterns:
- PreparedInvocation for two-phase tool execution
- ResponseStream for typed progress reporting  
- EditSession/WorkspaceEdit for atomic batched edits
- ToolSelector for tool enablement control
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

# New tools for Phase 2
from .document import DocumentTool
from .figure import FigureTool
from .permission import PermissionTool

# Namespace registry
from .namespaces import (
    ToolNamespace,
    NamespacedToolRef,
    NamespacedToolProxy,
    TOOL_REGISTRY,
    get_tool_for_namespace,
    list_namespace_tools,
    get_all_namespaces,
    resolve_tool_action,
    get_tools,
)

# VS Code-style patterns
from .prepared_invocation import (
    PreparedInvocation,
    ConfirmationLevel,
    ConfirmationMessage,
    get_default_confirmation_level,
    DESTRUCTIVE_ACTIONS,
    MODIFYING_ACTIONS,
    READONLY_ACTIONS,
)
from .response_stream import (
    ResponseStream,
    ConsoleResponseStream,
    BufferedResponseStream,
    NullResponseStream,
    CompositeResponseStream,
    ProgressKind,
    DiagnosticSeverity,
    FileReference,
    FileTreeNode,
    CodeBlock,
    StreamEvent,
)
from .edit_session import (
    EditSession,
    WorkspaceEdit,
    TextEdit,
    FileEdit,
    FileEditEntry,
    EditKind,
    EditState,
    Position,
    Range,
    get_edit_session,
    create_edit_session,
    close_edit_session,
)
from .tool_selector import (
    ToolSelector,
    ToolSelection,
    ToolPreset,
    get_tool_selector,
    set_tool_selector,
    TOOL_CATEGORIES,
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
    # New tools
    "DocumentTool",
    "FigureTool",
    "PermissionTool",
    # Namespace system
    "ToolNamespace",
    "NamespacedToolRef",
    "NamespacedToolProxy",
    "TOOL_REGISTRY",
    "get_tool_for_namespace",
    "list_namespace_tools",
    "get_all_namespaces",
    "resolve_tool_action",
    "get_tools",
    # PreparedInvocation (VS Code pattern)
    "PreparedInvocation",
    "ConfirmationLevel",
    "ConfirmationMessage",
    "get_default_confirmation_level",
    "DESTRUCTIVE_ACTIONS",
    "MODIFYING_ACTIONS",
    "READONLY_ACTIONS",
    # ResponseStream (VS Code pattern)
    "ResponseStream",
    "ConsoleResponseStream",
    "BufferedResponseStream",
    "NullResponseStream",
    "CompositeResponseStream",
    "ProgressKind",
    "DiagnosticSeverity",
    "FileReference",
    "FileTreeNode",
    "CodeBlock",
    "StreamEvent",
    # EditSession (VS Code pattern)
    "EditSession",
    "WorkspaceEdit",
    "TextEdit",
    "FileEdit",
    "FileEditEntry",
    "EditKind",
    "EditState",
    "Position",
    "Range",
    "get_edit_session",
    "create_edit_session",
    "close_edit_session",
    # ToolSelector (VS Code pattern)
    "ToolSelector",
    "ToolSelection",
    "ToolPreset",
    "get_tool_selector",
    "set_tool_selector",
    "TOOL_CATEGORIES",
]
