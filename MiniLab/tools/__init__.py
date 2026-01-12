"""
MiniLab Tools Module

Typed tool system with Pydantic models for input/output validation.
All tools integrate with PathGuard for security enforcement.

Now includes VS Code-style patterns:
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
