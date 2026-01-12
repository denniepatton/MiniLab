"""
Response Stream - VS Code-style typed progress reporting.

Provides a structured way for agents and tools to report progress,
replacing simple string callbacks with typed methods.

Mirrors VS Code's ChatResponseStream interface with methods like:
- markdown() - Formatted text content
- progress() - Status/spinner updates
- anchor() / reference() - File/symbol references
- filetree() - Directory structure visualization
- warning() / error() - Diagnostic messages
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union
from datetime import datetime


class ProgressKind(Enum):
    """Type of progress update."""
    STATUS = "status"           # General status message
    SPINNER = "spinner"         # Activity indicator
    PERCENTAGE = "percentage"   # Numeric progress (0-100)
    STEP = "step"               # Step X of Y


class DiagnosticSeverity(Enum):
    """Severity level for diagnostics."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class FileReference:
    """Reference to a file location."""
    path: Path
    line: Optional[int] = None
    end_line: Optional[int] = None
    column: Optional[int] = None
    label: Optional[str] = None
    
    def to_markdown(self) -> str:
        """Convert to markdown link format."""
        display = self.label or self.path.name
        if self.line:
            if self.end_line and self.end_line != self.line:
                return f"[{display}]({self.path}#L{self.line}-L{self.end_line})"
            return f"[{display}]({self.path}#L{self.line})"
        return f"[{display}]({self.path})"


@dataclass
class FileTreeNode:
    """Node in a file tree visualization."""
    name: str
    is_directory: bool = False
    children: list[FileTreeNode] = field(default_factory=list)
    metadata: Optional[str] = None  # e.g., file size, status
    
    def to_text(self, prefix: str = "", is_last: bool = True) -> str:
        """Render as text tree."""
        connector = "└── " if is_last else "├── "
        icon = "📁 " if self.is_directory else "📄 "
        suffix = f" ({self.metadata})" if self.metadata else ""
        
        lines = [f"{prefix}{connector}{icon}{self.name}{suffix}"]
        
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.children):
            is_last_child = i == len(self.children) - 1
            lines.append(child.to_text(child_prefix, is_last_child))
        
        return "\n".join(lines)


@dataclass
class CodeBlock:
    """A code block with optional metadata."""
    code: str
    language: str = "python"
    filename: Optional[str] = None
    start_line: Optional[int] = None
    is_diff: bool = False


@dataclass 
class StreamEvent:
    """
    A single event in the response stream.
    
    Events are timestamped and typed for replay/logging.
    """
    kind: str
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "content": self.content if isinstance(self.content, (str, int, float, bool, dict, list)) else str(self.content),
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
        }


class ResponseStream(ABC):
    """
    Abstract base for typed response streaming.
    
    Concrete implementations can:
    - Write to console (ConsoleResponseStream)
    - Write to transcript (TranscriptResponseStream)
    - Collect for later (BufferedResponseStream)
    - Send to UI (UIResponseStream)
    """
    
    def __init__(self, agent_id: Optional[str] = None):
        self.agent_id = agent_id
        self._events: list[StreamEvent] = []
    
    def _record_event(self, kind: str, content: Any) -> StreamEvent:
        """Record an event and return it."""
        event = StreamEvent(kind=kind, content=content, agent_id=self.agent_id)
        self._events.append(event)
        return event
    
    @property
    def events(self) -> list[StreamEvent]:
        """Get all recorded events."""
        return self._events.copy()
    
    # --- Core content methods ---
    
    @abstractmethod
    def markdown(self, content: str) -> ResponseStream:
        """
        Push markdown content.
        
        Args:
            content: Markdown-formatted text
            
        Returns:
            self for chaining
        """
        pass
    
    @abstractmethod
    def text(self, content: str) -> ResponseStream:
        """
        Push plain text content.
        
        Args:
            content: Plain text
            
        Returns:
            self for chaining
        """
        pass
    
    # --- Progress methods ---
    
    @abstractmethod
    def progress(
        self,
        message: str,
        kind: ProgressKind = ProgressKind.STATUS,
        current: Optional[int] = None,
        total: Optional[int] = None,
    ) -> ResponseStream:
        """
        Report progress.
        
        Args:
            message: Progress message
            kind: Type of progress indicator
            current: Current step (for STEP/PERCENTAGE)
            total: Total steps (for STEP kind)
            
        Returns:
            self for chaining
        """
        pass
    
    # --- Reference methods ---
    
    @abstractmethod
    def reference(
        self,
        path: Union[str, Path],
        line: Optional[int] = None,
        end_line: Optional[int] = None,
        label: Optional[str] = None,
    ) -> ResponseStream:
        """
        Add a file reference.
        
        Args:
            path: File path
            line: Optional line number
            end_line: Optional end line for range
            label: Display label (defaults to filename)
            
        Returns:
            self for chaining
        """
        pass
    
    @abstractmethod
    def filetree(
        self,
        root: Union[str, Path, FileTreeNode],
        label: Optional[str] = None,
    ) -> ResponseStream:
        """
        Display a file tree structure.
        
        Args:
            root: Root path or pre-built FileTreeNode
            label: Optional label for the tree
            
        Returns:
            self for chaining
        """
        pass
    
    # --- Code methods ---
    
    @abstractmethod
    def code(
        self,
        code: str,
        language: str = "python",
        filename: Optional[str] = None,
    ) -> ResponseStream:
        """
        Display a code block.
        
        Args:
            code: The code content
            language: Language for syntax highlighting
            filename: Optional filename for context
            
        Returns:
            self for chaining
        """
        pass
    
    @abstractmethod
    def diff(
        self,
        diff_content: str,
        filename: Optional[str] = None,
    ) -> ResponseStream:
        """
        Display a diff.
        
        Args:
            diff_content: Unified diff format
            filename: File being diffed
            
        Returns:
            self for chaining
        """
        pass
    
    # --- Diagnostic methods ---
    
    @abstractmethod
    def warning(self, message: str, source: Optional[str] = None) -> ResponseStream:
        """
        Display a warning message.
        
        Args:
            message: Warning text
            source: Optional source/location
            
        Returns:
            self for chaining
        """
        pass
    
    @abstractmethod
    def error(self, message: str, source: Optional[str] = None) -> ResponseStream:
        """
        Display an error message.
        
        Args:
            message: Error text
            source: Optional source/location
            
        Returns:
            self for chaining
        """
        pass
    
    @abstractmethod
    def info(self, message: str) -> ResponseStream:
        """
        Display an info message.
        
        Args:
            message: Info text
            
        Returns:
            self for chaining
        """
        pass
    
    # --- Interactive methods ---
    
    @abstractmethod
    def button(
        self,
        label: str,
        action: str,
        data: Optional[dict] = None,
    ) -> ResponseStream:
        """
        Display an interactive button.
        
        Args:
            label: Button text
            action: Action identifier when clicked
            data: Optional data to pass with action
            
        Returns:
            self for chaining
        """
        pass


class ConsoleResponseStream(ResponseStream):
    """
    Response stream that writes to console.
    
    Used for CLI-based agent interactions.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        use_rich: bool = True,
    ):
        super().__init__(agent_id)
        self.use_rich = use_rich
        try:
            from rich.console import Console
            from rich.markdown import Markdown
            from rich.syntax import Syntax
            from rich.tree import Tree
            from rich.panel import Panel
            self._console = Console()
            self._rich_available = True
        except ImportError:
            self._rich_available = False
    
    def markdown(self, content: str) -> ResponseStream:
        self._record_event("markdown", content)
        if self._rich_available and self.use_rich:
            from rich.markdown import Markdown
            self._console.print(Markdown(content))
        else:
            print(content)
        return self
    
    def text(self, content: str) -> ResponseStream:
        self._record_event("text", content)
        print(content)
        return self
    
    def progress(
        self,
        message: str,
        kind: ProgressKind = ProgressKind.STATUS,
        current: Optional[int] = None,
        total: Optional[int] = None,
    ) -> ResponseStream:
        self._record_event("progress", {
            "message": message,
            "kind": kind.value,
            "current": current,
            "total": total,
        })
        
        if kind == ProgressKind.STEP and current is not None and total is not None:
            print(f"  [{current}/{total}] {message}")
        elif kind == ProgressKind.PERCENTAGE and current is not None:
            print(f"  [{current}%] {message}")
        else:
            print(f"  ⟳ {message}")
        return self
    
    def reference(
        self,
        path: Union[str, Path],
        line: Optional[int] = None,
        end_line: Optional[int] = None,
        label: Optional[str] = None,
    ) -> ResponseStream:
        ref = FileReference(
            path=Path(path),
            line=line,
            end_line=end_line,
            label=label,
        )
        self._record_event("reference", {
            "path": str(path),
            "line": line,
            "end_line": end_line,
            "label": label,
        })
        print(f"  📄 {ref.to_markdown()}")
        return self
    
    def filetree(
        self,
        root: Union[str, Path, FileTreeNode],
        label: Optional[str] = None,
    ) -> ResponseStream:
        if isinstance(root, FileTreeNode):
            tree_text = root.to_text()
        else:
            # Build tree from path
            root_path = Path(root)
            tree_text = self._build_tree_text(root_path)
        
        self._record_event("filetree", {
            "root": str(root) if not isinstance(root, FileTreeNode) else root.name,
            "label": label,
        })
        
        if label:
            print(f"\n{label}:")
        
        if self._rich_available and self.use_rich:
            from rich.tree import Tree
            # Rich tree rendering
            if isinstance(root, FileTreeNode):
                self._print_rich_tree(root)
            else:
                print(tree_text)
        else:
            print(tree_text)
        return self
    
    def _build_tree_text(self, path: Path, prefix: str = "") -> str:
        """Build text tree from filesystem path."""
        if not path.exists():
            return f"{path.name} (not found)"
        
        lines = [path.name + "/"]
        if path.is_dir():
            items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                if item.is_dir():
                    lines.append(f"{prefix}{connector}{item.name}/")
                else:
                    lines.append(f"{prefix}{connector}{item.name}")
        return "\n".join(lines)
    
    def _print_rich_tree(self, node: FileTreeNode) -> None:
        """Print a FileTreeNode using rich Tree."""
        from rich.tree import Tree
        
        icon = "📁" if node.is_directory else "📄"
        tree = Tree(f"{icon} {node.name}")
        self._add_rich_children(tree, node.children)
        self._console.print(tree)
    
    def _add_rich_children(self, tree, children: list[FileTreeNode]) -> None:
        """Recursively add children to rich Tree."""
        for child in children:
            icon = "📁" if child.is_directory else "📄"
            suffix = f" ({child.metadata})" if child.metadata else ""
            branch = tree.add(f"{icon} {child.name}{suffix}")
            if child.children:
                self._add_rich_children(branch, child.children)
    
    def code(
        self,
        code: str,
        language: str = "python",
        filename: Optional[str] = None,
    ) -> ResponseStream:
        self._record_event("code", {
            "language": language,
            "filename": filename,
            "code": code[:500] + "..." if len(code) > 500 else code,
        })
        
        if self._rich_available and self.use_rich:
            from rich.syntax import Syntax
            from rich.panel import Panel
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            if filename:
                self._console.print(Panel(syntax, title=filename))
            else:
                self._console.print(syntax)
        else:
            if filename:
                print(f"\n--- {filename} ---")
            print(f"```{language}")
            print(code)
            print("```")
        return self
    
    def diff(
        self,
        diff_content: str,
        filename: Optional[str] = None,
    ) -> ResponseStream:
        self._record_event("diff", {
            "filename": filename,
            "diff": diff_content[:500] + "..." if len(diff_content) > 500 else diff_content,
        })
        
        if self._rich_available and self.use_rich:
            from rich.syntax import Syntax
            syntax = Syntax(diff_content, "diff", theme="monokai")
            if filename:
                from rich.panel import Panel
                self._console.print(Panel(syntax, title=f"Diff: {filename}"))
            else:
                self._console.print(syntax)
        else:
            if filename:
                print(f"\n--- Diff: {filename} ---")
            print(diff_content)
        return self
    
    def warning(self, message: str, source: Optional[str] = None) -> ResponseStream:
        self._record_event("warning", {"message": message, "source": source})
        loc = f" ({source})" if source else ""
        if self._rich_available:
            self._console.print(f"[yellow]⚠️ Warning{loc}:[/yellow] {message}")
        else:
            print(f"⚠️ Warning{loc}: {message}")
        return self
    
    def error(self, message: str, source: Optional[str] = None) -> ResponseStream:
        self._record_event("error", {"message": message, "source": source})
        loc = f" ({source})" if source else ""
        if self._rich_available:
            self._console.print(f"[red]❌ Error{loc}:[/red] {message}")
        else:
            print(f"❌ Error{loc}: {message}")
        return self
    
    def info(self, message: str) -> ResponseStream:
        self._record_event("info", message)
        if self._rich_available:
            self._console.print(f"[blue]ℹ️[/blue] {message}")
        else:
            print(f"ℹ️ {message}")
        return self
    
    def button(
        self,
        label: str,
        action: str,
        data: Optional[dict] = None,
    ) -> ResponseStream:
        # Console can't have interactive buttons, just show as text
        self._record_event("button", {"label": label, "action": action, "data": data})
        print(f"  [Button: {label}] → {action}")
        return self


class BufferedResponseStream(ResponseStream):
    """
    Response stream that buffers all output for later retrieval.
    
    Useful for:
    - Collecting agent output for transcript
    - Testing
    - Replaying streams
    """
    
    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id)
        self._markdown_buffer: list[str] = []
        self._text_buffer: list[str] = []
    
    def get_markdown(self) -> str:
        """Get all markdown content."""
        return "\n\n".join(self._markdown_buffer)
    
    def get_text(self) -> str:
        """Get all text content."""
        return "\n".join(self._text_buffer)
    
    def get_all_content(self) -> str:
        """Get all content (markdown + text) in order."""
        content = []
        for event in self._events:
            if event.kind == "markdown":
                content.append(event.content)
            elif event.kind == "text":
                content.append(event.content)
            elif event.kind == "code":
                lang = event.content.get("language", "")
                code = event.content.get("code", "")
                content.append(f"```{lang}\n{code}\n```")
            elif event.kind == "warning":
                content.append(f"⚠️ {event.content.get('message', '')}")
            elif event.kind == "error":
                content.append(f"❌ {event.content.get('message', '')}")
        return "\n\n".join(content)
    
    def markdown(self, content: str) -> ResponseStream:
        self._record_event("markdown", content)
        self._markdown_buffer.append(content)
        return self
    
    def text(self, content: str) -> ResponseStream:
        self._record_event("text", content)
        self._text_buffer.append(content)
        return self
    
    def progress(
        self,
        message: str,
        kind: ProgressKind = ProgressKind.STATUS,
        current: Optional[int] = None,
        total: Optional[int] = None,
    ) -> ResponseStream:
        self._record_event("progress", {
            "message": message,
            "kind": kind.value,
            "current": current,
            "total": total,
        })
        return self
    
    def reference(
        self,
        path: Union[str, Path],
        line: Optional[int] = None,
        end_line: Optional[int] = None,
        label: Optional[str] = None,
    ) -> ResponseStream:
        self._record_event("reference", {
            "path": str(path),
            "line": line,
            "end_line": end_line,
            "label": label,
        })
        return self
    
    def filetree(
        self,
        root: Union[str, Path, FileTreeNode],
        label: Optional[str] = None,
    ) -> ResponseStream:
        self._record_event("filetree", {
            "root": str(root) if not isinstance(root, FileTreeNode) else root.name,
            "label": label,
        })
        return self
    
    def code(
        self,
        code: str,
        language: str = "python",
        filename: Optional[str] = None,
    ) -> ResponseStream:
        self._record_event("code", {
            "language": language,
            "filename": filename,
            "code": code,
        })
        return self
    
    def diff(
        self,
        diff_content: str,
        filename: Optional[str] = None,
    ) -> ResponseStream:
        self._record_event("diff", {
            "filename": filename,
            "diff": diff_content,
        })
        return self
    
    def warning(self, message: str, source: Optional[str] = None) -> ResponseStream:
        self._record_event("warning", {"message": message, "source": source})
        return self
    
    def error(self, message: str, source: Optional[str] = None) -> ResponseStream:
        self._record_event("error", {"message": message, "source": source})
        return self
    
    def info(self, message: str) -> ResponseStream:
        self._record_event("info", message)
        return self
    
    def button(
        self,
        label: str,
        action: str,
        data: Optional[dict] = None,
    ) -> ResponseStream:
        self._record_event("button", {"label": label, "action": action, "data": data})
        return self


class NullResponseStream(ResponseStream):
    """
    Response stream that discards all output.
    
    Useful for testing or when output isn't needed.
    """
    
    def markdown(self, content: str) -> ResponseStream:
        return self
    
    def text(self, content: str) -> ResponseStream:
        return self
    
    def progress(
        self,
        message: str,
        kind: ProgressKind = ProgressKind.STATUS,
        current: Optional[int] = None,
        total: Optional[int] = None,
    ) -> ResponseStream:
        return self
    
    def reference(
        self,
        path: Union[str, Path],
        line: Optional[int] = None,
        end_line: Optional[int] = None,
        label: Optional[str] = None,
    ) -> ResponseStream:
        return self
    
    def filetree(
        self,
        root: Union[str, Path, FileTreeNode],
        label: Optional[str] = None,
    ) -> ResponseStream:
        return self
    
    def code(
        self,
        code: str,
        language: str = "python",
        filename: Optional[str] = None,
    ) -> ResponseStream:
        return self
    
    def diff(
        self,
        diff_content: str,
        filename: Optional[str] = None,
    ) -> ResponseStream:
        return self
    
    def warning(self, message: str, source: Optional[str] = None) -> ResponseStream:
        return self
    
    def error(self, message: str, source: Optional[str] = None) -> ResponseStream:
        return self
    
    def info(self, message: str) -> ResponseStream:
        return self
    
    def button(
        self,
        label: str,
        action: str,
        data: Optional[dict] = None,
    ) -> ResponseStream:
        return self


class CompositeResponseStream(ResponseStream):
    """
    Response stream that forwards to multiple streams.
    
    Useful for writing to both console and transcript.
    """
    
    def __init__(
        self,
        streams: list[ResponseStream],
        agent_id: Optional[str] = None,
    ):
        super().__init__(agent_id)
        self._streams = streams
    
    def markdown(self, content: str) -> ResponseStream:
        for stream in self._streams:
            stream.markdown(content)
        return self
    
    def text(self, content: str) -> ResponseStream:
        for stream in self._streams:
            stream.text(content)
        return self
    
    def progress(
        self,
        message: str,
        kind: ProgressKind = ProgressKind.STATUS,
        current: Optional[int] = None,
        total: Optional[int] = None,
    ) -> ResponseStream:
        for stream in self._streams:
            stream.progress(message, kind, current, total)
        return self
    
    def reference(
        self,
        path: Union[str, Path],
        line: Optional[int] = None,
        end_line: Optional[int] = None,
        label: Optional[str] = None,
    ) -> ResponseStream:
        for stream in self._streams:
            stream.reference(path, line, end_line, label)
        return self
    
    def filetree(
        self,
        root: Union[str, Path, FileTreeNode],
        label: Optional[str] = None,
    ) -> ResponseStream:
        for stream in self._streams:
            stream.filetree(root, label)
        return self
    
    def code(
        self,
        code: str,
        language: str = "python",
        filename: Optional[str] = None,
    ) -> ResponseStream:
        for stream in self._streams:
            stream.code(code, language, filename)
        return self
    
    def diff(
        self,
        diff_content: str,
        filename: Optional[str] = None,
    ) -> ResponseStream:
        for stream in self._streams:
            stream.diff(diff_content, filename)
        return self
    
    def warning(self, message: str, source: Optional[str] = None) -> ResponseStream:
        for stream in self._streams:
            stream.warning(message, source)
        return self
    
    def error(self, message: str, source: Optional[str] = None) -> ResponseStream:
        for stream in self._streams:
            stream.error(message, source)
        return self
    
    def info(self, message: str) -> ResponseStream:
        for stream in self._streams:
            stream.info(message)
        return self
    
    def button(
        self,
        label: str,
        action: str,
        data: Optional[dict] = None,
    ) -> ResponseStream:
        for stream in self._streams:
            stream.button(label, action, data)
        return self
