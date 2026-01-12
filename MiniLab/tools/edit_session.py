"""
Edit Session and WorkspaceEdit - VS Code-style atomic batched edits.

Provides a structured way to batch file edits, preview them, and apply
or rollback atomically. This mirrors VS Code's ChatEditingSession and
WorkspaceEdit patterns.

Key concepts:
- TextEdit: A single edit to a text file (insert, replace, delete)
- WorkspaceEdit: Collection of edits across multiple files
- EditSession: Manages a workspace edit with preview/commit/rollback

Benefits:
- Batch multiple changes before committing
- Preview diffs before applying
- Atomic commit or rollback
- Track which edits came from agents vs users
- Support undo by reverting to snapshots
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union
import json
import shutil


class EditKind(Enum):
    """Type of text edit operation."""
    INSERT = "insert"           # Insert text at position
    REPLACE = "replace"         # Replace range with new text
    DELETE = "delete"           # Delete range
    CREATE_FILE = "create_file" # Create new file
    DELETE_FILE = "delete_file" # Delete file
    RENAME_FILE = "rename_file" # Rename/move file


class EditState(Enum):
    """State of an edit in the session."""
    PENDING = "pending"         # Not yet applied
    APPLIED = "applied"         # Applied to file
    REJECTED = "rejected"       # User rejected
    FAILED = "failed"           # Application failed


@dataclass
class Position:
    """A position in a text document (0-indexed)."""
    line: int
    column: int = 0
    
    def to_dict(self) -> dict:
        return {"line": self.line, "column": self.column}
    
    @classmethod
    def from_dict(cls, data: dict) -> Position:
        return cls(line=data["line"], column=data.get("column", 0))
    
    def __lt__(self, other: Position) -> bool:
        if self.line != other.line:
            return self.line < other.line
        return self.column < other.column


@dataclass
class Range:
    """A range in a text document."""
    start: Position
    end: Position
    
    def to_dict(self) -> dict:
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}
    
    @classmethod
    def from_dict(cls, data: dict) -> Range:
        return cls(
            start=Position.from_dict(data["start"]),
            end=Position.from_dict(data["end"]),
        )
    
    @classmethod
    def from_lines(cls, start_line: int, end_line: int) -> Range:
        """Create a range spanning full lines (0-indexed)."""
        return cls(
            start=Position(start_line, 0),
            end=Position(end_line, 0),
        )
    
    @classmethod
    def at_line(cls, line: int) -> Range:
        """Create a range at a single line (for insertion)."""
        return cls(
            start=Position(line, 0),
            end=Position(line, 0),
        )


@dataclass
class TextEdit:
    """
    A single text edit operation.
    
    Mirrors VS Code's TextEdit with range and newText.
    """
    range: Range
    new_text: str
    kind: EditKind = EditKind.REPLACE
    
    # Metadata
    description: Optional[str] = None
    agent_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "range": self.range.to_dict(),
            "new_text": self.new_text,
            "kind": self.kind.value,
            "description": self.description,
            "agent_id": self.agent_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> TextEdit:
        return cls(
            range=Range.from_dict(data["range"]),
            new_text=data["new_text"],
            kind=EditKind(data.get("kind", "replace")),
            description=data.get("description"),
            agent_id=data.get("agent_id"),
        )
    
    @classmethod
    def insert(
        cls,
        line: int,
        text: str,
        description: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> TextEdit:
        """Create an insert edit at a line."""
        return cls(
            range=Range.at_line(line),
            new_text=text,
            kind=EditKind.INSERT,
            description=description,
            agent_id=agent_id,
        )
    
    @classmethod
    def replace_lines(
        cls,
        start_line: int,
        end_line: int,
        text: str,
        description: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> TextEdit:
        """Create a replace edit for a line range."""
        return cls(
            range=Range.from_lines(start_line, end_line),
            new_text=text,
            kind=EditKind.REPLACE,
            description=description,
            agent_id=agent_id,
        )
    
    @classmethod
    def delete_lines(
        cls,
        start_line: int,
        end_line: int,
        description: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> TextEdit:
        """Create a delete edit for a line range."""
        return cls(
            range=Range.from_lines(start_line, end_line),
            new_text="",
            kind=EditKind.DELETE,
            description=description,
            agent_id=agent_id,
        )


@dataclass
class FileEdit:
    """
    File-level edit operation (create, delete, rename).
    """
    kind: EditKind
    path: Path
    new_path: Optional[Path] = None  # For rename
    content: Optional[str] = None    # For create
    
    # Metadata
    description: Optional[str] = None
    agent_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "path": str(self.path),
            "new_path": str(self.new_path) if self.new_path else None,
            "content": self.content[:1000] if self.content else None,  # Truncate for serialization
            "description": self.description,
            "agent_id": self.agent_id,
        }
    
    @classmethod
    def create_file(
        cls,
        path: Path,
        content: str = "",
        description: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> FileEdit:
        """Create a new file."""
        return cls(
            kind=EditKind.CREATE_FILE,
            path=path,
            content=content,
            description=description,
            agent_id=agent_id,
        )
    
    @classmethod
    def delete_file(
        cls,
        path: Path,
        description: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> FileEdit:
        """Delete a file."""
        return cls(
            kind=EditKind.DELETE_FILE,
            path=path,
            description=description,
            agent_id=agent_id,
        )
    
    @classmethod
    def rename_file(
        cls,
        old_path: Path,
        new_path: Path,
        description: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> FileEdit:
        """Rename/move a file."""
        return cls(
            kind=EditKind.RENAME_FILE,
            path=old_path,
            new_path=new_path,
            description=description,
            agent_id=agent_id,
        )


@dataclass
class FileEditEntry:
    """
    Entry tracking edits to a single file in a WorkspaceEdit.
    """
    path: Path
    text_edits: list[TextEdit] = field(default_factory=list)
    file_edit: Optional[FileEdit] = None
    state: EditState = EditState.PENDING
    
    # Original content snapshot for rollback
    original_content: Optional[str] = None
    
    # Error info if application failed
    error: Optional[str] = None
    
    def has_changes(self) -> bool:
        """Check if there are any pending changes."""
        return bool(self.text_edits) or self.file_edit is not None
    
    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "text_edits": [e.to_dict() for e in self.text_edits],
            "file_edit": self.file_edit.to_dict() if self.file_edit else None,
            "state": self.state.value,
            "error": self.error,
        }


class WorkspaceEdit:
    """
    A collection of edits across multiple files.
    
    Mirrors VS Code's WorkspaceEdit class with methods like:
    - replace(uri, range, newText)
    - insert(uri, position, newText)  
    - delete(uri, range)
    - createFile(uri)
    - deleteFile(uri)
    - renameFile(oldUri, newUri)
    
    Edits can be staged and then applied atomically via EditSession.
    """
    
    def __init__(self, description: Optional[str] = None):
        self.description = description
        self._entries: dict[Path, FileEditEntry] = {}
        self.created_at = datetime.now()
        self.agent_id: Optional[str] = None
    
    def _get_or_create_entry(self, path: Path) -> FileEditEntry:
        """Get or create an entry for a file."""
        path = path.resolve()
        if path not in self._entries:
            self._entries[path] = FileEditEntry(path=path)
        return self._entries[path]
    
    @property
    def entries(self) -> list[FileEditEntry]:
        """Get all file entries."""
        return list(self._entries.values())
    
    @property
    def size(self) -> int:
        """Get total number of edits."""
        count = 0
        for entry in self._entries.values():
            count += len(entry.text_edits)
            if entry.file_edit:
                count += 1
        return count
    
    def has_changes(self) -> bool:
        """Check if there are any pending changes."""
        return any(entry.has_changes() for entry in self._entries.values())
    
    # --- VS Code-style methods ---
    
    def replace(
        self,
        path: Union[str, Path],
        range_or_start: Union[Range, int],
        end_line_or_text: Union[int, str],
        text: Optional[str] = None,
        description: Optional[str] = None,
    ) -> WorkspaceEdit:
        """
        Replace text in a range.
        
        Can be called as:
        - replace(path, Range, text)
        - replace(path, start_line, end_line, text)
        """
        path = Path(path).resolve()
        entry = self._get_or_create_entry(path)
        
        if isinstance(range_or_start, Range):
            edit_range = range_or_start
            new_text = str(end_line_or_text)
        else:
            edit_range = Range.from_lines(range_or_start, int(end_line_or_text))
            new_text = text or ""
        
        entry.text_edits.append(TextEdit(
            range=edit_range,
            new_text=new_text,
            kind=EditKind.REPLACE,
            description=description,
            agent_id=self.agent_id,
        ))
        return self
    
    def insert(
        self,
        path: Union[str, Path],
        position: Union[Position, int],
        text: str,
        description: Optional[str] = None,
    ) -> WorkspaceEdit:
        """
        Insert text at a position.
        
        Can be called as:
        - insert(path, Position, text)
        - insert(path, line, text)
        """
        path = Path(path).resolve()
        entry = self._get_or_create_entry(path)
        
        if isinstance(position, int):
            pos = Position(position, 0)
        else:
            pos = position
        
        entry.text_edits.append(TextEdit(
            range=Range(start=pos, end=pos),
            new_text=text,
            kind=EditKind.INSERT,
            description=description,
            agent_id=self.agent_id,
        ))
        return self
    
    def delete(
        self,
        path: Union[str, Path],
        range_or_start: Union[Range, int],
        end_line: Optional[int] = None,
        description: Optional[str] = None,
    ) -> WorkspaceEdit:
        """
        Delete text in a range.
        
        Can be called as:
        - delete(path, Range)
        - delete(path, start_line, end_line)
        """
        path = Path(path).resolve()
        entry = self._get_or_create_entry(path)
        
        if isinstance(range_or_start, Range):
            edit_range = range_or_start
        else:
            edit_range = Range.from_lines(range_or_start, end_line or range_or_start)
        
        entry.text_edits.append(TextEdit(
            range=edit_range,
            new_text="",
            kind=EditKind.DELETE,
            description=description,
            agent_id=self.agent_id,
        ))
        return self
    
    def create_file(
        self,
        path: Union[str, Path],
        content: str = "",
        overwrite: bool = False,
        description: Optional[str] = None,
    ) -> WorkspaceEdit:
        """Create a new file."""
        path = Path(path).resolve()
        entry = self._get_or_create_entry(path)
        entry.file_edit = FileEdit.create_file(
            path=path,
            content=content,
            description=description,
            agent_id=self.agent_id,
        )
        return self
    
    def delete_file(
        self,
        path: Union[str, Path],
        description: Optional[str] = None,
    ) -> WorkspaceEdit:
        """Delete a file."""
        path = Path(path).resolve()
        entry = self._get_or_create_entry(path)
        entry.file_edit = FileEdit.delete_file(
            path=path,
            description=description,
            agent_id=self.agent_id,
        )
        return self
    
    def rename_file(
        self,
        old_path: Union[str, Path],
        new_path: Union[str, Path],
        description: Optional[str] = None,
    ) -> WorkspaceEdit:
        """Rename/move a file."""
        old_path = Path(old_path).resolve()
        new_path = Path(new_path).resolve()
        entry = self._get_or_create_entry(old_path)
        entry.file_edit = FileEdit.rename_file(
            old_path=old_path,
            new_path=new_path,
            description=description,
            agent_id=self.agent_id,
        )
        return self
    
    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "entries": [e.to_dict() for e in self._entries.values()],
            "created_at": self.created_at.isoformat(),
            "agent_id": self.agent_id,
        }


class EditSession:
    """
    Manages a WorkspaceEdit with preview, commit, and rollback capabilities.
    
    Mirrors VS Code's IChatEditingSession with methods like:
    - stage_edit() - Add edits to the session
    - preview() - Generate diff preview
    - commit() / apply() - Apply all edits
    - rollback() / reject() - Discard all edits
    - accept(path) / reject(path) - Accept/reject per-file
    
    Usage:
        session = EditSession("my_project")
        session.stage_edit(path, TextEdit.replace_lines(10, 15, new_code))
        print(session.preview())  # See diff
        await session.commit()    # Apply all
        # or
        await session.rollback()  # Discard all
    """
    
    def __init__(
        self,
        project_name: str,
        workspace_root: Optional[Path] = None,
        agent_id: Optional[str] = None,
    ):
        self.project_name = project_name
        self.workspace_root = workspace_root
        self.agent_id = agent_id
        self.workspace_edit = WorkspaceEdit()
        self.workspace_edit.agent_id = agent_id
        
        # Session state
        self.created_at = datetime.now()
        self._is_committed = False
        self._is_rolled_back = False
        
        # Snapshots for rollback
        self._snapshots: dict[Path, str] = {}
        
        # Callbacks
        self.on_edit_staged: Optional[Callable[[Path, TextEdit], None]] = None
        self.on_edit_applied: Optional[Callable[[Path], None]] = None
        self.on_edit_rejected: Optional[Callable[[Path], None]] = None
    
    @property
    def is_active(self) -> bool:
        """Check if session is still active (not committed or rolled back)."""
        return not self._is_committed and not self._is_rolled_back
    
    @property
    def has_changes(self) -> bool:
        """Check if there are pending changes."""
        return self.workspace_edit.has_changes()
    
    @property
    def entries(self) -> list[FileEditEntry]:
        """Get all file entries."""
        return self.workspace_edit.entries
    
    def _snapshot_file(self, path: Path) -> None:
        """Capture file content for rollback."""
        if path not in self._snapshots:
            if path.exists():
                try:
                    self._snapshots[path] = path.read_text()
                except Exception:
                    pass  # File may be binary or inaccessible
    
    def stage_edit(
        self,
        path: Union[str, Path],
        edit: TextEdit,
    ) -> EditSession:
        """
        Stage a text edit for a file.
        
        Args:
            path: File path
            edit: TextEdit to stage
            
        Returns:
            self for chaining
        """
        if not self.is_active:
            raise RuntimeError("EditSession is no longer active")
        
        path = Path(path).resolve()
        self._snapshot_file(path)
        
        entry = self.workspace_edit._get_or_create_entry(path)
        edit.agent_id = edit.agent_id or self.agent_id
        entry.text_edits.append(edit)
        
        if self.on_edit_staged:
            self.on_edit_staged(path, edit)
        
        return self
    
    def stage_file_edit(
        self,
        edit: FileEdit,
    ) -> EditSession:
        """
        Stage a file-level edit (create, delete, rename).
        
        Args:
            edit: FileEdit to stage
            
        Returns:
            self for chaining
        """
        if not self.is_active:
            raise RuntimeError("EditSession is no longer active")
        
        self._snapshot_file(edit.path)
        if edit.new_path:
            self._snapshot_file(edit.new_path)
        
        entry = self.workspace_edit._get_or_create_entry(edit.path)
        edit.agent_id = edit.agent_id or self.agent_id
        entry.file_edit = edit
        
        return self
    
    def replace(
        self,
        path: Union[str, Path],
        start_line: int,
        end_line: int,
        text: str,
        description: Optional[str] = None,
    ) -> EditSession:
        """Convenience method to stage a replace edit."""
        return self.stage_edit(
            path,
            TextEdit.replace_lines(start_line, end_line, text, description, self.agent_id),
        )
    
    def insert(
        self,
        path: Union[str, Path],
        line: int,
        text: str,
        description: Optional[str] = None,
    ) -> EditSession:
        """Convenience method to stage an insert edit."""
        return self.stage_edit(
            path,
            TextEdit.insert(line, text, description, self.agent_id),
        )
    
    def delete_lines(
        self,
        path: Union[str, Path],
        start_line: int,
        end_line: int,
        description: Optional[str] = None,
    ) -> EditSession:
        """Convenience method to stage a delete edit."""
        return self.stage_edit(
            path,
            TextEdit.delete_lines(start_line, end_line, description, self.agent_id),
        )
    
    def preview(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a unified diff preview of all pending changes.
        
        Args:
            path: Optional specific file to preview (all files if None)
            
        Returns:
            Unified diff format string
        """
        diffs = []
        
        entries = self.workspace_edit.entries
        if path:
            path = Path(path).resolve()
            entries = [e for e in entries if e.path == path]
        
        for entry in entries:
            if not entry.has_changes():
                continue
            
            diff = self._generate_diff(entry)
            if diff:
                diffs.append(diff)
        
        return "\n".join(diffs)
    
    def _generate_diff(self, entry: FileEditEntry) -> str:
        """Generate diff for a single file entry."""
        path = entry.path
        
        # Handle file-level edits
        if entry.file_edit:
            if entry.file_edit.kind == EditKind.CREATE_FILE:
                new_lines = (entry.file_edit.content or "").splitlines(keepends=True)
                diff = difflib.unified_diff(
                    [],
                    new_lines,
                    fromfile="/dev/null",
                    tofile=str(path),
                )
                return "".join(diff)
            
            elif entry.file_edit.kind == EditKind.DELETE_FILE:
                if path.exists():
                    old_lines = path.read_text().splitlines(keepends=True)
                else:
                    old_lines = self._snapshots.get(path, "").splitlines(keepends=True)
                diff = difflib.unified_diff(
                    old_lines,
                    [],
                    fromfile=str(path),
                    tofile="/dev/null",
                )
                return "".join(diff)
            
            elif entry.file_edit.kind == EditKind.RENAME_FILE:
                return f"rename {path} -> {entry.file_edit.new_path}"
        
        # Handle text edits
        if not entry.text_edits:
            return ""
        
        # Get original content
        if path in self._snapshots:
            original = self._snapshots[path]
        elif path.exists():
            original = path.read_text()
        else:
            original = ""
        
        # Apply edits to get new content
        new_content = self._apply_edits_to_content(original, entry.text_edits)
        
        # Generate diff
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
        )
        return "".join(diff)
    
    def _apply_edits_to_content(
        self,
        content: str,
        edits: list[TextEdit],
    ) -> str:
        """Apply a list of edits to content string."""
        lines = content.splitlines(keepends=True)
        if content and not content.endswith("\n"):
            # Ensure last line has newline for consistent handling
            if lines:
                lines[-1] += "\n"
        
        # Sort edits by position, process in reverse order to maintain line numbers
        sorted_edits = sorted(
            edits,
            key=lambda e: (e.range.start.line, e.range.start.column),
            reverse=True,
        )
        
        for edit in sorted_edits:
            start_line = edit.range.start.line
            end_line = edit.range.end.line
            
            # Ensure new_text has proper line endings
            new_lines = edit.new_text.splitlines(keepends=True)
            if edit.new_text and not edit.new_text.endswith("\n"):
                if new_lines:
                    new_lines[-1] += "\n"
            
            if edit.kind == EditKind.INSERT:
                # Insert at position
                if start_line >= len(lines):
                    lines.extend(new_lines)
                else:
                    lines[start_line:start_line] = new_lines
            
            elif edit.kind == EditKind.DELETE:
                # Delete range
                del lines[start_line:end_line + 1]
            
            else:  # REPLACE
                # Replace range with new content
                lines[start_line:end_line + 1] = new_lines
        
        return "".join(lines)
    
    async def commit(self) -> dict[Path, bool]:
        """
        Apply all staged edits atomically.
        
        Returns:
            Dict mapping paths to success status
        """
        if not self.is_active:
            raise RuntimeError("EditSession is no longer active")
        
        results: dict[Path, bool] = {}
        
        try:
            for entry in self.workspace_edit.entries:
                if not entry.has_changes():
                    continue
                
                try:
                    await self._apply_entry(entry)
                    entry.state = EditState.APPLIED
                    results[entry.path] = True
                    
                    if self.on_edit_applied:
                        self.on_edit_applied(entry.path)
                        
                except Exception as e:
                    entry.state = EditState.FAILED
                    entry.error = str(e)
                    results[entry.path] = False
            
            self._is_committed = True
            return results
            
        except Exception as e:
            # Rollback on any failure
            await self.rollback()
            raise
    
    async def _apply_entry(self, entry: FileEditEntry) -> None:
        """Apply edits for a single file entry."""
        path = entry.path
        
        # Handle file-level edits first
        if entry.file_edit:
            if entry.file_edit.kind == EditKind.CREATE_FILE:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(entry.file_edit.content or "")
                return
            
            elif entry.file_edit.kind == EditKind.DELETE_FILE:
                if path.exists():
                    path.unlink()
                return
            
            elif entry.file_edit.kind == EditKind.RENAME_FILE:
                if path.exists() and entry.file_edit.new_path:
                    entry.file_edit.new_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(entry.file_edit.new_path))
                return
        
        # Handle text edits
        if not entry.text_edits:
            return
        
        # Get original content
        if path in self._snapshots:
            original = self._snapshots[path]
        elif path.exists():
            original = path.read_text()
        else:
            original = ""
        
        # Apply edits
        new_content = self._apply_edits_to_content(original, entry.text_edits)
        
        # Write result
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_content)
    
    async def rollback(self) -> None:
        """
        Rollback all changes to original state.
        
        Restores files from snapshots taken when edits were staged.
        """
        if self._is_rolled_back:
            return
        
        for path, content in self._snapshots.items():
            try:
                if content is not None:
                    path.write_text(content)
            except Exception:
                pass  # Best effort
        
        for entry in self.workspace_edit.entries:
            entry.state = EditState.REJECTED
            if self.on_edit_rejected:
                self.on_edit_rejected(entry.path)
        
        self._is_rolled_back = True
    
    async def accept(self, path: Union[str, Path]) -> bool:
        """
        Accept edits for a specific file.
        
        Args:
            path: File to accept edits for
            
        Returns:
            True if successful
        """
        path = Path(path).resolve()
        
        for entry in self.workspace_edit.entries:
            if entry.path == path and entry.state == EditState.PENDING:
                try:
                    await self._apply_entry(entry)
                    entry.state = EditState.APPLIED
                    if self.on_edit_applied:
                        self.on_edit_applied(path)
                    return True
                except Exception as e:
                    entry.state = EditState.FAILED
                    entry.error = str(e)
                    return False
        
        return False
    
    async def reject(self, path: Union[str, Path]) -> None:
        """
        Reject edits for a specific file.
        
        Args:
            path: File to reject edits for
        """
        path = Path(path).resolve()
        
        for entry in self.workspace_edit.entries:
            if entry.path == path and entry.state == EditState.PENDING:
                entry.state = EditState.REJECTED
                if self.on_edit_rejected:
                    self.on_edit_rejected(path)
                break
    
    def get_entry(self, path: Union[str, Path]) -> Optional[FileEditEntry]:
        """Get the entry for a specific file."""
        path = Path(path).resolve()
        return self.workspace_edit._entries.get(path)
    
    def summary(self) -> str:
        """Get a summary of the edit session."""
        lines = [f"EditSession: {self.project_name}"]
        lines.append(f"Created: {self.created_at.isoformat()}")
        lines.append(f"Status: {'active' if self.is_active else 'committed' if self._is_committed else 'rolled back'}")
        lines.append(f"Files: {len(self.workspace_edit.entries)}")
        lines.append(f"Total edits: {self.workspace_edit.size}")
        
        for entry in self.workspace_edit.entries:
            status = entry.state.value
            edits = len(entry.text_edits)
            file_op = f" [{entry.file_edit.kind.value}]" if entry.file_edit else ""
            lines.append(f"  {entry.path.name}: {edits} edits{file_op} ({status})")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "created_at": self.created_at.isoformat(),
            "agent_id": self.agent_id,
            "is_committed": self._is_committed,
            "is_rolled_back": self._is_rolled_back,
            "workspace_edit": self.workspace_edit.to_dict(),
        }


# Global edit session registry
_active_sessions: dict[str, EditSession] = {}


def get_edit_session(project_name: str) -> Optional[EditSession]:
    """Get active edit session for a project."""
    return _active_sessions.get(project_name)


def create_edit_session(
    project_name: str,
    workspace_root: Optional[Path] = None,
    agent_id: Optional[str] = None,
) -> EditSession:
    """Create a new edit session for a project."""
    session = EditSession(
        project_name=project_name,
        workspace_root=workspace_root,
        agent_id=agent_id,
    )
    _active_sessions[project_name] = session
    return session


def close_edit_session(project_name: str) -> None:
    """Close and remove an edit session."""
    if project_name in _active_sessions:
        del _active_sessions[project_name]
