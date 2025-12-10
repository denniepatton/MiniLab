"""
FileSystem Tool with PathGuard integration.

Provides read/write access to files with strict security enforcement.
- ReadData/: Read-only
- Sandbox/: Read-write
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union
from pydantic import Field

from .base import Tool, ToolInput, ToolOutput, ToolError
from ..security import PathGuard, AccessDenied
from ..utils import console


# Input schemas for each action
class ReadInput(ToolInput):
    """Input for reading a file."""
    path: str = Field(..., description="Path to the file to read")
    encoding: str = Field("utf-8", description="File encoding")


class WriteInput(ToolInput):
    """Input for writing to a file."""
    path: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write")
    encoding: str = Field("utf-8", description="File encoding")


class AppendInput(ToolInput):
    """Input for appending to a file."""
    path: str = Field(..., description="Path to the file to append to")
    content: str = Field(..., description="Content to append")
    encoding: str = Field("utf-8", description="File encoding")


class ListInput(ToolInput):
    """Input for listing directory contents."""
    path: str = Field(..., description="Path to the directory to list")
    pattern: Optional[str] = Field(None, description="Glob pattern to filter files")


class ExistsInput(ToolInput):
    """Input for checking if a path exists."""
    path: str = Field(..., description="Path to check")


class HeadInput(ToolInput):
    """Input for reading first N lines of a file."""
    path: str = Field(..., description="Path to the file")
    lines: int = Field(10, description="Number of lines to read")


class TailInput(ToolInput):
    """Input for reading last N lines of a file."""
    path: str = Field(..., description="Path to the file")
    lines: int = Field(10, description="Number of lines to read")


class SearchInput(ToolInput):
    """Input for searching within a file."""
    path: str = Field(..., description="Path to the file to search")
    pattern: str = Field(..., description="Search pattern (regex supported)")
    context_lines: int = Field(2, description="Lines of context around matches")


class CreateDirInput(ToolInput):
    """Input for creating a directory."""
    path: str = Field(..., description="Path of the directory to create")


class DeleteInput(ToolInput):
    """Input for deleting a file or directory."""
    path: str = Field(..., description="Path to delete")


class CopyInput(ToolInput):
    """Input for copying a file."""
    source: str = Field(..., description="Source path")
    destination: str = Field(..., description="Destination path")


class MoveInput(ToolInput):
    """Input for moving a file."""
    source: str = Field(..., description="Source path")
    destination: str = Field(..., description="Destination path")


class FileOutput(ToolOutput):
    """Output for file operations."""
    path: Optional[str] = None
    content: Optional[str] = None
    exists: Optional[bool] = None
    files: Optional[list[str]] = None
    matches: Optional[list[dict]] = None


class FileSystemTool(Tool):
    """
    File system operations with security enforcement.
    
    All operations are validated against PathGuard before execution.
    """
    
    name = "filesystem"
    description = "Read and write files within allowed directories (ReadData/ for reading, Sandbox/ for writing)"
    
    def __init__(self, agent_id: str, workspace_root: Path, **kwargs):
        super().__init__(agent_id, **kwargs)
        self.workspace_root = workspace_root
        self.path_guard = PathGuard.get_instance()
    
    def get_actions(self) -> dict[str, str]:
        return {
            "read": "Read the contents of a file",
            "write": "Write content to a file (Sandbox only)",
            "append": "Append content to a file (Sandbox only)",
            "list": "List contents of a directory",
            "exists": "Check if a path exists",
            "head": "Read the first N lines of a file",
            "tail": "Read the last N lines of a file",
            "search": "Search for a pattern within a file",
            "create_dir": "Create a directory (Sandbox only)",
            "delete": "Delete a file or directory (Sandbox only)",
            "copy": "Copy a file (destination must be in Sandbox)",
            "move": "Move a file (destination must be in Sandbox)",
        }
    
    def get_input_schema(self, action: str) -> type[ToolInput]:
        schemas = {
            "read": ReadInput,
            "write": WriteInput,
            "append": AppendInput,
            "list": ListInput,
            "exists": ExistsInput,
            "head": HeadInput,
            "tail": TailInput,
            "search": SearchInput,
            "create_dir": CreateDirInput,
            "delete": DeleteInput,
            "copy": CopyInput,
            "move": MoveInput,
        }
        if action not in schemas:
            raise ToolError(self.name, action, f"Unknown action: {action}")
        return schemas[action]
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to workspace root."""
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace_root / p
        return p.resolve()
    
    async def execute(self, action: str, params: dict[str, Any]) -> FileOutput:
        """Execute a filesystem action."""
        validated = self.validate_input(action, params)
        
        try:
            if action == "read":
                return await self._read(validated)
            elif action == "write":
                return await self._write(validated)
            elif action == "append":
                return await self._append(validated)
            elif action == "list":
                return await self._list(validated)
            elif action == "exists":
                return await self._exists(validated)
            elif action == "head":
                return await self._head(validated)
            elif action == "tail":
                return await self._tail(validated)
            elif action == "search":
                return await self._search(validated)
            elif action == "create_dir":
                return await self._create_dir(validated)
            elif action == "delete":
                return await self._delete(validated)
            elif action == "copy":
                return await self._copy(validated)
            elif action == "move":
                return await self._move(validated)
            else:
                raise ToolError(self.name, action, f"Unknown action: {action}")
        except AccessDenied as e:
            return FileOutput(success=False, error=str(e))
        except Exception as e:
            return FileOutput(success=False, error=f"Operation failed: {e}")
    
    async def _read(self, params: ReadInput) -> FileOutput:
        """Read file contents."""
        path = self._resolve_path(params.path)
        self.path_guard.validate_read(path, self.agent_id)
        
        if not path.exists():
            console.tool_error("Read", f"File not found: {params.path}")
            return FileOutput(success=False, error=f"File not found: {params.path}")
        
        content = path.read_text(encoding=params.encoding)
        console.file_read(params.path)
        return FileOutput(success=True, path=str(path), content=content)
    
    async def _write(self, params: WriteInput) -> FileOutput:
        """Write content to file."""
        path = self._resolve_path(params.path)
        self.path_guard.validate_write(path, self.agent_id)
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(params.content, encoding=params.encoding)
        
        console.file_write(params.path)
        return FileOutput(success=True, path=str(path), data=f"Wrote {len(params.content)} bytes")
    
    async def _append(self, params: AppendInput) -> FileOutput:
        """Append content to file."""
        path = self._resolve_path(params.path)
        self.path_guard.validate_write(path, self.agent_id)
        
        with open(path, "a", encoding=params.encoding) as f:
            f.write(params.content)
        
        return FileOutput(success=True, path=str(path), data=f"Appended {len(params.content)} bytes")
    
    async def _list(self, params: ListInput) -> FileOutput:
        """List directory contents."""
        path = self._resolve_path(params.path)
        self.path_guard.validate_read(path, self.agent_id)
        
        if not path.exists():
            console.tool_error("List", f"Directory not found: {params.path}")
            return FileOutput(success=False, error=f"Directory not found: {params.path}")
        
        if not path.is_dir():
            console.tool_error("List", f"Not a directory: {params.path}")
            return FileOutput(success=False, error=f"Not a directory: {params.path}")
        
        if params.pattern:
            files = [str(f.relative_to(path)) for f in path.glob(params.pattern)]
        else:
            files = [f.name + ("/" if f.is_dir() else "") for f in sorted(path.iterdir())]
        
        console.file_list(params.path, len(files))
        return FileOutput(success=True, path=str(path), files=files)
    
    async def _exists(self, params: ExistsInput) -> FileOutput:
        """Check if path exists."""
        path = self._resolve_path(params.path)
        
        # Allow checking existence without full read permissions
        # but still must be in allowed directories
        try:
            self.path_guard.validate_read(path, self.agent_id)
        except AccessDenied:
            return FileOutput(success=True, path=str(path), exists=False)
        
        return FileOutput(success=True, path=str(path), exists=path.exists())
    
    async def _head(self, params: HeadInput) -> FileOutput:
        """Read first N lines."""
        path = self._resolve_path(params.path)
        self.path_guard.validate_read(path, self.agent_id)
        
        if not path.exists():
            return FileOutput(success=False, error=f"File not found: {params.path}")
        
        with open(path, "r") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= params.lines:
                    break
                lines.append(line)
        
        return FileOutput(success=True, path=str(path), content="".join(lines))
    
    async def _tail(self, params: TailInput) -> FileOutput:
        """Read last N lines."""
        path = self._resolve_path(params.path)
        self.path_guard.validate_read(path, self.agent_id)
        
        if not path.exists():
            return FileOutput(success=False, error=f"File not found: {params.path}")
        
        with open(path, "r") as f:
            lines = f.readlines()
        
        content = "".join(lines[-params.lines:])
        return FileOutput(success=True, path=str(path), content=content)
    
    async def _search(self, params: SearchInput) -> FileOutput:
        """Search for pattern in file."""
        import re
        
        path = self._resolve_path(params.path)
        self.path_guard.validate_read(path, self.agent_id)
        
        if not path.exists():
            return FileOutput(success=False, error=f"File not found: {params.path}")
        
        with open(path, "r") as f:
            lines = f.readlines()
        
        matches = []
        pattern = re.compile(params.pattern, re.IGNORECASE)
        
        for i, line in enumerate(lines):
            if pattern.search(line):
                start = max(0, i - params.context_lines)
                end = min(len(lines), i + params.context_lines + 1)
                context = "".join(lines[start:end])
                matches.append({
                    "line_number": i + 1,
                    "line": line.strip(),
                    "context": context,
                })
        
        return FileOutput(success=True, path=str(path), matches=matches)
    
    async def _create_dir(self, params: CreateDirInput) -> FileOutput:
        """Create directory."""
        path = self._resolve_path(params.path)
        self.path_guard.validate_write(path, self.agent_id)
        
        path.mkdir(parents=True, exist_ok=True)
        console.tool_success("Created directory", params.path)
        return FileOutput(success=True, path=str(path), data="Directory created")
    
    async def _delete(self, params: DeleteInput) -> FileOutput:
        """Delete file or directory."""
        import shutil
        
        path = self._resolve_path(params.path)
        self.path_guard.validate_delete(path, self.agent_id)
        
        if not path.exists():
            return FileOutput(success=False, error=f"Path not found: {params.path}")
        
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        
        console.file_delete(params.path)
        return FileOutput(success=True, path=str(path), data="Deleted")
    
    async def _copy(self, params: CopyInput) -> FileOutput:
        """Copy file."""
        import shutil
        
        src = self._resolve_path(params.source)
        dst = self._resolve_path(params.destination)
        
        self.path_guard.validate_read(src, self.agent_id)
        self.path_guard.validate_write(dst, self.agent_id)
        
        if not src.exists():
            return FileOutput(success=False, error=f"Source not found: {params.source}")
        
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        
        return FileOutput(success=True, data=f"Copied {src} to {dst}")
    
    async def _move(self, params: MoveInput) -> FileOutput:
        """Move file."""
        import shutil
        
        src = self._resolve_path(params.source)
        dst = self._resolve_path(params.destination)
        
        self.path_guard.validate_read(src, self.agent_id)
        self.path_guard.validate_delete(src, self.agent_id)
        self.path_guard.validate_write(dst, self.agent_id)
        
        if not src.exists():
            return FileOutput(success=False, error=f"Source not found: {params.source}")
        
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        
        return FileOutput(success=True, data=f"Moved {src} to {dst}")
