"""
Code Editor Tool with PathGuard integration.

Provides code editing capabilities restricted to Sandbox/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Optional
from pydantic import Field

from .base import Tool, ToolInput, ToolOutput, ToolError
from ..security import PathGuard, AccessDenied
from ..utils import console


# Input schemas
class CreateInput(ToolInput):
    """Input for creating a new code file."""
    path: str = Field(..., description="Path for the new file")
    content: str = Field(..., description="Initial content")
    language: Optional[str] = Field(None, description="Programming language hint")


class ViewInput(ToolInput):
    """Input for viewing a code file."""
    path: str = Field(..., description="Path to the file")
    start_line: Optional[int] = Field(None, description="Starting line number (1-indexed)")
    end_line: Optional[int] = Field(None, description="Ending line number (1-indexed)")


class InsertInput(ToolInput):
    """Input for inserting lines into a file."""
    path: str = Field(..., description="Path to the file")
    line_number: int = Field(..., description="Line number to insert at (1-indexed)")
    content: str = Field(..., description="Content to insert")


class ReplaceInput(ToolInput):
    """Input for replacing lines in a file."""
    path: str = Field(..., description="Path to the file")
    start_line: int = Field(..., description="Starting line number (1-indexed)")
    end_line: int = Field(..., description="Ending line number (1-indexed)")
    content: str = Field(..., description="Replacement content")


class DeleteLinesInput(ToolInput):
    """Input for deleting lines from a file."""
    path: str = Field(..., description="Path to the file")
    start_line: int = Field(..., description="Starting line number (1-indexed)")
    end_line: int = Field(..., description="Ending line number (1-indexed)")


class ReplaceTextInput(ToolInput):
    """Input for find/replace in a file."""
    path: str = Field(..., description="Path to the file")
    find: str = Field(..., description="Text to find")
    replace: str = Field(..., description="Replacement text")
    count: int = Field(-1, description="Max replacements (-1 for all)")


class CheckSyntaxInput(ToolInput):
    """Input for syntax checking."""
    path: str = Field(..., description="Path to the file")
    language: Optional[str] = Field(None, description="Language (auto-detected if not provided)")


class RunInput(ToolInput):
    """Input for running a script."""
    path: str = Field(..., description="Path to the script")
    args: list[str] = Field(default_factory=list, description="Command-line arguments")
    timeout: int = Field(300, description="Timeout in seconds")


class CodeOutput(ToolOutput):
    """Output for code operations."""
    path: Optional[str] = None
    content: Optional[str] = None
    line_count: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    return_code: Optional[int] = None
    syntax_errors: Optional[list[dict]] = None


class CodeEditorTool(Tool):
    """
    Code editing operations with security enforcement.
    
    All write operations are restricted to Sandbox/.
    """
    
    name = "code_editor"
    description = "Create, edit, and run code files (writing restricted to Sandbox/)"
    
    def __init__(self, agent_id: str, workspace_root: Path, **kwargs):
        super().__init__(agent_id, **kwargs)
        self.workspace_root = workspace_root
        self.path_guard = PathGuard.get_instance()
    
    def get_actions(self) -> dict[str, str]:
        return {
            "create": "Create a new code file",
            "view": "View contents of a code file with line numbers",
            "insert": "Insert lines at a specific position",
            "replace": "Replace a range of lines",
            "delete_lines": "Delete a range of lines",
            "replace_text": "Find and replace text in a file",
            "check_syntax": "Check file for syntax errors",
            "run": "Execute a script",
        }
    
    def get_input_schema(self, action: str) -> type[ToolInput]:
        schemas = {
            "create": CreateInput,
            "view": ViewInput,
            "insert": InsertInput,
            "replace": ReplaceInput,
            "delete_lines": DeleteLinesInput,
            "replace_text": ReplaceTextInput,
            "check_syntax": CheckSyntaxInput,
            "run": RunInput,
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
    
    async def execute(self, action: str, params: dict[str, Any]) -> CodeOutput:
        """Execute a code editor action."""
        validated = self.validate_input(action, params)
        
        try:
            if action == "create":
                return await self._create(validated)
            elif action == "view":
                return await self._view(validated)
            elif action == "insert":
                return await self._insert(validated)
            elif action == "replace":
                return await self._replace(validated)
            elif action == "delete_lines":
                return await self._delete_lines(validated)
            elif action == "replace_text":
                return await self._replace_text(validated)
            elif action == "check_syntax":
                return await self._check_syntax(validated)
            elif action == "run":
                return await self._run(validated)
            else:
                raise ToolError(self.name, action, f"Unknown action: {action}")
        except AccessDenied as e:
            return CodeOutput(success=False, error=str(e))
        except Exception as e:
            return CodeOutput(success=False, error=f"Operation failed: {e}")
    
    async def _create(self, params: CreateInput) -> CodeOutput:
        """Create a new code file."""
        path = self._resolve_path(params.path)
        self.path_guard.validate_write(path, self.agent_id)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(params.content)
        
        lines = params.content.count("\n") + 1
        console.file_write(params.path)
        return CodeOutput(
            success=True,
            path=str(path),
            line_count=lines,
            data=f"Created file with {lines} lines"
        )
    
    async def _view(self, params: ViewInput) -> CodeOutput:
        """View file contents with line numbers."""
        path = self._resolve_path(params.path)
        self.path_guard.validate_read(path, self.agent_id)
        
        if not path.exists():
            console.tool_error("View", f"File not found: {params.path}")
            return CodeOutput(success=False, error=f"File not found: {params.path}")
        
        lines = path.read_text().splitlines()
        
        start = (params.start_line or 1) - 1
        end = params.end_line or len(lines)
        
        # Format with line numbers
        numbered_lines = []
        for i, line in enumerate(lines[start:end], start=start + 1):
            numbered_lines.append(f"{i:4d} | {line}")
        
        content = "\n".join(numbered_lines)
        console.file_read(params.path)
        return CodeOutput(
            success=True,
            path=str(path),
            content=content,
            line_count=len(lines)
        )
    
    async def _insert(self, params: InsertInput) -> CodeOutput:
        """Insert lines at a position."""
        path = self._resolve_path(params.path)
        self.path_guard.validate_write(path, self.agent_id)
        
        if not path.exists():
            return CodeOutput(success=False, error=f"File not found: {params.path}")
        
        lines = path.read_text().splitlines(keepends=True)
        insert_idx = params.line_number - 1
        
        new_lines = params.content.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        
        lines[insert_idx:insert_idx] = new_lines
        path.write_text("".join(lines))
        
        return CodeOutput(
            success=True,
            path=str(path),
            line_count=len(lines),
            data=f"Inserted {len(new_lines)} lines at line {params.line_number}"
        )
    
    async def _replace(self, params: ReplaceInput) -> CodeOutput:
        """Replace a range of lines."""
        path = self._resolve_path(params.path)
        self.path_guard.validate_write(path, self.agent_id)
        
        if not path.exists():
            return CodeOutput(success=False, error=f"File not found: {params.path}")
        
        lines = path.read_text().splitlines(keepends=True)
        
        start_idx = params.start_line - 1
        end_idx = params.end_line
        
        new_lines = params.content.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        
        lines[start_idx:end_idx] = new_lines
        path.write_text("".join(lines))
        
        return CodeOutput(
            success=True,
            path=str(path),
            line_count=len(lines),
            data=f"Replaced lines {params.start_line}-{params.end_line}"
        )
    
    async def _delete_lines(self, params: DeleteLinesInput) -> CodeOutput:
        """Delete a range of lines."""
        path = self._resolve_path(params.path)
        self.path_guard.validate_write(path, self.agent_id)
        
        if not path.exists():
            return CodeOutput(success=False, error=f"File not found: {params.path}")
        
        lines = path.read_text().splitlines(keepends=True)
        
        start_idx = params.start_line - 1
        end_idx = params.end_line
        
        del lines[start_idx:end_idx]
        path.write_text("".join(lines))
        
        return CodeOutput(
            success=True,
            path=str(path),
            line_count=len(lines),
            data=f"Deleted lines {params.start_line}-{params.end_line}"
        )
    
    async def _replace_text(self, params: ReplaceTextInput) -> CodeOutput:
        """Find and replace text."""
        path = self._resolve_path(params.path)
        self.path_guard.validate_write(path, self.agent_id)
        
        if not path.exists():
            return CodeOutput(success=False, error=f"File not found: {params.path}")
        
        content = path.read_text()
        
        if params.count == -1:
            new_content = content.replace(params.find, params.replace)
        else:
            new_content = content.replace(params.find, params.replace, params.count)
        
        replacements = content.count(params.find) if params.count == -1 else min(content.count(params.find), params.count)
        path.write_text(new_content)
        
        return CodeOutput(
            success=True,
            path=str(path),
            data=f"Made {replacements} replacement(s)"
        )
    
    async def _check_syntax(self, params: CheckSyntaxInput) -> CodeOutput:
        """Check file for syntax errors."""
        path = self._resolve_path(params.path)
        self.path_guard.validate_read(path, self.agent_id)
        
        if not path.exists():
            return CodeOutput(success=False, error=f"File not found: {params.path}")
        
        # Detect language from extension if not provided
        lang = params.language or path.suffix.lstrip(".")
        
        errors = []
        
        if lang in ("py", "python"):
            # Use Python's compile to check syntax
            try:
                content = path.read_text()
                compile(content, str(path), "exec")
            except SyntaxError as e:
                errors.append({
                    "line": e.lineno,
                    "column": e.offset,
                    "message": e.msg,
                })
        elif lang in ("r", "R"):
            # For R, we'd need R installed - just note this
            return CodeOutput(
                success=True,
                path=str(path),
                syntax_errors=[],
                data="R syntax checking requires R to be installed"
            )
        else:
            return CodeOutput(
                success=True,
                path=str(path),
                syntax_errors=[],
                data=f"No syntax checker available for .{lang} files"
            )
        
        return CodeOutput(
            success=True,
            path=str(path),
            syntax_errors=errors,
            data="No syntax errors found" if not errors else f"Found {len(errors)} syntax error(s)"
        )
    
    async def _run(self, params: RunInput) -> CodeOutput:
        """Run a script."""
        path = self._resolve_path(params.path)
        
        # Must be in Sandbox to run
        self.path_guard.validate_read(path, self.agent_id)
        
        # Check if agent can execute
        permissions = self.path_guard.get_agent_permissions(self.agent_id)
        if not permissions.can_execute_shell:
            return CodeOutput(
                success=False,
                error=f"Agent {self.agent_id} is not allowed to execute code"
            )
        
        if not path.exists():
            return CodeOutput(success=False, error=f"File not found: {params.path}")
        
        # Determine interpreter
        ext = path.suffix.lower()
        if ext == ".py":
            cmd = [sys.executable, str(path)] + params.args
        elif ext in (".r", ".R"):
            cmd = ["Rscript", str(path)] + params.args
        elif ext == ".sh":
            cmd = ["bash", str(path)] + params.args
        else:
            return CodeOutput(success=False, error=f"Don't know how to run .{ext} files")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=params.timeout,
                cwd=path.parent,
            )
            
            return CodeOutput(
                success=result.returncode == 0,
                path=str(path),
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return CodeOutput(
                success=False,
                error=f"Script timed out after {params.timeout} seconds"
            )
        except Exception as e:
            return CodeOutput(success=False, error=f"Failed to run script: {e}")
