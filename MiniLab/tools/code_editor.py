"""
Code Editor Tool for MiniLab Agents

Provides TRUE incremental code editing capabilities, mimicking how VS Code agents work:
- Start with empty/skeleton files
- Make small, targeted edits (replace specific lines, insert at position)
- Append code incrementally
- See the current state of the file
- Run and get output
- Make surgical fixes based on errors

This approach uses FEWER TOKENS than regenerating entire scripts because:
1. You don't re-send the whole file every time
2. Fixes are surgical - "change line 45" not "here's all 300 lines again"
3. No regeneration of working code - keep what works, touch only what's broken
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import Tool


class CodeEditorTool(Tool):
    """
    Incremental code editor for building scripts piece by piece.
    
    Like a VS Code agent, this tool allows:
    - Creating empty files
    - Appending code chunks
    - Replacing specific line ranges
    - Inserting at specific positions
    - Viewing current file state with line numbers
    - Running and capturing output
    - Making surgical fixes
    
    This is MORE TOKEN EFFICIENT than regenerating whole scripts.
    """

    def __init__(
        self, 
        workspace_root: str | Path,
        sandbox_dir: str = "Sandbox",
        python_command: List[str] = None,
    ):
        super().__init__(
            name="code_editor",
            description=(
                "Build and edit Python scripts incrementally.\n\n"
                "WORKFLOW: 1) CREATE the script, 2) RUN it, 3) FIX errors with edit/replace, 4) RUN again.\n\n"
                "Actions:\n"
                "  - create: {path, content} - Create script file with content (MUST DO FIRST!)\n"
                "  - view: {path} - View file with line numbers\n"
                "  - append: {path, code} - Add code to end of file\n"
                "  - replace: {path, start_line, end_line, code} - Replace specific lines\n"
                "  - run: {path} - Execute script, get stdout/stderr\n\n"
                "IMPORTANT: Always CREATE a script before trying to RUN it!\n\n"
                "Examples:\n"
                "  Create: {\"tool\": \"code_editor\", \"action\": \"create\", \"params\": {\"path\": \"Sandbox/Project/scripts/analysis.py\", \"content\": \"import pandas as pd\\n...\"}}\n"
                "  Run: {\"tool\": \"code_editor\", \"action\": \"run\", \"params\": {\"path\": \"Sandbox/Project/scripts/analysis.py\"}}\n"
                "  Fix: {\"tool\": \"code_editor\", \"action\": \"replace\", \"params\": {\"path\": \"Sandbox/Project/scripts/analysis.py\", \"start_line\": 15, \"end_line\": 17, \"code\": \"# fixed code\"}}"
            )
        )
        self.workspace_root = Path(workspace_root).resolve()
        self.sandbox_dir = sandbox_dir
        self.python_command = python_command or ["micromamba", "run", "-n", "minilab", "python"]
        
        # Track open files for context
        self._file_cache: Dict[str, List[str]] = {}
    
    def _resolve_path(self, path: str | Path) -> Tuple[Path, Optional[str]]:
        """
        Resolve path and validate it's in Sandbox.
        Returns (resolved_path, error_message or None).
        """
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace_root / p
        p = p.resolve()
        
        sandbox_path = (self.workspace_root / self.sandbox_dir).resolve()
        try:
            p.relative_to(sandbox_path)
            return p, None
        except ValueError:
            return p, f"Path must be in {self.sandbox_dir}/ directory. Got: {path}"
    
    def _get_lines(self, path: Path) -> List[str]:
        """Get file contents as list of lines (cached)."""
        path_str = str(path)
        if path_str not in self._file_cache:
            if path.exists():
                self._file_cache[path_str] = path.read_text().splitlines(keepends=True)
            else:
                self._file_cache[path_str] = []
        return self._file_cache[path_str]
    
    def _save_lines(self, path: Path, lines: List[str]) -> None:
        """Save lines to file and update cache."""
        path.parent.mkdir(parents=True, exist_ok=True)
        content = ''.join(lines)
        # Ensure file ends with newline
        if content and not content.endswith('\n'):
            content += '\n'
        path.write_text(content)
        self._file_cache[str(path)] = content.splitlines(keepends=True)
    
    def _invalidate_cache(self, path: Path) -> None:
        """Clear cache for a path."""
        self._file_cache.pop(str(path), None)
    
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a code editing operation.
        
        Actions:
        - create: Create new empty file or with skeleton (requires: path, optional: skeleton/code/content)
        - view: View file with line numbers (requires: path, optional: start_line, end_line)
        - append: Append code to end of file (requires: path, code/content)
        - insert: Insert code at line number (requires: path, line, code/content)
        - replace: Replace line range with new code (requires: path, start_line, end_line, code/content)
        - delete_lines: Delete line range (requires: path, start_line, end_line)
        - replace_text: Find and replace specific text (requires: path, old_text, new_text)
        - check_syntax: Check Python syntax (requires: path)
        - run: Run the script and return output (requires: path, optional: timeout)
        - run_check: Quick import/syntax check without full run (requires: path)
        """
        # Handle parameter aliases: 'content' -> 'code', 'skeleton' can be 'code'/'content'
        if 'content' in kwargs and 'code' not in kwargs:
            kwargs['code'] = kwargs.pop('content')
        if 'skeleton' not in kwargs and 'code' in kwargs and action == 'create':
            kwargs['skeleton'] = kwargs.get('code', '')
        
        try:
            if action == "create":
                return await self._create_file(
                    kwargs["path"], 
                    kwargs.get("skeleton", kwargs.get("code", ""))
                )
            elif action == "view":
                return await self._view_file(
                    kwargs["path"],
                    kwargs.get("start_line"),
                    kwargs.get("end_line"),
                )
            elif action == "append":
                # Check for code parameter and give helpful error
                code = kwargs.get("code")
                if code is None:
                    return {
                        "success": False,
                        "error": "Missing required parameter: 'code'. Use 'code' (not 'content') for the text to append.",
                        "hint": "Example: {\"tool\": \"code_editor\", \"action\": \"append\", \"params\": {\"path\": \"...\", \"code\": \"...\"}}"
                    }
                return await self._append_code(kwargs["path"], code)
            elif action == "insert":
                return await self._insert_code(
                    kwargs["path"], 
                    kwargs["line"], 
                    kwargs.get("code", "")
                )
            elif action == "replace":
                return await self._replace_lines(
                    kwargs["path"],
                    kwargs["start_line"],
                    kwargs["end_line"],
                    kwargs["code"],
                )
            elif action == "delete_lines":
                return await self._delete_lines(
                    kwargs["path"],
                    kwargs["start_line"],
                    kwargs["end_line"],
                )
            elif action == "replace_text":
                return await self._replace_text(
                    kwargs["path"],
                    kwargs["old_text"],
                    kwargs["new_text"],
                )
            elif action == "check_syntax":
                return await self._check_syntax(kwargs["path"])
            elif action == "run":
                return await self._run_script(
                    kwargs["path"],
                    kwargs.get("timeout", 300),
                )
            elif action == "run_check":
                return await self._run_import_check(kwargs["path"])
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": [
                        "create", "view", "append", "insert", "replace",
                        "delete_lines", "replace_text", "check_syntax", 
                        "run", "run_check"
                    ],
                }
        except KeyError as e:
            return {"success": False, "error": f"Missing required parameter: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================
    
    async def _create_file(self, path: str, skeleton: str = "", overwrite: bool = False) -> Dict[str, Any]:
        """Create a new file, optionally with skeleton code. If file exists, overwrites with new content."""
        full_path, error = self._resolve_path(path)
        if error:
            return {"success": False, "error": error}
        
        # If file exists, overwrite it (more agent-friendly behavior)
        if full_path.exists():
            if skeleton:
                # Overwrite with new content
                self._save_lines(full_path, [skeleton] if skeleton else [])
                lines = self._get_lines(full_path)
                return {
                    "success": True,
                    "action": "create (overwrite)",
                    "path": str(path),
                    "lines": len(lines),
                    "message": f"Overwrote {path} with {len(lines)} lines. Use 'view' to see it.",
                }
            else:
                # No new content provided, just acknowledge it exists
                lines = self._get_lines(full_path)
                return {
                    "success": True,
                    "action": "create (already exists)",
                    "path": str(path),
                    "lines": len(lines),
                    "message": f"File already exists with {len(lines)} lines. Use 'view' to see it, 'append' to add code.",
                }
        
        # Default Python skeleton if none provided
        if not skeleton and path.endswith('.py'):
            skeleton = '''"""
Script: {filename}
Generated by MiniLab
"""

import numpy as np
import pandas as pd

np.random.seed(42)


if __name__ == "__main__":
    pass  # TODO: implement main logic
'''.format(filename=full_path.name)
        
        self._save_lines(full_path, [skeleton] if skeleton else [])
        
        lines = self._get_lines(full_path)
        return {
            "success": True,
            "action": "create",
            "path": str(path),
            "lines": len(lines),
            "message": f"Created {path} with {len(lines)} lines. Use 'view' to see it, 'append' to add code.",
        }
    
    async def _view_file(
        self, 
        path: str, 
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """View file contents with line numbers."""
        full_path, error = self._resolve_path(path)
        if error:
            return {"success": False, "error": error}
        
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        self._invalidate_cache(full_path)  # Refresh from disk
        lines = self._get_lines(full_path)
        
        # Determine range
        start = (start_line or 1) - 1  # Convert to 0-indexed
        end = end_line if end_line else len(lines)
        
        start = max(0, start)
        end = min(len(lines), end)
        
        # Format with line numbers
        numbered_lines = []
        for i, line in enumerate(lines[start:end], start=start+1):
            # Remove trailing newline for display
            display_line = line.rstrip('\n')
            numbered_lines.append(f"{i:4d} | {display_line}")
        
        return {
            "success": True,
            "path": str(path),
            "total_lines": len(lines),
            "showing_lines": f"{start+1}-{end}",
            "content": '\n'.join(numbered_lines),
        }
    
    async def _append_code(self, path: str, code: str) -> Dict[str, Any]:
        """Append code to end of file. Auto-creates file if it doesn't exist."""
        full_path, error = self._resolve_path(path)
        if error:
            return {"success": False, "error": error}
        
        # Auto-create file if it doesn't exist
        if not full_path.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text("")
            self._file_cache[str(full_path)] = []
            print(f"      📄 Created new file: {path}")
        
        self._invalidate_cache(full_path)
        lines = self._get_lines(full_path)
        old_count = len(lines)
        
        # Ensure code ends with newline
        if code and not code.endswith('\n'):
            code += '\n'
        
        # Add blank line separator if file doesn't end with blank line
        if lines and lines[-1].strip():
            code = '\n' + code
        
        new_lines = code.splitlines(keepends=True)
        lines.extend(new_lines)
        
        self._save_lines(full_path, lines)
        
        return {
            "success": True,
            "action": "append",
            "path": str(path),
            "lines_added": len(new_lines),
            "total_lines": len(lines),
            "message": f"Appended {len(new_lines)} lines (lines {old_count+1}-{len(lines)}). Use 'view' to see result.",
        }
    
    async def _insert_code(self, path: str, line: int, code: str) -> Dict[str, Any]:
        """Insert code at a specific line number (1-indexed)."""
        full_path, error = self._resolve_path(path)
        if error:
            return {"success": False, "error": error}
        
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        self._invalidate_cache(full_path)
        lines = self._get_lines(full_path)
        
        if line < 1 or line > len(lines) + 1:
            return {
                "success": False, 
                "error": f"Invalid line number: {line}. File has {len(lines)} lines."
            }
        
        # Ensure code ends with newline
        if code and not code.endswith('\n'):
            code += '\n'
        
        new_lines = code.splitlines(keepends=True)
        
        # Insert at position (convert to 0-indexed)
        insert_pos = line - 1
        for i, new_line in enumerate(new_lines):
            lines.insert(insert_pos + i, new_line)
        
        self._save_lines(full_path, lines)
        
        return {
            "success": True,
            "action": "insert",
            "path": str(path),
            "inserted_at_line": line,
            "lines_inserted": len(new_lines),
            "total_lines": len(lines),
            "message": f"Inserted {len(new_lines)} lines at line {line}.",
        }
    
    async def _replace_lines(
        self, 
        path: str, 
        start_line: int, 
        end_line: int, 
        code: str
    ) -> Dict[str, Any]:
        """Replace a range of lines with new code."""
        full_path, error = self._resolve_path(path)
        if error:
            return {"success": False, "error": error}
        
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        self._invalidate_cache(full_path)
        lines = self._get_lines(full_path)
        
        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return {
                "success": False,
                "error": f"Invalid line range: {start_line}-{end_line}. File has {len(lines)} lines."
            }
        
        # Ensure code ends with newline
        if code and not code.endswith('\n'):
            code += '\n'
        
        new_lines = code.splitlines(keepends=True)
        
        # Replace range (convert to 0-indexed)
        lines[start_line-1:end_line] = new_lines
        
        self._save_lines(full_path, lines)
        
        lines_removed = end_line - start_line + 1
        return {
            "success": True,
            "action": "replace",
            "path": str(path),
            "replaced_lines": f"{start_line}-{end_line}",
            "lines_removed": lines_removed,
            "lines_added": len(new_lines),
            "total_lines": len(lines),
            "message": f"Replaced lines {start_line}-{end_line} ({lines_removed} lines) with {len(new_lines)} new lines.",
        }
    
    async def _delete_lines(self, path: str, start_line: int, end_line: int) -> Dict[str, Any]:
        """Delete a range of lines."""
        full_path, error = self._resolve_path(path)
        if error:
            return {"success": False, "error": error}
        
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        self._invalidate_cache(full_path)
        lines = self._get_lines(full_path)
        
        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return {
                "success": False,
                "error": f"Invalid line range: {start_line}-{end_line}. File has {len(lines)} lines."
            }
        
        # Delete range (convert to 0-indexed)
        del lines[start_line-1:end_line]
        
        self._save_lines(full_path, lines)
        
        lines_removed = end_line - start_line + 1
        return {
            "success": True,
            "action": "delete_lines",
            "path": str(path),
            "deleted_lines": f"{start_line}-{end_line}",
            "lines_removed": lines_removed,
            "total_lines": len(lines),
        }
    
    async def _replace_text(self, path: str, old_text: str, new_text: str) -> Dict[str, Any]:
        """Find and replace specific text in file."""
        full_path, error = self._resolve_path(path)
        if error:
            return {"success": False, "error": error}
        
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        content = full_path.read_text()
        
        if old_text not in content:
            return {
                "success": False,
                "error": f"Text not found in file. Make sure to match exactly including whitespace.",
                "hint": "Use 'view' to see the exact content, including indentation.",
            }
        
        count = content.count(old_text)
        new_content = content.replace(old_text, new_text)
        
        full_path.write_text(new_content)
        self._invalidate_cache(full_path)
        
        return {
            "success": True,
            "action": "replace_text",
            "path": str(path),
            "replacements": count,
            "message": f"Replaced {count} occurrence(s) of the specified text.",
        }
    
    # =========================================================================
    # VALIDATION & EXECUTION
    # =========================================================================
    
    async def _check_syntax(self, path: str) -> Dict[str, Any]:
        """Check Python syntax without running."""
        full_path, error = self._resolve_path(path)
        if error:
            return {"success": False, "error": error}
        
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        code = full_path.read_text()
        
        try:
            ast.parse(code)
            return {
                "success": True,
                "path": str(path),
                "syntax_valid": True,
                "message": "✓ Syntax is valid.",
            }
        except SyntaxError as e:
            return {
                "success": True,  # The check succeeded, syntax is just invalid
                "path": str(path),
                "syntax_valid": False,
                "error_line": e.lineno,
                "error_offset": e.offset,
                "error_message": str(e.msg),
                "message": f"✗ Syntax error at line {e.lineno}: {e.msg}",
            }
    
    async def _run_import_check(self, path: str) -> Dict[str, Any]:
        """Quick check: just try to import/compile the file."""
        full_path, error = self._resolve_path(path)
        if error:
            return {"success": False, "error": error}
        
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        # Run python -m py_compile
        cmd = self.python_command + ["-m", "py_compile", str(full_path)]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.workspace_root),
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "path": str(path),
                    "compile_valid": True,
                    "message": "✓ File compiles successfully.",
                }
            else:
                return {
                    "success": True,
                    "path": str(path),
                    "compile_valid": False,
                    "error": result.stderr or result.stdout,
                    "message": "✗ Compilation failed. See error for details.",
                }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Compilation check timed out.",
            }
    
    async def _run_script(self, path: str, timeout: int = 300) -> Dict[str, Any]:
        """Run the script and return output."""
        full_path, error = self._resolve_path(path)
        if error:
            return {"success": False, "error": error}
        
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        cmd = self.python_command + [str(full_path)]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace_root),
            )
            
            # Truncate output to prevent token explosion
            stdout = result.stdout[-3000:] if result.stdout else ""
            stderr = result.stderr[-2000:] if result.stderr else ""
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "path": str(path),
                    "returncode": 0,
                    "stdout": stdout,
                    "stderr": stderr if stderr else None,
                    "message": "✓ Script ran successfully.",
                }
            else:
                # Parse error for line number
                error_line = None
                if stderr:
                    import re
                    match = re.search(r'line (\d+)', stderr)
                    if match:
                        error_line = int(match.group(1))
                
                return {
                    "success": True,  # The run completed, just with error
                    "path": str(path),
                    "returncode": result.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "error_line": error_line,
                    "message": f"✗ Script failed with exit code {result.returncode}.",
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "path": str(path),
                "error": f"Script timed out after {timeout} seconds.",
                "message": "✗ Script timed out. Consider breaking into smaller pieces.",
            }
