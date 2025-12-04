"""
Dual-Mode Filesystem Tool for MiniLab Agents

Provides:
- READ-ONLY access to ReadData/ directory
- READ-WRITE access to Sandbox/ directory
- Prevents any file creation or modification in ReadData/
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import Tool


class DualModeFileSystemTool(Tool):
    """
    Dual-mode filesystem tool with read-only access to data and read-write access to sandbox.
    
    Access Modes:
    - ReadData/: READ-ONLY (can list, read, check existence)
    - Sandbox/: READ-WRITE (can list, read, write, create, delete)
    
    This protects source data while allowing full manipulation in the working area.
    """

    def __init__(
        self, 
        workspace_root: str | Path,
        read_only_dirs: List[str] = None,
        read_write_dirs: List[str] = None,
    ):
        super().__init__(
            name="filesystem",
            description=(
                "Access the file system with read-only access to ReadData/ and "
                "read-write access to Sandbox/. Use this to explore data files, "
                "read their contents, and create/modify files in the Sandbox."
            )
        )
        self.workspace_root = Path(workspace_root).resolve()
        
        # Default directories
        self.read_only_dirs = read_only_dirs or ["ReadData"]
        self.read_write_dirs = read_write_dirs or ["Sandbox"]
        
        # Ensure directories exist
        for dir_name in self.read_write_dirs:
            (self.workspace_root / dir_name).mkdir(parents=True, exist_ok=True)
    
    def _get_access_mode(self, path: Path) -> str:
        """
        Determine access mode for a path.
        
        Returns:
            "read-only": Path is in a read-only directory
            "read-write": Path is in a read-write directory
            "denied": Path is outside allowed directories
        """
        resolved = path.resolve()
        
        # Check read-only directories
        for ro_dir in self.read_only_dirs:
            ro_path = (self.workspace_root / ro_dir).resolve()
            try:
                resolved.relative_to(ro_path)
                return "read-only"
            except ValueError:
                continue
        
        # Check read-write directories
        for rw_dir in self.read_write_dirs:
            rw_path = (self.workspace_root / rw_dir).resolve()
            try:
                resolved.relative_to(rw_path)
                return "read-write"
            except ValueError:
                continue
        
        return "denied"
    
    def _resolve_path(self, path: str | Path) -> Path:
        """
        Resolve a path relative to workspace root.
        Handles both absolute and relative paths.
        """
        p = Path(path)
        
        if not p.is_absolute():
            # Try relative to workspace root
            return (self.workspace_root / p).resolve()
        
        return p.resolve()
    
    def _validate_read_access(self, path: str | Path) -> tuple[Path, str | None]:
        """
        Validate that path can be read.
        Returns (resolved_path, error_message or None).
        """
        resolved = self._resolve_path(path)
        mode = self._get_access_mode(resolved)
        
        if mode == "denied":
            return resolved, (
                f"Access denied: '{path}' is outside allowed directories.\n"
                f"Read access allowed in: {', '.join(self.read_only_dirs)}\n"
                f"Read/write access allowed in: {', '.join(self.read_write_dirs)}"
            )
        
        return resolved, None
    
    def _validate_write_access(self, path: str | Path) -> tuple[Path, str | None]:
        """
        Validate that path can be written.
        Returns (resolved_path, error_message or None).
        """
        resolved = self._resolve_path(path)
        mode = self._get_access_mode(resolved)
        
        if mode == "denied":
            return resolved, (
                f"Access denied: '{path}' is outside allowed directories.\n"
                f"Write access only allowed in: {', '.join(self.read_write_dirs)}"
            )
        
        if mode == "read-only":
            return resolved, (
                f"Write access denied: '{path}' is in a read-only directory.\n"
                f"ReadData/ is protected - use Sandbox/ for creating/modifying files."
            )
        
        return resolved, None
    
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a filesystem operation.
        
        Read Actions (work on ReadData/ and Sandbox/):
        - read: Read file contents (requires: path)
        - list: List directory contents (requires: path)
        - exists: Check if path exists (requires: path)
        - head: Read first N lines of file (requires: path, optional: lines)
        - tail: Read last N lines of file (requires: path, optional: lines)
        - search: Search for files matching pattern (requires: pattern, optional: path)
        
        Write Actions (Sandbox/ only):
        - write: Write/overwrite file (requires: path, content)
        - append: Append to file (requires: path, content)
        - create_dir: Create directory (requires: path)
        - delete: Delete file or directory (requires: path)
        - copy: Copy file (requires: src, dst)
        - move: Move/rename file (requires: src, dst)
        """
        try:
            # Read operations
            if action == "read":
                return await self._read_file(kwargs["path"])
            elif action == "list":
                return await self._list_dir(kwargs.get("path", "."))
            elif action == "exists":
                return await self._check_exists(kwargs["path"])
            elif action == "head":
                return await self._read_head(kwargs["path"], kwargs.get("lines", 20))
            elif action == "tail":
                return await self._read_tail(kwargs["path"], kwargs.get("lines", 20))
            elif action == "search":
                return await self._search_files(
                    kwargs["pattern"], 
                    kwargs.get("path", None)
                )
            
            # Write operations
            elif action == "write":
                return await self._write_file(kwargs["path"], kwargs["content"])
            elif action == "append":
                return await self._append_file(kwargs["path"], kwargs["content"])
            elif action == "create_dir":
                return await self._create_dir(kwargs["path"])
            elif action == "delete":
                return await self._delete(kwargs["path"])
            elif action == "copy":
                return await self._copy(kwargs["src"], kwargs["dst"])
            elif action == "move":
                return await self._move(kwargs["src"], kwargs["dst"])
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": {
                        "read_operations": ["read", "list", "exists", "head", "tail", "search"],
                        "write_operations": ["write", "append", "create_dir", "delete", "copy", "move"],
                    }
                }
        except KeyError as e:
            return {
                "success": False,
                "error": f"Missing required parameter: {e}",
                "action": action,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": action,
            }
    
    # =========================================================================
    # READ OPERATIONS
    # =========================================================================
    
    async def _read_file(self, path: str) -> Dict[str, Any]:
        """Read entire file contents."""
        full_path, error = self._validate_read_access(path)
        if error:
            return {"success": False, "error": error}
        
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        if not full_path.is_file():
            return {"success": False, "error": f"Not a file: {path}"}
        
        try:
            content = full_path.read_text()
            return {
                "success": True,
                "path": str(path),
                "content": content,
                "size_bytes": len(content.encode()),
                "lines": len(content.split('\n')),
            }
        except UnicodeDecodeError:
            # Binary file
            return {
                "success": False,
                "error": f"Cannot read binary file as text: {path}",
                "suggestion": "This appears to be a binary file. Use appropriate tools for this file type.",
            }
    
    async def _read_head(self, path: str, lines: int = 20) -> Dict[str, Any]:
        """Read first N lines of a file."""
        full_path, error = self._validate_read_access(path)
        if error:
            return {"success": False, "error": error}
        
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        try:
            with open(full_path, 'r') as f:
                head_lines = []
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    head_lines.append(line.rstrip('\n'))
            
            return {
                "success": True,
                "path": str(path),
                "lines_requested": lines,
                "lines_returned": len(head_lines),
                "content": '\n'.join(head_lines),
            }
        except UnicodeDecodeError:
            return {"success": False, "error": f"Cannot read binary file: {path}"}
    
    async def _read_tail(self, path: str, lines: int = 20) -> Dict[str, Any]:
        """Read last N lines of a file."""
        full_path, error = self._validate_read_access(path)
        if error:
            return {"success": False, "error": error}
        
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        try:
            with open(full_path, 'r') as f:
                all_lines = f.readlines()
            
            tail_lines = [line.rstrip('\n') for line in all_lines[-lines:]]
            
            return {
                "success": True,
                "path": str(path),
                "lines_requested": lines,
                "lines_returned": len(tail_lines),
                "total_lines": len(all_lines),
                "content": '\n'.join(tail_lines),
            }
        except UnicodeDecodeError:
            return {"success": False, "error": f"Cannot read binary file: {path}"}
    
    async def _list_dir(self, path: str) -> Dict[str, Any]:
        """List directory contents."""
        full_path, error = self._validate_read_access(path)
        if error:
            return {"success": False, "error": error}
        
        if not full_path.exists():
            return {"success": False, "error": f"Directory not found: {path}"}
        
        if not full_path.is_dir():
            return {"success": False, "error": f"Not a directory: {path}"}
        
        items = []
        for item in sorted(full_path.iterdir()):
            if item.name.startswith('.'):
                continue  # Skip hidden files
            
            item_info = {
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
            }
            
            if item.is_file():
                item_info["size_bytes"] = item.stat().st_size
                # Detect file type from extension
                ext = item.suffix.lower()
                if ext in ['.csv', '.tsv', '.txt']:
                    item_info["format"] = "tabular"
                elif ext in ['.json', '.yaml', '.yml']:
                    item_info["format"] = "structured"
                elif ext in ['.py', '.r', '.sh']:
                    item_info["format"] = "script"
                elif ext in ['.png', '.jpg', '.jpeg', '.pdf']:
                    item_info["format"] = "image/document"
                elif ext in ['.h5', '.hdf5', '.h5ad']:
                    item_info["format"] = "hdf5"
            
            items.append(item_info)
        
        return {
            "success": True,
            "path": str(path),
            "items": items,
            "count": len(items),
            "access_mode": self._get_access_mode(full_path),
        }
    
    async def _check_exists(self, path: str) -> Dict[str, Any]:
        """Check if path exists."""
        full_path, error = self._validate_read_access(path)
        if error:
            return {"success": False, "error": error}
        
        exists = full_path.exists()
        result = {
            "success": True,
            "path": str(path),
            "exists": exists,
        }
        
        if exists:
            result["type"] = "directory" if full_path.is_dir() else "file"
            result["access_mode"] = self._get_access_mode(full_path)
            if full_path.is_file():
                result["size_bytes"] = full_path.stat().st_size
        
        return result
    
    async def _search_files(
        self, 
        pattern: str, 
        base_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for files matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "*.csv", "**/*.py")
            base_path: Directory to search in (defaults to workspace root)
        """
        if base_path:
            search_root, error = self._validate_read_access(base_path)
            if error:
                return {"success": False, "error": error}
        else:
            search_root = self.workspace_root
        
        if not search_root.exists():
            return {"success": False, "error": f"Search path not found: {base_path}"}
        
        matches = []
        for match in search_root.glob(pattern):
            # Check access mode
            mode = self._get_access_mode(match)
            if mode != "denied":
                try:
                    rel_path = match.relative_to(self.workspace_root)
                    matches.append({
                        "path": str(rel_path),
                        "type": "directory" if match.is_dir() else "file",
                        "access_mode": mode,
                    })
                except ValueError:
                    pass  # Skip paths outside workspace
        
        return {
            "success": True,
            "pattern": pattern,
            "base_path": str(base_path) if base_path else "workspace_root",
            "matches": matches,
            "count": len(matches),
        }
    
    # =========================================================================
    # WRITE OPERATIONS (Sandbox only)
    # =========================================================================
    
    async def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write or overwrite file (Sandbox only)."""
        full_path, error = self._validate_write_access(path)
        if error:
            return {"success": False, "error": error}
        
        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        full_path.write_text(content)
        
        return {
            "success": True,
            "action": "write",
            "path": str(path),
            "size_bytes": len(content.encode()),
            "lines": len(content.split('\n')),
        }
    
    async def _append_file(self, path: str, content: str) -> Dict[str, Any]:
        """Append to file (Sandbox only)."""
        full_path, error = self._validate_write_access(path)
        if error:
            return {"success": False, "error": error}
        
        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, "a") as f:
            f.write(content)
        
        return {
            "success": True,
            "action": "append",
            "path": str(path),
            "appended_bytes": len(content.encode()),
        }
    
    async def _create_dir(self, path: str) -> Dict[str, Any]:
        """Create directory (Sandbox only)."""
        full_path, error = self._validate_write_access(path)
        if error:
            return {"success": False, "error": error}
        
        full_path.mkdir(parents=True, exist_ok=True)
        
        return {
            "success": True,
            "action": "create_dir",
            "path": str(path),
        }
    
    async def _delete(self, path: str) -> Dict[str, Any]:
        """Delete file or directory (Sandbox only)."""
        full_path, error = self._validate_write_access(path)
        if error:
            return {"success": False, "error": error}
        
        if not full_path.exists():
            return {"success": False, "error": f"Path not found: {path}"}
        
        if full_path.is_file():
            full_path.unlink()
            deleted_type = "file"
        else:
            shutil.rmtree(full_path)
            deleted_type = "directory"
        
        return {
            "success": True,
            "action": "delete",
            "path": str(path),
            "type": deleted_type,
        }
    
    async def _copy(self, src: str, dst: str) -> Dict[str, Any]:
        """
        Copy file or directory.
        Source can be in ReadData/ or Sandbox/.
        Destination must be in Sandbox/.
        """
        src_path, error = self._validate_read_access(src)
        if error:
            return {"success": False, "error": f"Cannot read source: {error}"}
        
        dst_path, error = self._validate_write_access(dst)
        if error:
            return {"success": False, "error": f"Cannot write destination: {error}"}
        
        if not src_path.exists():
            return {"success": False, "error": f"Source not found: {src}"}
        
        # Create parent directories for destination
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        if src_path.is_file():
            shutil.copy2(src_path, dst_path)
            copy_type = "file"
        else:
            shutil.copytree(src_path, dst_path)
            copy_type = "directory"
        
        return {
            "success": True,
            "action": "copy",
            "src": str(src),
            "dst": str(dst),
            "type": copy_type,
        }
    
    async def _move(self, src: str, dst: str) -> Dict[str, Any]:
        """
        Move/rename file or directory (Sandbox only for both src and dst).
        """
        src_path, error = self._validate_write_access(src)
        if error:
            return {"success": False, "error": f"Cannot modify source: {error}"}
        
        dst_path, error = self._validate_write_access(dst)
        if error:
            return {"success": False, "error": f"Cannot write destination: {error}"}
        
        if not src_path.exists():
            return {"success": False, "error": f"Source not found: {src}"}
        
        # Create parent directories for destination
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.move(str(src_path), str(dst_path))
        
        return {
            "success": True,
            "action": "move",
            "src": str(src),
            "dst": str(dst),
        }
