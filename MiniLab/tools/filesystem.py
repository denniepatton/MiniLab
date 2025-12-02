from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from . import Tool


class FileSystemTool(Tool):
    """
    Tool for creating, reading, editing, and managing files.
    Restricted to the Sandbox directory for safety.
    """

    def __init__(self, sandbox_root: str | Path):
        super().__init__(
            name="filesystem",
            description="Create, read, write, and edit files within the Sandbox directory"
        )
        self.sandbox_root = Path(sandbox_root).resolve()
        
        # Ensure sandbox exists
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
    
    def _validate_path(self, path: str | Path) -> Path:
        """
        Validate that a path is within the sandbox.
        Raises ValueError if path escapes sandbox.
        """
        requested_path = Path(path)
        
        # If not absolute, treat as relative to sandbox
        if not requested_path.is_absolute():
            full_path = (self.sandbox_root / requested_path).resolve()
        else:
            full_path = requested_path.resolve()
        
        # Ensure the path is within sandbox (prevent ../ escapes)
        try:
            full_path.relative_to(self.sandbox_root)
        except ValueError:
            raise ValueError(
                f"Access denied: '{path}' is outside the Sandbox directory. "
                f"All file operations must be within {self.sandbox_root}"
            )
        
        return full_path
    
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a filesystem operation.
        
        Actions:
        - read: Read file contents (requires: path)
        - write: Write/overwrite file (requires: path, content)
        - append: Append to file (requires: path, content)
        - create_dir: Create directory (requires: path)
        - list: List directory contents (requires: path)
        - exists: Check if path exists (requires: path)
        - delete: Delete file or directory (requires: path)
        """
        try:
            if action == "read":
                return await self._read_file(kwargs["path"])
            elif action == "write":
                return await self._write_file(kwargs["path"], kwargs["content"])
            elif action == "append":
                return await self._append_file(kwargs["path"], kwargs["content"])
            elif action == "create_dir":
                return await self._create_dir(kwargs["path"])
            elif action == "list":
                return await self._list_dir(kwargs.get("path", "."))
            elif action == "exists":
                return await self._check_exists(kwargs["path"])
            elif action == "delete":
                return await self._delete(kwargs["path"])
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": ["read", "write", "append", "create_dir", "list", "exists", "delete"]
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": action,
            }
    
    async def _read_file(self, path: str) -> Dict[str, Any]:
        """Read file contents."""
        full_path = self._validate_path(path)
        
        if not full_path.exists():
            return {
                "success": False,
                "error": f"File not found: {path}",
            }
        
        if not full_path.is_file():
            return {
                "success": False,
                "error": f"Not a file: {path}",
            }
        
        content = full_path.read_text()
        return {
            "success": True,
            "path": str(full_path.relative_to(self.sandbox_root)),
            "content": content,
            "size_bytes": len(content.encode()),
        }
    
    async def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write or overwrite file."""
        full_path = self._validate_path(path)
        
        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        full_path.write_text(content)
        return {
            "success": True,
            "action": "write",
            "path": str(full_path.relative_to(self.sandbox_root)),
            "size_bytes": len(content.encode()),
        }
    
    async def _append_file(self, path: str, content: str) -> Dict[str, Any]:
        """Append to file."""
        full_path = self._validate_path(path)
        
        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, "a") as f:
            f.write(content)
        
        return {
            "success": True,
            "action": "append",
            "path": str(full_path.relative_to(self.sandbox_root)),
        }
    
    async def _create_dir(self, path: str) -> Dict[str, Any]:
        """Create directory."""
        full_path = self._validate_path(path)
        full_path.mkdir(parents=True, exist_ok=True)
        
        return {
            "success": True,
            "action": "create_dir",
            "path": str(full_path.relative_to(self.sandbox_root)),
        }
    
    async def _list_dir(self, path: str) -> Dict[str, Any]:
        """List directory contents."""
        full_path = self._validate_path(path)
        
        if not full_path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {path}",
            }
        
        if not full_path.is_dir():
            return {
                "success": False,
                "error": f"Not a directory: {path}",
            }
        
        items = []
        for item in sorted(full_path.iterdir()):
            items.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size_bytes": item.stat().st_size if item.is_file() else None,
            })
        
        return {
            "success": True,
            "path": str(full_path.relative_to(self.sandbox_root)),
            "items": items,
            "count": len(items),
        }
    
    async def _check_exists(self, path: str) -> Dict[str, Any]:
        """Check if path exists."""
        full_path = self._validate_path(path)
        exists = full_path.exists()
        
        result = {
            "success": True,
            "path": str(full_path.relative_to(self.sandbox_root)),
            "exists": exists,
        }
        
        if exists:
            result["type"] = "directory" if full_path.is_dir() else "file"
        
        return result
    
    async def _delete(self, path: str) -> Dict[str, Any]:
        """Delete file or directory."""
        full_path = self._validate_path(path)
        
        if not full_path.exists():
            return {
                "success": False,
                "error": f"Path not found: {path}",
            }
        
        if full_path.is_file():
            full_path.unlink()
            deleted_type = "file"
        else:
            import shutil
            shutil.rmtree(full_path)
            deleted_type = "directory"
        
        return {
            "success": True,
            "action": "delete",
            "path": str(full_path.relative_to(self.sandbox_root)),
            "type": deleted_type,
        }
