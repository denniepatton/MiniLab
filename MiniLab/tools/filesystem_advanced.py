from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal

from . import Tool


class DualModeFileSystemTool(Tool):
    """
    Advanced filesystem tool with read/write and read-only capabilities.
    
    - Sandbox directory: Full read/write access
    - ReadData directory: Read-only access
    
    Enforces strict path validation to prevent directory traversal attacks.
    """

    def __init__(
        self, 
        sandbox_root: str | Path,
        readonly_root: str | Path | None = None
    ):
        super().__init__(
            name="filesystem",
            description="File operations with Sandbox (RW) and ReadData (RO) access"
        )
        self.sandbox_root = Path(sandbox_root).resolve()
        self.readonly_root = Path(readonly_root).resolve() if readonly_root else None
        
        # Ensure directories exist
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        if self.readonly_root:
            self.readonly_root.mkdir(parents=True, exist_ok=True)
    
    def _validate_and_resolve_path(
        self, 
        path: str | Path,
        require_write: bool = False
    ) -> tuple[Path, Literal["sandbox", "readonly"]]:
        """
        Validate path and determine which root it belongs to.
        
        Args:
            path: Requested file path
            require_write: If True, path must be in sandbox (writable)
            
        Returns:
            (resolved_path, access_type)
            
        Raises:
            ValueError: If path is invalid or requires write but is in readonly
        """
        # Strip "Sandbox/" or "ReadData/" prefixes if provided
        # (agents might explicitly name the directory, but we don't want to double it)
        path_str = str(path)
        
        # Strip workspace root if agent provided absolute path
        workspace_root = str(Path.cwd())
        if path_str.startswith(workspace_root):
            path_str = path_str[len(workspace_root):].lstrip("/\\")
        
        # Strip common prefixes (with or without trailing slash)
        original_had_sandbox_prefix = path_str.startswith("Sandbox/") or path_str.startswith("Sandbox\\") or path_str == "Sandbox"
        original_had_readdata_prefix = path_str.startswith("ReadData/") or path_str.startswith("ReadData\\") or path_str == "ReadData"
        original_had_outputs_prefix = path_str.startswith("Outputs/") or path_str.startswith("Outputs\\") or path_str == "Outputs"
        
        if original_had_sandbox_prefix:
            if path_str == "Sandbox":
                path_str = "."
            else:
                path_str = path_str[8:]  # Remove "Sandbox/"
        elif original_had_readdata_prefix:
            if path_str == "ReadData":
                path_str = "."
            else:
                path_str = path_str[9:]  # Remove "ReadData/"
        elif original_had_outputs_prefix:
            # Redirect Outputs/ to Sandbox/ - agents should work in Sandbox, files will be copied
            if path_str == "Outputs":
                path_str = "."
            else:
                path_str = path_str[8:]  # Remove "Outputs/"
        
        requested_path = Path(path_str)
        
        # For write operations, if original path explicitly referenced ReadData, reject immediately
        if require_write and original_had_readdata_prefix:
            raise ValueError(
                f"Access denied: '{path}' is in ReadData (read-only). "
                f"Write operations are only allowed in Sandbox."
            )
        
        # Note: Outputs/ paths are automatically redirected to Sandbox/ above
        # The actual copy to Outputs/ happens at the end of the workflow
        
        # For write operations, validate it's in sandbox
        if require_write:
            if not requested_path.is_absolute():
                sandbox_path = (self.sandbox_root / requested_path).resolve()
            else:
                sandbox_path = requested_path.resolve()
            
            try:
                sandbox_path.relative_to(self.sandbox_root)
                return (sandbox_path, "sandbox")
            except ValueError:
                raise ValueError(
                    f"Access denied: Write operations only allowed in Sandbox. "
                    f"Path '{path}' is outside Sandbox ({self.sandbox_root})"
                )
        
        # For read operations, try readonly first (since it's more common for data exploration)
        if self.readonly_root:
            if not requested_path.is_absolute():
                readonly_path = (self.readonly_root / requested_path).resolve()
            else:
                readonly_path = requested_path.resolve()
            
            try:
                readonly_path.relative_to(self.readonly_root)
                # Check if path exists in readonly
                if readonly_path.exists():
                    return (readonly_path, "readonly")
            except ValueError:
                pass  # Not in readonly, try sandbox
        
        # Try sandbox for read operations
        if not requested_path.is_absolute():
            sandbox_path = (self.sandbox_root / requested_path).resolve()
        else:
            sandbox_path = requested_path.resolve()
        
        try:
            sandbox_path.relative_to(self.sandbox_root)
            # For read operations, we allow non-existent paths in sandbox
            # (they might be created later)
            return (sandbox_path, "sandbox")
        except ValueError:
            pass  # Not in sandbox either
        
        # Path is in neither allowed directory
        allowed = [f"Sandbox ({self.sandbox_root})"]
        if self.readonly_root:
            allowed.append(f"ReadData ({self.readonly_root})")
        
        raise ValueError(
            f"Access denied: '{path}' is outside allowed directories. "
            f"Allowed: {', '.join(allowed)}"
        )
    
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a filesystem operation.
        
        Actions:
        - read: Read file contents (requires: path)
        - write: Write/overwrite file (requires: path, content) [Sandbox only]
        - append: Append to file (requires: path, content) [Sandbox only]
        - create_dir: Create directory (requires: path) [Sandbox only]
        - list: List directory contents (requires: path)
        - exists: Check if path exists (requires: path)
        - delete: Delete file or directory (requires: path) [Sandbox only]
        - copy_to_sandbox: Copy file from ReadData to Sandbox (requires: source, dest)
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
            elif action == "copy_to_sandbox":
                return await self._copy_to_sandbox(kwargs["source"], kwargs.get("dest"))
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": [
                        "read", "write", "append", "create_dir", 
                        "list", "exists", "delete", "copy_to_sandbox"
                    ]
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": action,
            }
    
    async def _read_file(self, path: str) -> Dict[str, Any]:
        """Read file contents (works in both Sandbox and ReadData)."""
        full_path, access_type = self._validate_and_resolve_path(path, require_write=False)
        
        if not full_path.exists():
            return {
                "success": False,
                "error": f"File not found: {path}",
            }
        
        if not full_path.is_file():
            if full_path.is_dir():
                return {
                    "success": False,
                    "error": f"Cannot read directory: {path}. Use action='list' to list directory contents.",
                }
            else:
                return {
                    "success": False,
                    "error": f"Not a file: {path}",
                }
        
        content = full_path.read_text()
        root = self.sandbox_root if access_type == "sandbox" else self.readonly_root
        
        return {
            "success": True,
            "path": str(full_path.relative_to(root)),
            "content": content,
            "size_bytes": len(content.encode()),
            "access_type": access_type,
        }
    
    async def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write or overwrite file (Sandbox only)."""
        full_path, _ = self._validate_and_resolve_path(path, require_write=True)
        
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
        """Append to file (Sandbox only)."""
        full_path, _ = self._validate_and_resolve_path(path, require_write=True)
        
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
        """Create directory (Sandbox only)."""
        full_path, _ = self._validate_and_resolve_path(path, require_write=True)
        full_path.mkdir(parents=True, exist_ok=True)
        
        return {
            "success": True,
            "action": "create_dir",
            "path": str(full_path.relative_to(self.sandbox_root)),
        }
    
    async def _list_dir(self, path: str) -> Dict[str, Any]:
        """List directory contents (works in both Sandbox and ReadData)."""
        full_path, access_type = self._validate_and_resolve_path(path, require_write=False)
        
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
        
        root = self.sandbox_root if access_type == "sandbox" else self.readonly_root
        
        return {
            "success": True,
            "path": str(full_path.relative_to(root)),
            "items": items,
            "count": len(items),
            "access_type": access_type,
        }
    
    async def _check_exists(self, path: str) -> Dict[str, Any]:
        """Check if path exists (works in both Sandbox and ReadData)."""
        try:
            full_path, access_type = self._validate_and_resolve_path(path, require_write=False)
            exists = full_path.exists()
            
            root = self.sandbox_root if access_type == "sandbox" else self.readonly_root
            
            result = {
                "success": True,
                "path": str(full_path.relative_to(root)),
                "exists": exists,
                "access_type": access_type,
            }
            
            if exists:
                result["type"] = "directory" if full_path.is_dir() else "file"
            
            return result
        except ValueError:
            # Path is outside both directories
            return {
                "success": True,
                "path": path,
                "exists": False,
            }
    
    async def _delete(self, path: str) -> Dict[str, Any]:
        """Delete file or directory (Sandbox only)."""
        full_path, _ = self._validate_and_resolve_path(path, require_write=True)
        
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
    
    async def _copy_to_sandbox(self, source: str, dest: str | None = None) -> Dict[str, Any]:
        """
        Copy file from ReadData to Sandbox.
        
        Args:
            source: Path in ReadData
            dest: Destination path in Sandbox (optional, uses same name if not provided)
        """
        import shutil
        
        # Source must be readable (can be in ReadData or Sandbox)
        source_path, source_type = self._validate_and_resolve_path(source, require_write=False)
        
        if not source_path.exists():
            return {
                "success": False,
                "error": f"Source file not found: {source}",
            }
        
        if not source_path.is_file():
            return {
                "success": False,
                "error": f"Source must be a file, not directory: {source}",
            }
        
        # Destination must be in Sandbox
        if dest is None:
            dest = source_path.name
        
        dest_path, _ = self._validate_and_resolve_path(dest, require_write=True)
        
        # Create parent directories if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        
        return {
            "success": True,
            "action": "copy_to_sandbox",
            "source": source,
            "dest": str(dest_path.relative_to(self.sandbox_root)),
            "size_bytes": dest_path.stat().st_size,
        }
