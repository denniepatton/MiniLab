from __future__ import annotations

from typing import Any, Dict, List
import subprocess
import os

from . import Tool


class TerminalTool(Tool):
    """
    Execute terminal commands.
    Use with caution - should have safeguards in production.
    """

    def __init__(self, allowed_commands: List[str] | None = None):
        super().__init__(
            name="terminal",
            description="Execute terminal commands for data analysis and scripting"
        )
        # Whitelist of allowed command prefixes for safety
        self.allowed_commands = allowed_commands or [
            "python", "R", "bash", "sh",
            "git", "ls", "cat", "grep", "awk", "sed",
            "jupyter", "pytest",
        ]

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a terminal command.
        
        Args:
            command: Command to execute
            working_dir: Working directory for the command
            timeout: Timeout in seconds
            
        Returns:
            Dict with stdout, stderr, and return code
        """
        # Basic safety check
        cmd_start = command.split()[0] if command.split() else ""
        if not any(cmd_start.startswith(allowed) for allowed in self.allowed_commands):
            return {
                "status": "error",
                "message": f"Command '{cmd_start}' not in allowed list",
                "stdout": "",
                "stderr": "",
                "returncode": -1,
            }

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir,
            )
            
            return {
                "status": "success" if result.returncode == 0 else "error",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "message": f"Command timed out after {timeout}s",
                "stdout": "",
                "stderr": "",
                "returncode": -1,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "stdout": "",
                "stderr": "",
                "returncode": -1,
            }


class FilesystemTool(Tool):
    """
    Safe filesystem operations for reading/writing files.
    """

    def __init__(self, workspace_dir: str | None = None):
        super().__init__(
            name="filesystem",
            description="Read and write files in the workspace"
        )
        self.workspace_dir = workspace_dir or os.getcwd()

    async def execute(
        self,
        action: str,
        path: str,
        content: str | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute filesystem operation.
        
        Args:
            action: Action to perform ("read", "write", "list", "exists")
            path: File path (relative to workspace)
            content: Content to write (for write action)
            
        Returns:
            Dict with results
        """
        # Ensure path is within workspace (basic safety)
        full_path = os.path.join(self.workspace_dir, path)
        if not full_path.startswith(self.workspace_dir):
            return {
                "status": "error",
                "message": "Path must be within workspace directory"
            }

        try:
            if action == "read":
                with open(full_path, 'r') as f:
                    return {
                        "status": "success",
                        "path": path,
                        "content": f.read()
                    }
            
            elif action == "write":
                if content is None:
                    return {"status": "error", "message": "content required for write"}
                
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
                return {
                    "status": "success",
                    "path": path,
                    "message": "File written successfully"
                }
            
            elif action == "list":
                if os.path.isdir(full_path):
                    items = os.listdir(full_path)
                    return {
                        "status": "success",
                        "path": path,
                        "items": items
                    }
                else:
                    return {"status": "error", "message": "Path is not a directory"}
            
            elif action == "exists":
                return {
                    "status": "success",
                    "path": path,
                    "exists": os.path.exists(full_path)
                }
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {action}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }


class GitTool(Tool):
    """
    Git operations for version control.
    """

    def __init__(self, repo_path: str | None = None):
        super().__init__(
            name="git",
            description="Git version control operations"
        )
        self.repo_path = repo_path or os.getcwd()

    async def execute(
        self,
        action: str,
        message: str | None = None,
        branch: str | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute git operation.
        
        Args:
            action: Action to perform ("status", "commit", "branch", "log")
            message: Commit message (for commit action)
            branch: Branch name (for branch operations)
            
        Returns:
            Dict with results
        """
        try:
            if action == "status":
                result = subprocess.run(
                    ["git", "status", "--short"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                )
                return {
                    "status": "success",
                    "output": result.stdout
                }
            
            elif action == "commit":
                if not message:
                    return {"status": "error", "message": "commit message required"}
                
                # Stage all changes
                subprocess.run(
                    ["git", "add", "-A"],
                    cwd=self.repo_path,
                    check=True,
                )
                
                # Commit
                result = subprocess.run(
                    ["git", "commit", "-m", message],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                )
                return {
                    "status": "success",
                    "output": result.stdout
                }
            
            elif action == "log":
                result = subprocess.run(
                    ["git", "log", "--oneline", "-n", "10"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                )
                return {
                    "status": "success",
                    "output": result.stdout
                }
            
            elif action == "branch":
                result = subprocess.run(
                    ["git", "branch"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                )
                return {
                    "status": "success",
                    "output": result.stdout
                }
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {action}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
