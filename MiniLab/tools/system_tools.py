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


# NOTE: FilesystemTool is DEPRECATED. Use DualModeFileSystemTool from filesystem_advanced.py instead.


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
