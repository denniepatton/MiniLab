from __future__ import annotations

from typing import Any, Dict, List
import subprocess
import os
import re
from pathlib import Path

from . import Tool


class TerminalTool(Tool):
    """
    Execute terminal commands with path-based security.
    
    SECURITY MODEL:
    - ReadData/ is READ-ONLY (can cat, head, ls, but not write/delete)
    - Sandbox/ is READ-WRITE (full access)
    - Commands that would modify ReadData/ are blocked
    - Everything else is allowed
    
    This is the PRIMARY tool for agents to interact with the filesystem.
    Agents should use normal shell commands: mkdir, cp, cat, python, etc.
    """

    def __init__(self, workspace_root: str | Path | None = None):
        super().__init__(
            name="terminal",
            description="""Execute shell commands in the workspace.
Action: execute
Params: {command: "shell command"}

IMPORTANT:
- All commands run from workspace root (NOT /home/user or ~)
- Use RELATIVE paths: Sandbox/Project/script.py (not /home/user/...)
- To run Python: micromamba run -n minilab python Sandbox/path/script.py
- Create files with code_editor BEFORE trying to run them!

Examples:
- List data: ls -la ReadData/
- Run script: micromamba run -n minilab python Sandbox/MyProject/scripts/analysis.py
- Create dir: mkdir -p Sandbox/MyProject/outputs
- View file: head -50 ReadData/data.csv

RULES: ReadData/ = read-only. Sandbox/ = read-write.
DO NOT: cd to /home/user, use absolute paths outside workspace, run scripts that don't exist yet.

Example: {"tool": "terminal", "action": "execute", "params": {"command": "ls -la Sandbox/"}}"""
        )
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
    
    def _would_modify_readonly(self, command: str) -> bool:
        """
        Check if command would modify ReadData/ (read-only area).
        
        We block commands that would write to, delete from, or modify ReadData/.
        """
        # Commands that modify files
        write_commands = [
            r'\brm\b', r'\brmdir\b', r'\bmv\b', r'\bcp\b.*ReadData/',  # cp INTO ReadData
            r'\btouch\b', r'\bmkdir\b', r'\bchmod\b', r'\bchown\b',
            r'\becho\b.*>', r'\bcat\b.*>', r'\btee\b',  # redirections
            r'>\s*ReadData', r'>>\s*ReadData',  # explicit redirects to ReadData
        ]
        
        # Check if command targets ReadData/ for modification
        for pattern in write_commands:
            if re.search(pattern, command):
                # Now check if ReadData is the TARGET (not source)
                # For cp: "cp ReadData/x Sandbox/y" is OK, "cp x ReadData/y" is NOT
                if 'cp' in command:
                    parts = command.split()
                    if len(parts) >= 3:
                        dest = parts[-1]
                        if 'ReadData' in dest:
                            return True
                    continue
                
                # For other write commands, block if ReadData is mentioned
                if 'ReadData' in command:
                    return True
        
        # Check for Python/scripts that might write to ReadData
        # We can't fully prevent this, but we trust the agent to respect the rules
        
        return False

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int = 300,  # 5 minutes default for analysis scripts
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a terminal command.
        
        Args:
            command: Shell command to execute (mkdir, python, cat, etc.)
            working_dir: Working directory (defaults to workspace root)
            timeout: Timeout in seconds (default 5 min for scripts)
            
        Returns:
            Dict with success, stdout, stderr, returncode
        """
        # Security check: block modifications to ReadData/
        if self._would_modify_readonly(command):
            return {
                "success": False,
                "error": "Cannot modify ReadData/ - it is read-only. Use Sandbox/ for output.",
                "stdout": "",
                "stderr": "",
                "returncode": -1,
            }
        
        # Set working directory
        cwd = working_dir or str(self.workspace_root)
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            
            # Truncate very long output
            stdout = result.stdout
            stderr = result.stderr
            if len(stdout) > 10000:
                stdout = stdout[:5000] + "\n...[truncated]...\n" + stdout[-2000:]
            if len(stderr) > 5000:
                stderr = stderr[:2000] + "\n...[truncated]...\n" + stderr[-1000:]
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": result.returncode,
                }
            else:
                # Command failed - provide helpful error message
                error_msg = stderr.strip() if stderr else f"Command failed with exit code {result.returncode}"
                return {
                    "success": False,
                    "error": error_msg,
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": result.returncode,
                }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "stdout": "",
                "stderr": "",
                "returncode": -1,
            }
        except FileNotFoundError as e:
            return {
                "success": False,
                "error": f"Command or file not found: {e}",
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Command failed: {type(e).__name__}: {str(e)}",
                "stdout": "",
                "stderr": str(e),
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
