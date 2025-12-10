"""
Terminal Tool with PathGuard integration.

Provides shell command execution with strict security enforcement.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any, Optional
from pydantic import Field

from .base import Tool, ToolInput, ToolOutput, ToolError
from ..security import PathGuard, AccessDenied


class ExecuteInput(ToolInput):
    """Input for executing a shell command."""
    command: str = Field(..., description="The shell command to execute")
    working_dir: Optional[str] = Field(None, description="Working directory (defaults to Sandbox/)")
    timeout: int = Field(300, description="Timeout in seconds")
    env: Optional[dict[str, str]] = Field(None, description="Additional environment variables")


class TerminalOutput(ToolOutput):
    """Output for terminal operations."""
    command: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    return_code: Optional[int] = None
    working_dir: Optional[str] = None


class TerminalTool(Tool):
    """
    Shell command execution with security enforcement.
    
    Commands are validated against PathGuard before execution.
    Write operations must target Sandbox/.
    """
    
    name = "terminal"
    description = "Execute shell commands (write operations restricted to Sandbox/)"
    
    def __init__(self, agent_id: str, workspace_root: Path, **kwargs):
        super().__init__(agent_id, **kwargs)
        self.workspace_root = workspace_root
        self.path_guard = PathGuard.get_instance()
    
    def get_actions(self) -> dict[str, str]:
        return {
            "execute": "Execute a shell command",
        }
    
    def get_input_schema(self, action: str) -> type[ToolInput]:
        schemas = {
            "execute": ExecuteInput,
        }
        if action not in schemas:
            raise ToolError(self.name, action, f"Unknown action: {action}")
        return schemas[action]
    
    async def execute(self, action: str, params: dict[str, Any]) -> TerminalOutput:
        """Execute a terminal action."""
        validated = self.validate_input(action, params)
        
        try:
            if action == "execute":
                return await self._execute(validated)
            else:
                raise ToolError(self.name, action, f"Unknown action: {action}")
        except AccessDenied as e:
            return TerminalOutput(success=False, error=str(e))
        except Exception as e:
            return TerminalOutput(success=False, error=f"Operation failed: {e}")
    
    async def _execute(self, params: ExecuteInput) -> TerminalOutput:
        """Execute a shell command."""
        # Validate command through PathGuard
        self.path_guard.validate_execute(params.command, self.agent_id)
        
        # Determine working directory
        if params.working_dir:
            cwd = Path(params.working_dir)
            if not cwd.is_absolute():
                cwd = self.workspace_root / cwd
        else:
            # Default to Sandbox/
            cwd = self.workspace_root / "Sandbox"
            cwd.mkdir(exist_ok=True)
        
        # Build environment
        import os
        env = os.environ.copy()
        if params.env:
            env.update(params.env)
        
        try:
            process = await asyncio.create_subprocess_shell(
                params.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=params.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return TerminalOutput(
                    success=False,
                    command=params.command,
                    error=f"Command timed out after {params.timeout} seconds",
                    working_dir=str(cwd),
                )
            
            return TerminalOutput(
                success=process.returncode == 0,
                command=params.command,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                return_code=process.returncode,
                working_dir=str(cwd),
            )
            
        except Exception as e:
            return TerminalOutput(
                success=False,
                command=params.command,
                error=f"Failed to execute command: {e}",
                working_dir=str(cwd),
            )
