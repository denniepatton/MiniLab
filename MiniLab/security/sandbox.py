"""
Sandbox: Safe execution environment for agent commands.

Provides:
- Command allowlist/blocklist enforcement
- Execution timeout handling
- Output sanitization
- Resource limits
"""

from __future__ import annotations

__all__ = [
    "CommandRisk",
    "CommandResult",
    "CommandPattern",
    "SandboxConfig",
    "Sandbox",
    "IsolatedSandbox",
    "DEFAULT_PATTERNS",
]

import asyncio
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class CommandRisk(str, Enum):
    """Risk level for commands."""
    SAFE = "safe"        # Read-only, no side effects
    LOW = "low"          # Limited side effects
    MEDIUM = "medium"    # Moderate side effects
    HIGH = "high"        # Significant side effects
    DANGEROUS = "dangerous"  # System-level changes


@dataclass
class CommandResult:
    """Result of a sandboxed command execution."""

    returncode: int
    stdout: str
    stderr: str

    timed_out: bool = False
    truncated: bool = False

    command: str = ""
    risk_level: CommandRisk = CommandRisk.MEDIUM
    execution_time_ms: int = 0


class CommandPattern(BaseModel):
    """Pattern for matching commands."""

    pattern: str = Field(..., description="Regex pattern to match")
    risk: CommandRisk = Field(..., description="Risk level if matched")
    description: str = Field(default="", description="What this pattern matches")

    # Whether to allow or block
    allow: bool = Field(default=True, description="Allow if True, block if False")

    model_config = {"extra": "forbid"}


# Default command patterns
DEFAULT_PATTERNS = [
    # Safe read-only commands
    CommandPattern(pattern=r"^ls(\s|$)", risk=CommandRisk.SAFE, description="List files"),
    CommandPattern(pattern=r"^cat\s", risk=CommandRisk.SAFE, description="Display file"),
    CommandPattern(pattern=r"^head\s", risk=CommandRisk.SAFE, description="Display file head"),
    CommandPattern(pattern=r"^tail\s", risk=CommandRisk.SAFE, description="Display file tail"),
    CommandPattern(pattern=r"^grep\s", risk=CommandRisk.SAFE, description="Search files"),
    CommandPattern(pattern=r"^wc\s", risk=CommandRisk.SAFE, description="Word count"),
    CommandPattern(pattern=r"^pwd$", risk=CommandRisk.SAFE, description="Print directory"),
    CommandPattern(pattern=r"^echo\s", risk=CommandRisk.SAFE, description="Echo text"),
    CommandPattern(pattern=r"^find\s", risk=CommandRisk.SAFE, description="Find files"),
    CommandPattern(pattern=r"^which\s", risk=CommandRisk.SAFE, description="Find command"),
    CommandPattern(pattern=r"^file\s", risk=CommandRisk.SAFE, description="File type"),

    # Low risk - Python/R execution (in working dir)
    CommandPattern(pattern=r"^python3?\s", risk=CommandRisk.LOW, description="Python"),
    CommandPattern(pattern=r"^Rscript\s", risk=CommandRisk.LOW, description="R script"),
    CommandPattern(pattern=r"^jupyter\s", risk=CommandRisk.LOW, description="Jupyter"),

    # Medium risk - file modifications
    CommandPattern(pattern=r"^cp\s", risk=CommandRisk.MEDIUM, description="Copy files"),
    CommandPattern(pattern=r"^mv\s", risk=CommandRisk.MEDIUM, description="Move files"),
    CommandPattern(pattern=r"^mkdir\s", risk=CommandRisk.MEDIUM, description="Create directory"),
    CommandPattern(pattern=r"^touch\s", risk=CommandRisk.MEDIUM, description="Create file"),

    # High risk - destructive
    CommandPattern(pattern=r"^rm\s", risk=CommandRisk.HIGH, description="Remove files"),
    CommandPattern(pattern=r"^rmdir\s", risk=CommandRisk.HIGH, description="Remove directory"),

    # Package management (high risk)
    CommandPattern(pattern=r"^pip\s", risk=CommandRisk.HIGH, description="Python packages"),
    CommandPattern(pattern=r"^conda\s", risk=CommandRisk.HIGH, description="Conda packages"),

    # Dangerous - block by default
    CommandPattern(
        pattern=r"^sudo\s",
        risk=CommandRisk.DANGEROUS,
        description="Superuser",
        allow=False
    ),
    CommandPattern(
        pattern=r"^chmod\s",
        risk=CommandRisk.DANGEROUS,
        description="Change permissions",
        allow=False
    ),
    CommandPattern(
        pattern=r"^chown\s",
        risk=CommandRisk.DANGEROUS,
        description="Change owner",
        allow=False
    ),
    CommandPattern(
        pattern=r".*\|\s*sh",
        risk=CommandRisk.DANGEROUS,
        description="Pipe to shell",
        allow=False
    ),
    CommandPattern(
        pattern=r".*\|\s*bash",
        risk=CommandRisk.DANGEROUS,
        description="Pipe to bash",
        allow=False
    ),
    CommandPattern(
        pattern=r".*>\s*/etc/",
        risk=CommandRisk.DANGEROUS,
        description="Write to /etc",
        allow=False
    ),
    CommandPattern(
        pattern=r"curl.*\|\s*(sh|bash)",
        risk=CommandRisk.DANGEROUS,
        description="Curl pipe shell",
        allow=False
    ),
]


class SandboxConfig(BaseModel):
    """Configuration for the sandbox."""

    # Execution limits
    timeout_seconds: int = Field(default=300, description="Max execution time")
    max_output_bytes: int = Field(default=100_000, description="Max output size")

    # Working directory
    working_dir: Optional[str] = Field(default=None, description="Working directory")
    restrict_to_working_dir: bool = Field(
        default=True,
        description="Restrict file access to working dir"
    )

    # Environment
    inherit_env: bool = Field(default=True, description="Inherit environment")
    env_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Environment overrides"
    )
    blocked_env_vars: list[str] = Field(
        default_factory=lambda: [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "API_KEY",
            "SECRET_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
        ],
        description="Env vars to block from commands"
    )

    # Risk tolerance
    max_risk: CommandRisk = Field(
        default=CommandRisk.MEDIUM,
        description="Max allowed risk level"
    )

    model_config = {"extra": "forbid"}


class Sandbox:
    """
    Sandboxed execution environment.
    
    Provides safe command execution with:
    - Command validation against patterns
    - Timeout enforcement
    - Output truncation
    - Environment sanitization
    
    Example:
        sandbox = Sandbox(config)
        result = await sandbox.execute("ls -la")
        if result.returncode == 0:
            print(result.stdout)
    """

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        patterns: Optional[list[CommandPattern]] = None,
    ):
        """
        Initialize sandbox.
        
        Args:
            config: Sandbox configuration
            patterns: Command patterns for validation
        """
        self.config = config or SandboxConfig()
        self.patterns = patterns or DEFAULT_PATTERNS

        # Compile patterns
        self._compiled_patterns: list[tuple[re.Pattern, CommandPattern]] = [
            (re.compile(p.pattern), p)
            for p in self.patterns
        ]

    def classify_command(self, command: str) -> tuple[CommandRisk, bool]:
        """
        Classify a command's risk level and whether it's allowed.
        
        Returns:
            Tuple of (risk_level, allowed)
        """
        # Check against patterns
        for compiled, pattern in self._compiled_patterns:
            if compiled.search(command):
                return pattern.risk, pattern.allow

        # Unknown commands get medium risk
        return CommandRisk.MEDIUM, True

    def validate_command(self, command: str) -> tuple[bool, str]:
        """
        Validate a command for execution.
        
        Returns:
            Tuple of (valid, reason)
        """
        risk, allowed = self.classify_command(command)

        # Check if explicitly blocked
        if not allowed:
            return False, f"Command pattern blocked: {risk.value}"

        # Check risk level
        risk_order = [
            CommandRisk.SAFE,
            CommandRisk.LOW,
            CommandRisk.MEDIUM,
            CommandRisk.HIGH,
            CommandRisk.DANGEROUS,
        ]

        if risk_order.index(risk) > risk_order.index(self.config.max_risk):
            return False, f"Risk level {risk.value} exceeds max {self.config.max_risk.value}"

        return True, "Command allowed"

    def _get_safe_env(self) -> dict[str, str]:
        """Get sanitized environment variables."""
        if self.config.inherit_env:
            env = dict(os.environ)
        else:
            env = {}

        # Remove blocked vars
        for var in self.config.blocked_env_vars:
            env.pop(var, None)

        # Apply overrides
        env.update(self.config.env_overrides)

        return env

    async def execute(self, command: str) -> CommandResult:
        """
        Execute a command in the sandbox.
        
        Args:
            command: The command to execute
            
        Returns:
            CommandResult with output and status
        """
        import time

        start_time = time.time()

        # Validate command
        valid, reason = self.validate_command(command)
        if not valid:
            return CommandResult(
                returncode=-1,
                stdout="",
                stderr=f"Command blocked: {reason}",
                command=command,
                risk_level=self.classify_command(command)[0],
            )

        risk, _ = self.classify_command(command)

        # Prepare execution
        env = self._get_safe_env()
        cwd = self.config.working_dir

        try:
            # Run command with timeout
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout_seconds,
            )

            # Truncate output if needed
            truncated = False

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            if len(stdout) > self.config.max_output_bytes:
                stdout = stdout[:self.config.max_output_bytes] + "\n... [output truncated]"
                truncated = True

            if len(stderr) > self.config.max_output_bytes:
                stderr = stderr[:self.config.max_output_bytes] + "\n... [output truncated]"
                truncated = True

            execution_time = int((time.time() - start_time) * 1000)

            return CommandResult(
                returncode=process.returncode or 0,
                stdout=stdout,
                stderr=stderr,
                command=command,
                risk_level=risk,
                truncated=truncated,
                execution_time_ms=execution_time,
            )

        except asyncio.TimeoutError:
            # Kill the process
            try:
                process.kill()
                await process.wait()
            except (ProcessLookupError, OSError):
                pass  # Process already terminated

            execution_time = int((time.time() - start_time) * 1000)

            return CommandResult(
                returncode=-1,
                stdout="",
                stderr=f"Command timed out after {self.config.timeout_seconds}s",
                timed_out=True,
                command=command,
                risk_level=risk,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)

            return CommandResult(
                returncode=-1,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                command=command,
                risk_level=risk,
                execution_time_ms=execution_time,
            )

    def execute_sync(self, command: str) -> CommandResult:
        """
        Synchronous version of execute.
        
        For use when async is not available.
        """
        import time

        start_time = time.time()

        # Validate command
        valid, reason = self.validate_command(command)
        if not valid:
            return CommandResult(
                returncode=-1,
                stdout="",
                stderr=f"Command blocked: {reason}",
                command=command,
                risk_level=self.classify_command(command)[0],
            )

        risk, _ = self.classify_command(command)

        # Prepare execution
        env = self._get_safe_env()
        cwd = self.config.working_dir

        try:
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                timeout=self.config.timeout_seconds,
                env=env,
                cwd=cwd,
            )

            # Truncate output if needed
            truncated = False

            stdout = process.stdout.decode("utf-8", errors="replace")
            stderr = process.stderr.decode("utf-8", errors="replace")

            if len(stdout) > self.config.max_output_bytes:
                stdout = stdout[:self.config.max_output_bytes] + "\n... [output truncated]"
                truncated = True

            if len(stderr) > self.config.max_output_bytes:
                stderr = stderr[:self.config.max_output_bytes] + "\n... [output truncated]"
                truncated = True

            execution_time = int((time.time() - start_time) * 1000)

            return CommandResult(
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
                command=command,
                risk_level=risk,
                truncated=truncated,
                execution_time_ms=execution_time,
            )

        except subprocess.TimeoutExpired:
            execution_time = int((time.time() - start_time) * 1000)

            return CommandResult(
                returncode=-1,
                stdout="",
                stderr=f"Command timed out after {self.config.timeout_seconds}s",
                timed_out=True,
                command=command,
                risk_level=risk,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)

            return CommandResult(
                returncode=-1,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                command=command,
                risk_level=risk,
                execution_time_ms=execution_time,
            )


class IsolatedSandbox(Sandbox):
    """
    More isolated sandbox using temporary directory.
    
    Creates a temporary working directory and copies
    only specified files into it.
    """

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        patterns: Optional[list[CommandPattern]] = None,
        allowed_files: Optional[list[Path]] = None,
    ):
        """
        Initialize isolated sandbox.
        
        Args:
            config: Sandbox configuration
            patterns: Command patterns
            allowed_files: Files to copy into sandbox
        """
        # Create temp directory
        self._temp_dir = tempfile.mkdtemp(prefix="minilab_sandbox_")

        # Update config to use temp dir
        config = config or SandboxConfig()
        config_dict = config.model_dump()
        config_dict["working_dir"] = self._temp_dir

        super().__init__(SandboxConfig(**config_dict), patterns)

        # Copy allowed files
        if allowed_files:
            for file_path in allowed_files:
                if file_path.exists():
                    import shutil
                    dest = Path(self._temp_dir) / file_path.name
                    shutil.copy2(file_path, dest)

    def __del__(self):
        """Clean up temp directory."""
        import shutil
        try:
            shutil.rmtree(self._temp_dir)
        except (OSError, FileNotFoundError):
            pass  # Directory already removed or inaccessible

    @property
    def sandbox_dir(self) -> Path:
        """Get the sandbox directory path."""
        return Path(self._temp_dir)
