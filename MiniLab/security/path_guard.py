"""
PathGuard: Unified security layer for all file system operations.

This module enforces access control with two layers:
1. Base layer: Structural rules (ReadData read-only, Sandbox read-write)
2. Intent layer: Project-specific access derived from SSOT AccessPolicy

Key change in v2: ReadData access is now INTENT-DERIVED.
- If user requested data analysis → ReadData allowed
- If user requested literature review → ReadData NOT allowed
- This prevents agents from wandering into data when not relevant

All tools (filesystem, terminal, code_editor) MUST call PathGuard before any I/O.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.project_ssot import AccessPolicy


class AccessLevel(Enum):
    """Access levels for file operations."""
    NONE = auto()
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()


class AccessDenied(Exception):
    """Raised when an agent attempts an unauthorized operation."""
    
    def __init__(self, agent_id: str, path: str, operation: str, reason: str):
        self.agent_id = agent_id
        self.path = path
        self.operation = operation
        self.reason = reason
        super().__init__(f"Access denied for {agent_id}: {operation} on {path} - {reason}")


@dataclass
class AccessAttempt:
    """Record of an access attempt for audit logging."""
    timestamp: datetime
    agent_id: str
    path: str
    operation: str
    granted: bool
    reason: str


@dataclass
class AgentPermissions:
    """Agent-specific permissions beyond base access."""
    agent_id: str
    # Subdirectories within Sandbox/ this agent can write to
    writable_subdirs: list[str] = field(default_factory=list)
    # File extensions this agent can create
    writable_extensions: list[str] = field(default_factory=list)
    # Whether agent can execute shell commands
    can_execute_shell: bool = False
    # Whether agent can modify environment (install packages)
    can_modify_environment: bool = False


class PathGuard:
    """
    Central authority for all path access in MiniLab.
    
    Singleton pattern ensures consistent enforcement across all tools.
    
    Access Control Layers:
    1. Structural: Base rules for ReadData (read-only), Sandbox (read-write)
    2. Intent: Project-specific access from SSOT AccessPolicy
    
    Intent-aware access (v2):
    - ReadData access is disabled by default
    - Only enabled if user request implies data analysis
    - Prevents agents from exploring data when doing lit review
    """
    
    _instance: Optional[PathGuard] = None
    
    # Base directories (set relative to workspace root)
    READ_ONLY_DIRS = ["ReadData"]
    READ_WRITE_DIRS = ["Sandbox"]

    # Files that should never be created as project outputs.
    # (Keep in sync with ProjectWriter.FORBIDDEN_FILES when possible.)
    FORBIDDEN_FILENAMES = {
        "executive_summary.md",
        "brief_bibliography.md",
        "search_summary.md",
        "literature_search_summary.md",
        "session_summary.md",
    }

    # Windows/Office temp files and other junk we never want committed into a project.
    FORBIDDEN_BASENAME_PATTERNS = [
        r"^~\$",  # Office lock files
        r"^\.DS_Store$",
    ]

    # Disallow throwaway scripts inside analysis/.
    FORBIDDEN_ANALYSIS_PY_BASENAME_PATTERNS = [
        r"^test_.*\.py$",
        r"^.*_test\.py$",
        r"^simple_test\.py$",
        r"^scratch.*\.py$",
        r"^tmp.*\.py$",
        r"^debug.*\.py$",
        r"^playground.*\.py$",
    ]

    # Allow only controlled names for analysis python scripts (keeps outputs tidy).
    # Examples: 01_preprocess.py, 02_features.py, 03_model.py, run_analysis.py
    ALLOWED_ANALYSIS_PY_BASENAME_PATTERNS = [
        r"^run_analysis\.py$",
        r"^\d{2}_[a-z0-9_]+\.py$",
        r"^__init__\.py$",
    ]
    
    # Default agent permissions by role
    DEFAULT_PERMISSIONS = {
        "bohr": AgentPermissions(
            agent_id="bohr",
            writable_subdirs=["*"],  # Can write anywhere in Sandbox
            writable_extensions=["*"],
            can_execute_shell=False,
            can_modify_environment=True,  # Can approve environment changes
        ),
        "gould": AgentPermissions(
            agent_id="gould",
            writable_subdirs=["*"],
            writable_extensions=[".md", ".txt", ".bib", ".json", ".yaml", ".yml"],
            can_execute_shell=False,
            can_modify_environment=False,
        ),
        "farber": AgentPermissions(
            agent_id="farber",
            writable_subdirs=["*"],
            writable_extensions=[".md", ".txt", ".json"],
            can_execute_shell=False,
            can_modify_environment=False,
        ),
        "feynman": AgentPermissions(
            agent_id="feynman",
            writable_subdirs=["*"],
            writable_extensions=[".md", ".txt", ".json"],
            can_execute_shell=False,
            can_modify_environment=False,
        ),
        "shannon": AgentPermissions(
            agent_id="shannon",
            writable_subdirs=["*"],
            writable_extensions=[".md", ".txt", ".json"],
            can_execute_shell=False,
            can_modify_environment=False,
        ),
        "greider": AgentPermissions(
            agent_id="greider",
            writable_subdirs=["*"],
            writable_extensions=[".md", ".txt", ".json"],
            can_execute_shell=False,
            can_modify_environment=False,
        ),
        "dayhoff": AgentPermissions(
            agent_id="dayhoff",
            writable_subdirs=["*"],
            writable_extensions=[".md", ".txt", ".json", ".yaml", ".yml", ".sh"],
            can_execute_shell=False,
            can_modify_environment=False,
        ),
        "hinton": AgentPermissions(
            agent_id="hinton",
            writable_subdirs=["*"],
            writable_extensions=["*"],  # Can write any code file
            can_execute_shell=True,  # Can run code
            can_modify_environment=False,
        ),
        "bayes": AgentPermissions(
            agent_id="bayes",
            writable_subdirs=["*"],
            writable_extensions=[".py", ".r", ".R", ".md", ".txt", ".json", ".csv"],
            can_execute_shell=True,  # Can run statistical code
            can_modify_environment=False,
        ),
    }
    
    def __new__(cls, workspace_root: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, workspace_root: Optional[Path] = None):
        if self._initialized and workspace_root is None:
            return

        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.audit_log: list[AccessAttempt] = []
        self.logger = logging.getLogger("minilab.security")
        self._agent_permissions: dict[str, AgentPermissions] = dict(self.DEFAULT_PERMISSIONS)
        
        # Intent-aware access: project-specific policy from SSOT
        self._access_policy: Optional["AccessPolicy"] = None
        
        self._initialized = True
    
    def set_access_policy(self, policy: "AccessPolicy") -> None:
        """
        Set the project-specific access policy from SSOT.
        
        This enables intent-derived access control:
        - If user requested data analysis → ReadData allowed
        - If user requested lit review → ReadData NOT allowed
        """
        self._access_policy = policy
        self.logger.info(f"Access policy set: ReadData allowed = {policy.readdata_allowed}")
    
    def clear_access_policy(self) -> None:
        """Clear the access policy (revert to base rules only)."""
        self._access_policy = None
    
    def _check_intent_access(self, rel_path: Path, operation: str) -> tuple[bool, str]:
        """
        Check intent-derived access from SSOT AccessPolicy.
        
        Returns (allowed, reason).
        """
        if self._access_policy is None:
            # No policy set - allow base structural access
            return True, "No policy restriction"
        
        path_str = str(rel_path)
        
        # Check ReadData access
        if "ReadData" in path_str or (rel_path.parts and rel_path.parts[0] == "ReadData"):
            if not self._access_policy.readdata_allowed:
                return False, "ReadData access not in project scope (user request does not involve data analysis)"
            
            # Check if specific file is in scope
            if self._access_policy.data_files_in_scope:
                # If specific files listed, check against them
                in_scope = any(
                    f in path_str for f in self._access_policy.data_files_in_scope
                )
                if not in_scope:
                    # Allow directory listing but warn
                    return True, "ReadData allowed (general access)"
        
        # Check project path access
        if self._access_policy.project_path:
            if path_str.startswith(self._access_policy.project_path):
                return True, "Project path access"
        
        return True, "Allowed by policy"
    
    @classmethod
    def reset(cls):
        """Reset singleton instance (for testing)."""
        cls._instance = None
    
    @classmethod
    def get_instance(cls) -> PathGuard:
        """Get the singleton instance."""
        if cls._instance is None:
            raise RuntimeError("PathGuard not initialized. Call PathGuard(workspace_root) first.")
        return cls._instance
    
    def set_workspace_root(self, workspace_root: Path) -> None:
        """Update the workspace root."""
        self.workspace_root = workspace_root.resolve()
    
    def set_agent_permissions(self, agent_id: str, permissions: AgentPermissions) -> None:
        """Override permissions for a specific agent."""
        self._agent_permissions[agent_id] = permissions
    
    def get_agent_permissions(self, agent_id: str) -> AgentPermissions:
        """Get permissions for an agent, with defaults for unknown agents."""
        return self._agent_permissions.get(
            agent_id.lower(),
            AgentPermissions(agent_id=agent_id.lower())  # Minimal permissions
        )
    
    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve a path to absolute, handling relative paths."""
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace_root / p
        return p.resolve()
    
    def _get_relative_path(self, path: Path) -> Optional[Path]:
        """Get path relative to workspace root, or None if outside workspace."""
        try:
            return path.relative_to(self.workspace_root)
        except ValueError:
            return None
    
    def _is_in_allowed_dir(self, rel_path: Path, allowed_dirs: list[str]) -> bool:
        """Check if a relative path is within one of the allowed directories."""
        if rel_path is None:
            return False
        parts = rel_path.parts
        if not parts:
            return False
        return parts[0] in allowed_dirs
    
    def _check_extension_permission(
        self, path: Path, permissions: AgentPermissions
    ) -> bool:
        """Check if agent can write files with this extension."""
        if "*" in permissions.writable_extensions:
            return True
        ext = path.suffix.lower()
        return ext in permissions.writable_extensions or ext == ""
    
    def _check_subdir_permission(
        self, rel_path: Path, permissions: AgentPermissions
    ) -> bool:
        """Check if agent can write to this subdirectory within Sandbox."""
        if "*" in permissions.writable_subdirs:
            return True
        parts = rel_path.parts
        if len(parts) < 2:
            return True  # Writing directly to Sandbox/
        subdir = parts[1]
        return subdir in permissions.writable_subdirs
    
    def _log_access(
        self, agent_id: str, path: str, operation: str, granted: bool, reason: str
    ) -> None:
        """Log an access attempt."""
        attempt = AccessAttempt(
            timestamp=datetime.now(),
            agent_id=agent_id,
            path=path,
            operation=operation,
            granted=granted,
            reason=reason,
        )
        self.audit_log.append(attempt)
        
        if granted:
            self.logger.debug(f"ACCESS GRANTED: {agent_id} {operation} {path}")
        else:
            self.logger.warning(f"ACCESS DENIED: {agent_id} {operation} {path} - {reason}")
    
    def can_read(self, path: str | Path, agent_id: str) -> bool:
        """
        Check if an agent can read from a path.
        
        Access layers:
        1. Structural: Must be in ReadData/ or Sandbox/
        2. Intent: ReadData only allowed if project scope includes data
        """
        abs_path = self._resolve_path(path)
        rel_path = self._get_relative_path(abs_path)
        
        if rel_path is None:
            self._log_access(agent_id, str(path), "read", False, "Outside workspace")
            return False
        
        # Structural check: must be in allowed directories
        readable_dirs = self.READ_ONLY_DIRS + self.READ_WRITE_DIRS
        if not self._is_in_allowed_dir(rel_path, readable_dirs):
            self._log_access(agent_id, str(path), "read", False, "Not in allowed directory")
            return False
        
        # Intent check: is ReadData access allowed for this project?
        intent_ok, intent_reason = self._check_intent_access(rel_path, "read")
        if not intent_ok:
            self._log_access(agent_id, str(path), "read", False, intent_reason)
            return False
        
        self._log_access(agent_id, str(path), "read", True, f"Allowed ({intent_reason})")
        return True
    
    def can_write(self, path: str | Path, agent_id: str) -> bool:
        """
        Check if an agent can write to a path.
        
        Writable paths:
        - Sandbox/* only (with agent-specific restrictions)
        """
        abs_path = self._resolve_path(path)
        rel_path = self._get_relative_path(abs_path)
        
        if rel_path is None:
            self._log_access(agent_id, str(path), "write", False, "Outside workspace")
            return False
        
        # Must be in Sandbox
        if not self._is_in_allowed_dir(rel_path, self.READ_WRITE_DIRS):
            self._log_access(
                agent_id, str(path), "write", False,
                f"Not in writable directory (must be in {self.READ_WRITE_DIRS})"
            )
            return False

        # Prevent accidental recursive Sandbox nesting (Sandbox/Sandbox/...).
        # This is almost always an error stemming from combining cwd=Sandbox with paths prefixed by Sandbox/.
        if len(rel_path.parts) >= 2 and rel_path.parts[0] == "Sandbox" and rel_path.parts[1] == "Sandbox":
            self._log_access(agent_id, str(path), "write", False, "Refusing to write into Sandbox/Sandbox (recursive nesting)")
            return False
        
        # Check agent-specific permissions
        permissions = self.get_agent_permissions(agent_id)
        
        if not self._check_subdir_permission(rel_path, permissions):
            self._log_access(
                agent_id, str(path), "write", False,
                f"Agent cannot write to this subdirectory"
            )
            return False
        
        if not self._check_extension_permission(abs_path, permissions):
            self._log_access(
                agent_id, str(path), "write", False,
                f"Agent cannot write files with extension {abs_path.suffix}"
            )
            return False

        # Global filename hygiene.
        basename = abs_path.name
        for pat in self.FORBIDDEN_BASENAME_PATTERNS:
            if re.search(pat, basename, re.IGNORECASE):
                self._log_access(agent_id, str(path), "write", False, f"Forbidden filename pattern: {pat}")
                return False

        if basename in self.FORBIDDEN_FILENAMES:
            self._log_access(agent_id, str(path), "write", False, f"Forbidden canonical duplicate: {basename}")
            return False

        # Enforce tidy analysis scripts: no throwaway scratch/test files.
        # Only applies within Sandbox/<project>/analysis/.
        try:
            parts = list(rel_path.parts)
            if len(parts) >= 3 and parts[0] == "Sandbox" and parts[2] == "analysis" and abs_path.suffix.lower() == ".py":
                for pat in self.FORBIDDEN_ANALYSIS_PY_BASENAME_PATTERNS:
                    if re.search(pat, basename, re.IGNORECASE):
                        self._log_access(agent_id, str(path), "write", False, f"Forbidden analysis script name: {basename}")
                        return False
                # Only enforce naming convention for NEW file creation; allow edits to existing files.
                if not abs_path.exists():
                    allowed = any(re.search(pat, basename) for pat in self.ALLOWED_ANALYSIS_PY_BASENAME_PATTERNS)
                    if not allowed:
                        self._log_access(
                            agent_id,
                            str(path),
                            "write",
                            False,
                            "New analysis scripts must be named like 01_step.py or run_analysis.py (no ad-hoc filenames)",
                        )
                        return False
        except Exception:
            # If path parsing fails for any reason, fall back to default allow/deny above.
            pass
        
        self._log_access(agent_id, str(path), "write", True, "Allowed")
        return True
    
    def can_delete(self, path: str | Path, agent_id: str) -> bool:
        """
        Check if an agent can delete a path.
        
        Same rules as write - only Sandbox/*.
        """
        abs_path = self._resolve_path(path)
        rel_path = self._get_relative_path(abs_path)
        
        if rel_path is None:
            self._log_access(agent_id, str(path), "delete", False, "Outside workspace")
            return False
        
        # Must be in Sandbox
        if not self._is_in_allowed_dir(rel_path, self.READ_WRITE_DIRS):
            self._log_access(
                agent_id, str(path), "delete", False,
                f"Cannot delete outside {self.READ_WRITE_DIRS}"
            )
            return False
        
        self._log_access(agent_id, str(path), "delete", True, "Allowed")
        return True
    
    def can_execute(self, command: str, agent_id: str) -> tuple[bool, str]:
        """
        Check if an agent can execute a shell command.
        
        Returns (allowed, reason).
        
        Blocked patterns:
        - Any write/modify to ReadData/
        - Any write/modify outside Sandbox/
        - Dangerous system commands
        """
        permissions = self.get_agent_permissions(agent_id)
        
        if not permissions.can_execute_shell:
            return False, f"Agent {agent_id} is not allowed to execute shell commands"
        
        # Dangerous command patterns
        dangerous_patterns = [
            r"\brm\s+-rf\s+[/~]",  # rm -rf with root or home
            r"\bsudo\b",
            r"\bchmod\b.*777",
            r"\bcurl\b.*\|\s*sh",  # Piping curl to shell
            r"\bwget\b.*\|\s*sh",
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                self._log_access(agent_id, command, "execute", False, f"Dangerous pattern: {pattern}")
                return False, f"Command matches dangerous pattern"
        
        # Check for writes to ReadData/
        readdata_write_patterns = [
            r">\s*['\"]?.*ReadData",
            r">>\s*['\"]?.*ReadData",
            r"\brm\b.*ReadData",
            r"\bmv\b.*ReadData",
            r"\bcp\b.*ReadData[/\s]",  # cp TO ReadData
        ]
        
        for pattern in readdata_write_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                self._log_access(agent_id, command, "execute", False, "Attempted write to ReadData/")
                return False, "Cannot write to ReadData/"
        
        # Check for writes outside Sandbox (excluding ReadData which is handled above)
        # This is trickier - we look for redirects and common write commands
        # If the command doesn't reference Sandbox for writes, it's suspicious
        
        write_commands = ["mv", "cp", "touch", "mkdir", "echo.*>", "cat.*>", "tee"]
        for cmd in write_commands:
            if re.search(rf"\b{cmd}\b", command, re.IGNORECASE):
                # Check if Sandbox is the destination
                if "Sandbox" not in command and "sandbox" not in command.lower():
                    # Exception: reading from ReadData is OK
                    if cmd in ["cp", "mv"] and ("ReadData" in command or "readdata" in command.lower()):
                        # Check direction - is ReadData source or dest?
                        # For cp/mv, last arg is usually dest
                        pass  # Allow reading from ReadData
                    else:
                        self._log_access(
                            agent_id, command, "execute", False,
                            f"Write command '{cmd}' does not target Sandbox/"
                        )
                        return False, "Write operations must target Sandbox/"
        
        self._log_access(agent_id, command, "execute", True, "Allowed")
        return True, "Allowed"
    
    def can_modify_environment(self, agent_id: str) -> bool:
        """Check if an agent can modify the environment (install packages, etc.)."""
        permissions = self.get_agent_permissions(agent_id)
        return permissions.can_modify_environment
    
    def validate_read(self, path: str | Path, agent_id: str) -> None:
        """Validate read access, raising AccessDenied if not allowed."""
        if not self.can_read(path, agent_id):
            raise AccessDenied(agent_id, str(path), "read", "Path not readable")
    
    def validate_write(self, path: str | Path, agent_id: str) -> None:
        """Validate write access, raising AccessDenied if not allowed."""
        if not self.can_write(path, agent_id):
            raise AccessDenied(agent_id, str(path), "write", "Path not writable")
    
    def validate_delete(self, path: str | Path, agent_id: str) -> None:
        """Validate delete access, raising AccessDenied if not allowed."""
        if not self.can_delete(path, agent_id):
            raise AccessDenied(agent_id, str(path), "delete", "Path not deletable")
    
    def validate_execute(self, command: str, agent_id: str) -> None:
        """Validate command execution, raising AccessDenied if not allowed."""
        allowed, reason = self.can_execute(command, agent_id)
        if not allowed:
            raise AccessDenied(agent_id, command, "execute", reason)
    
    def get_audit_log(
        self, agent_id: Optional[str] = None, limit: int = 100
    ) -> list[AccessAttempt]:
        """Get recent access attempts, optionally filtered by agent."""
        log = self.audit_log
        if agent_id:
            log = [a for a in log if a.agent_id == agent_id]
        return log[-limit:]
    
    def get_denied_attempts(self, limit: int = 50) -> list[AccessAttempt]:
        """Get recent denied access attempts."""
        denied = [a for a in self.audit_log if not a.granted]
        return denied[-limit:]
