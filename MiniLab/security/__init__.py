"""
MiniLab Security Module

Provides centralized access control and path validation for all agent operations.
"""

from .path_guard import PathGuard, AccessLevel, AccessDenied, AgentPermissions
from .policy_engine import (
    PolicyEngine,
    PolicySet,
    PolicyRule,
    PolicyContext,
    PolicyResult,
    PolicyDecision,
    SecurityScope,
    FileAuditLog,
    create_default_policy,
    create_permissive_policy,
)
from .sandbox import (
    Sandbox,
    SandboxConfig,
    IsolatedSandbox,
    CommandResult,
    CommandRisk,
    CommandPattern,
)

__all__ = [
    # Path guard
    "PathGuard",
    "AccessLevel",
    "AccessDenied",
    "AgentPermissions",
    # Policy engine
    "PolicyEngine",
    "PolicySet",
    "PolicyRule",
    "PolicyContext",
    "PolicyResult",
    "PolicyDecision",
    "SecurityScope",
    "FileAuditLog",
    "create_default_policy",
    "create_permissive_policy",
    # Sandbox
    "Sandbox",
    "SandboxConfig",
    "IsolatedSandbox",
    "CommandResult",
    "CommandRisk",
    "CommandPattern",
]
