"""
PolicyEngine: Deny-by-default security policy enforcement.

Provides:
- Declarative policy definitions
- Scope-based access control
- Audit logging
- Policy violation handling
"""

from __future__ import annotations

__all__ = [
    "SecurityScope",
    "PolicyDecision",
    "PolicyContext",
    "PolicyResult",
    "PolicyRule",
    "PolicySet",
    "FileAuditLog",
    "PolicyEngine",
    "create_default_policy",
    "create_permissive_policy",
]

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol
import json
import re

from pydantic import BaseModel, Field


class SecurityScope(str, Enum):
    """Security scopes (deny-by-default)."""

    # File system
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    DELETE_FILE = "delete_file"
    LIST_DIR = "list_dir"

    # Execution
    RUN_COMMAND = "run_command"
    SPAWN_PROCESS = "spawn_process"
    INSTALL_PACKAGE = "install_package"

    # Network
    HTTP_GET = "http_get"
    HTTP_POST = "http_post"
    EXTERNAL_API = "external_api"

    # Data
    READ_SENSITIVE = "read_sensitive"
    WRITE_SENSITIVE = "write_sensitive"
    EXPORT_DATA = "export_data"

    # System
    MODIFY_CONFIG = "modify_config"
    ACCESS_SECRETS = "access_secrets"
    ADMIN_OPERATION = "admin_operation"


class PolicyDecision(str, Enum):
    """Result of a policy evaluation."""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class PolicyContext:
    """Context for policy evaluation."""

    agent_id: str
    scope: SecurityScope
    resource: str  # e.g., file path, URL, command

    task_id: Optional[str] = None
    workflow_id: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PolicyResult:
    """Result of a policy evaluation."""

    decision: PolicyDecision
    reason: str

    context: Optional[PolicyContext] = None
    matched_rule: Optional[str] = None

    # For REQUIRE_APPROVAL
    approval_prompt: Optional[str] = None


class PolicyRule(BaseModel):
    """A single policy rule."""

    name: str = Field(..., description="Rule identifier")
    description: str = Field(default="", description="What this rule does")

    # Matching conditions
    scopes: list[SecurityScope] = Field(default_factory=list, description="Scopes this applies to")
    agents: list[str] = Field(default_factory=list, description="Agent IDs (empty = all)")
    resource_patterns: list[str] = Field(default_factory=list, description="Regex patterns for resources")

    # Decision
    decision: PolicyDecision = Field(..., description="What to do when matched")

    # Priority (higher = evaluated first)
    priority: int = Field(default=0, description="Rule priority")

    # Conditions
    conditions: dict[str, Any] = Field(default_factory=dict, description="Additional conditions")

    model_config = {"extra": "forbid"}

    def matches(self, context: PolicyContext) -> bool:
        """Check if this rule matches the context."""
        # Check scope
        if self.scopes and context.scope not in self.scopes:
            return False

        # Check agent
        if self.agents and context.agent_id not in self.agents:
            return False

        # Check resource patterns
        if self.resource_patterns:
            matched = False
            for pattern in self.resource_patterns:
                if re.match(pattern, context.resource):
                    matched = True
                    break
            if not matched:
                return False

        return True


class PolicySet(BaseModel):
    """A collection of policy rules."""

    name: str = Field(..., description="Policy set name")
    version: str = Field(default="1.0.0", description="Policy version")
    description: str = Field(default="", description="Policy set description")

    # Default decision when no rules match
    default_decision: PolicyDecision = Field(
        default=PolicyDecision.DENY,
        description="Default decision (should be DENY)"
    )

    rules: list[PolicyRule] = Field(default_factory=list, description="Policy rules")

    model_config = {"extra": "forbid"}

    def get_sorted_rules(self) -> list[PolicyRule]:
        """Get rules sorted by priority (highest first)."""
        return sorted(self.rules, key=lambda r: r.priority, reverse=True)


class AuditLog(Protocol):
    """Protocol for audit logging."""

    def log_access(
        self,
        context: PolicyContext,
        result: PolicyResult,
    ) -> None:
        """Log an access decision."""
        ...


class FileAuditLog:
    """Audit log that writes to a file."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_access(
        self,
        context: PolicyContext,
        result: PolicyResult,
    ) -> None:
        """Log an access decision to file."""
        entry = {
            "timestamp": context.timestamp.isoformat(),
            "agent_id": context.agent_id,
            "scope": context.scope.value,
            "resource": context.resource,
            "task_id": context.task_id,
            "decision": result.decision.value,
            "reason": result.reason,
            "matched_rule": result.matched_rule,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


class ApprovalCallback(Protocol):
    """Protocol for approval requests."""

    async def request_approval(
        self,
        prompt: str,
        context: PolicyContext,
    ) -> bool:
        """Request user approval. Returns True if approved."""
        ...


class PolicyEngine:
    """
    Central policy enforcement engine.
    
    Implements deny-by-default security model:
    - All operations require explicit permission
    - Audit logging of all access decisions
    - Support for user approval workflows
    
    Example:
        engine = PolicyEngine(policy_set)
        result = engine.evaluate(context)
        if result.decision == PolicyDecision.ALLOW:
            # Proceed with operation
        elif result.decision == PolicyDecision.REQUIRE_APPROVAL:
            # Request user approval
        else:
            # Deny operation
    """

    def __init__(
        self,
        policy_set: PolicySet,
        audit_log: Optional[AuditLog] = None,
        approval_callback: Optional[ApprovalCallback] = None,
    ):
        """
        Initialize policy engine.
        
        Args:
            policy_set: The policies to enforce
            audit_log: Optional audit logger
            approval_callback: Optional approval handler
        """
        self.policy_set = policy_set
        self.audit_log = audit_log
        self.approval_callback = approval_callback

        # Cache sorted rules
        self._sorted_rules = policy_set.get_sorted_rules()

        # Track pending approvals
        self._pending_approvals: dict[str, PolicyContext] = {}

    def evaluate(self, context: PolicyContext) -> PolicyResult:
        """
        Evaluate a policy decision.
        
        Args:
            context: The access context to evaluate
            
        Returns:
            PolicyResult with decision
        """
        # Find first matching rule
        for rule in self._sorted_rules:
            if rule.matches(context):
                result = PolicyResult(
                    decision=rule.decision,
                    reason=rule.description or f"Matched rule: {rule.name}",
                    context=context,
                    matched_rule=rule.name,
                )

                # Generate approval prompt if needed
                if rule.decision == PolicyDecision.REQUIRE_APPROVAL:
                    result.approval_prompt = self._generate_approval_prompt(context, rule)

                # Log the decision
                if self.audit_log:
                    self.audit_log.log_access(context, result)

                return result

        # No rule matched - use default (should be DENY)
        result = PolicyResult(
            decision=self.policy_set.default_decision,
            reason="No matching rule found - default deny",
            context=context,
        )

        if self.audit_log:
            self.audit_log.log_access(context, result)

        return result

    def _generate_approval_prompt(
        self,
        context: PolicyContext,
        rule: PolicyRule
    ) -> str:
        """Generate an approval prompt for the user."""
        return (
            f"Agent '{context.agent_id}' is requesting permission to:\n"
            f"  Operation: {context.scope.value}\n"
            f"  Resource: {context.resource}\n"
            f"  Reason: {rule.description or 'No reason provided'}\n\n"
            f"Allow this operation?"
        )

    async def evaluate_with_approval(
        self,
        context: PolicyContext
    ) -> PolicyResult:
        """
        Evaluate with automatic approval handling.
        
        If decision is REQUIRE_APPROVAL and a callback is configured,
        automatically requests approval.
        """
        result = self.evaluate(context)

        if (
            result.decision == PolicyDecision.REQUIRE_APPROVAL
            and self.approval_callback
            and result.approval_prompt
        ):
            approved = await self.approval_callback.request_approval(
                result.approval_prompt,
                context,
            )

            if approved:
                result.decision = PolicyDecision.ALLOW
                result.reason = f"User approved: {result.reason}"
            else:
                result.decision = PolicyDecision.DENY
                result.reason = f"User denied: {result.reason}"

        return result

    def check(
        self,
        agent_id: str,
        scope: SecurityScope,
        resource: str,
        task_id: Optional[str] = None,
    ) -> bool:
        """
        Quick check if an operation is allowed.
        
        Returns:
            True if allowed, False otherwise
        """
        context = PolicyContext(
            agent_id=agent_id,
            scope=scope,
            resource=resource,
            task_id=task_id,
        )
        result = self.evaluate(context)
        return result.decision == PolicyDecision.ALLOW

    def require(
        self,
        agent_id: str,
        scope: SecurityScope,
        resource: str,
        task_id: Optional[str] = None,
    ) -> None:
        """
        Require that an operation is allowed.
        
        Raises:
            PermissionError: If operation is not allowed
        """
        context = PolicyContext(
            agent_id=agent_id,
            scope=scope,
            resource=resource,
            task_id=task_id,
        )
        result = self.evaluate(context)

        if result.decision != PolicyDecision.ALLOW:
            raise PermissionError(
                f"Access denied: {scope.value} on {resource} - {result.reason}"
            )


# Pre-defined policy templates

def create_default_policy() -> PolicySet:
    """Create the default deny-by-default policy set."""
    return PolicySet(
        name="minilab_default",
        version="1.0.0",
        description="Default MiniLab security policy",
        default_decision=PolicyDecision.DENY,
        rules=[
            # Allow reading from ReadData/
            PolicyRule(
                name="allow_read_readdata",
                description="Allow reading files from ReadData directory",
                scopes=[SecurityScope.READ_FILE, SecurityScope.LIST_DIR],
                resource_patterns=[r".*ReadData/.*", r"ReadData/.*"],
                decision=PolicyDecision.ALLOW,
                priority=100,
            ),
            # Allow read/write in Sandbox/
            PolicyRule(
                name="allow_sandbox_access",
                description="Allow full access to Sandbox directory",
                scopes=[
                    SecurityScope.READ_FILE,
                    SecurityScope.WRITE_FILE,
                    SecurityScope.LIST_DIR,
                    SecurityScope.DELETE_FILE,
                ],
                resource_patterns=[r".*Sandbox/.*", r"Sandbox/.*"],
                decision=PolicyDecision.ALLOW,
                priority=100,
            ),
            # Require approval for command execution
            PolicyRule(
                name="approve_commands",
                description="Require approval for shell command execution",
                scopes=[SecurityScope.RUN_COMMAND, SecurityScope.SPAWN_PROCESS],
                decision=PolicyDecision.REQUIRE_APPROVAL,
                priority=50,
            ),
            # Allow specific external APIs
            PolicyRule(
                name="allow_literature_apis",
                description="Allow access to literature search APIs",
                scopes=[SecurityScope.EXTERNAL_API],
                resource_patterns=[
                    r".*pubmed\.ncbi\.nlm\.nih\.gov.*",
                    r".*arxiv\.org.*",
                    r".*api\.semanticscholar\.org.*",
                ],
                decision=PolicyDecision.ALLOW,
                priority=80,
            ),
            # Require approval for package installation
            PolicyRule(
                name="approve_package_install",
                description="Require approval for package installation",
                scopes=[SecurityScope.INSTALL_PACKAGE],
                decision=PolicyDecision.REQUIRE_APPROVAL,
                priority=50,
            ),
            # Deny sensitive operations by default
            PolicyRule(
                name="deny_sensitive",
                description="Deny access to sensitive data and secrets",
                scopes=[
                    SecurityScope.READ_SENSITIVE,
                    SecurityScope.WRITE_SENSITIVE,
                    SecurityScope.ACCESS_SECRETS,
                    SecurityScope.ADMIN_OPERATION,
                ],
                decision=PolicyDecision.DENY,
                priority=200,
            ),
        ],
    )


def create_permissive_policy() -> PolicySet:
    """Create a more permissive policy for development/testing."""
    return PolicySet(
        name="minilab_permissive",
        version="1.0.0",
        description="Permissive policy for development",
        default_decision=PolicyDecision.ALLOW,
        rules=[
            # Still deny truly sensitive operations
            PolicyRule(
                name="deny_sensitive",
                description="Deny access to secrets",
                scopes=[SecurityScope.ACCESS_SECRETS, SecurityScope.ADMIN_OPERATION],
                decision=PolicyDecision.DENY,
                priority=200,
            ),
        ],
    )
