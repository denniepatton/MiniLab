"""
Prepared Tool Invocation - VS Code-style two-phase tool execution.

Implements the preparation phase pattern from VS Code's LanguageModelTool:
1. prepare() - Validate, estimate cost, request confirmation if needed
2. invoke() - Actually execute the operation

This enables:
- Pre-flight validation before expensive operations
- User confirmation for destructive actions
- Cost estimation for budget awareness
- Graceful cancellation before side effects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ConfirmationLevel(Enum):
    """Level of confirmation required for an operation."""
    NONE = "none"                    # No confirmation needed
    PASSIVE = "passive"              # Show what will happen, auto-proceed
    ACTIVE = "active"                # Require explicit user confirmation
    DESTRUCTIVE = "destructive"      # Warn about irreversible action


@dataclass
class ConfirmationMessage:
    """
    Message to display when requesting user confirmation.
    
    Follows VS Code's confirmationMessages pattern.
    """
    title: str
    message: str
    level: ConfirmationLevel = ConfirmationLevel.ACTIVE
    details: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "message": self.message,
            "level": self.level.value,
            "details": self.details,
        }


@dataclass
class PreparedInvocation:
    """
    Result of preparing a tool invocation.
    
    Contains validation results, confirmation requirements, and cost estimates.
    The agent/orchestrator can use this to:
    - Abort early if validation fails
    - Request user confirmation if needed
    - Track estimated vs actual costs
    - Preview what will happen
    
    Mirrors VS Code's IPreparedToolInvocation.
    """
    # Validation
    is_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)
    
    # Confirmation
    requires_confirmation: bool = False
    confirmation: Optional[ConfirmationMessage] = None
    
    # Cost estimation (for budget awareness)
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_duration_ms: int = 0
    
    # Preview of what will happen
    preview: Optional[str] = None
    affected_paths: list[str] = field(default_factory=list)
    
    # Tool-specific metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # If prepare determines the result without execution
    # (e.g., cache hit, no-op detection)
    has_result: bool = False
    cached_result: Optional[Any] = None
    
    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "requires_confirmation": self.requires_confirmation,
            "confirmation": self.confirmation.to_dict() if self.confirmation else None,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_duration_ms": self.estimated_duration_ms,
            "preview": self.preview,
            "affected_paths": self.affected_paths,
            "metadata": self.metadata,
            "has_result": self.has_result,
        }
    
    @classmethod
    def valid(cls, **kwargs) -> PreparedInvocation:
        """Create a valid preparation with no confirmation needed."""
        return cls(is_valid=True, **kwargs)
    
    @classmethod
    def invalid(cls, errors: list[str]) -> PreparedInvocation:
        """Create an invalid preparation with validation errors."""
        return cls(is_valid=False, validation_errors=errors)
    
    @classmethod
    def needs_confirmation(
        cls,
        title: str,
        message: str,
        level: ConfirmationLevel = ConfirmationLevel.ACTIVE,
        details: Optional[str] = None,
        **kwargs,
    ) -> PreparedInvocation:
        """Create a preparation requiring user confirmation."""
        return cls(
            is_valid=True,
            requires_confirmation=True,
            confirmation=ConfirmationMessage(
                title=title,
                message=message,
                level=level,
                details=details,
            ),
            **kwargs,
        )
    
    @classmethod
    def cached(cls, result: Any) -> PreparedInvocation:
        """Create a preparation with a cached result (no execution needed)."""
        return cls(
            is_valid=True,
            has_result=True,
            cached_result=result,
        )


# Actions that typically require confirmation
DESTRUCTIVE_ACTIONS = {
    "filesystem.delete",
    "filesystem.move",
    "code_editor.delete_lines",
    "terminal.execute",  # Some commands may be destructive
}

# Actions that modify state but are usually safe
MODIFYING_ACTIONS = {
    "filesystem.write",
    "filesystem.append",
    "code_editor.create",
    "code_editor.edit",
    "code_editor.insert",
    "code_editor.replace",
    "code_editor.replace_text",
    "environment.install",
}

# Read-only actions that never need confirmation
READONLY_ACTIONS = {
    "filesystem.read",
    "filesystem.head",
    "filesystem.list",
    "filesystem.stats",
    "filesystem.search",
    "code_editor.view",
    "code_editor.check_syntax",
    "web_search.search",
    "pubmed.search",
    "arxiv.search",
    "environment.check",
}


def get_default_confirmation_level(tool_name: str, action: str) -> ConfirmationLevel:
    """
    Get the default confirmation level for a tool action.
    
    Args:
        tool_name: Name of the tool
        action: Action being performed
        
    Returns:
        Appropriate ConfirmationLevel
    """
    key = f"{tool_name}.{action}"
    
    if key in DESTRUCTIVE_ACTIONS:
        return ConfirmationLevel.DESTRUCTIVE
    elif key in MODIFYING_ACTIONS:
        return ConfirmationLevel.PASSIVE
    elif key in READONLY_ACTIONS:
        return ConfirmationLevel.NONE
    else:
        # Default to passive for unknown actions
        return ConfirmationLevel.PASSIVE
