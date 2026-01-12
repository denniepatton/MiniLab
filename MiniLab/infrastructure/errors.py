"""
MiniLab Error Handling Framework.

Categorizes errors for consistent handling:
- FATAL: Stop execution immediately
- DEGRADED: Continue with reduced functionality, warn user
- OPTIONAL: Silent skip, log for debugging
- RECOVERABLE: Retry with backoff

This replaces scattered try-except patterns with explicit error categorization.
"""

from enum import Enum
from typing import Optional, Callable, Any
import asyncio
from pathlib import Path

from ..utils import console


class ErrorCategory(Enum):
    """Classifies errors for handling decisions."""
    FATAL = "fatal"  # Stop execution, user action required
    DEGRADED = "degraded"  # Continue with warning
    OPTIONAL = "optional"  # Skip silently
    RECOVERABLE = "recoverable"  # Retry with backoff


class MiniLabError(Exception):
    """Base exception for MiniLab errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        self.message = message
        self.category = category
        self.context = context or {}
        self.original_error = original_error
        super().__init__(message)
    
    def __str__(self) -> str:
        parts = [f"[{self.category.value.upper()}] {self.message}"]
        if self.original_error:
            parts.append(f"Cause: {str(self.original_error)}")
        return "\n".join(parts)


class FatalError(MiniLabError):
    """Stop execution - requires user intervention or retry."""
    
    def __init__(
        self,
        message: str,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, ErrorCategory.FATAL, context, original_error)


class DegradedError(MiniLabError):
    """Continue with reduced functionality and warning."""
    
    def __init__(
        self,
        message: str,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, ErrorCategory.DEGRADED, context, original_error)


class OptionalError(MiniLabError):
    """Skip silently - feature not available."""
    
    def __init__(
        self,
        message: str,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, ErrorCategory.OPTIONAL, context, original_error)


class RecoverableError(MiniLabError):
    """Retry with exponential backoff."""
    
    def __init__(
        self,
        message: str,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        max_retries: int = 3,
    ):
        super().__init__(message, ErrorCategory.RECOVERABLE, context, original_error)
        self.max_retries = max_retries


async def handle_error(error: MiniLabError, action_name: str = "operation") -> bool:
    """
    Handle an error based on its category.
    
    Args:
        error: MiniLabError to handle
        action_name: Name of action that failed (for logging)
        
    Returns:
        True if execution should continue, False if should stop
    """
    if error.category == ErrorCategory.FATAL:
        console.error(f"FATAL ERROR in {action_name}:")
        console.error(str(error))
        return False
    
    elif error.category == ErrorCategory.DEGRADED:
        console.warning(f"DEGRADED MODE in {action_name}:")
        console.warning(str(error))
        console.warning("Continuing with reduced functionality...")
        return True
    
    elif error.category == ErrorCategory.OPTIONAL:
        console.debug(f"Optional feature unavailable ({action_name}): {error.message}")
        return True
    
    elif error.category == ErrorCategory.RECOVERABLE:
        console.warning(f"Recoverable error in {action_name}: {error.message}")
        # Caller responsible for retry logic
        return False
    
    return False


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    action_name: str = "operation",
) -> Any:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to call
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay (exponential backoff)
        action_name: Name of action (for logging)
        
    Returns:
        Result of function call
        
    Raises:
        RecoverableError: If all retries exhausted
    """
    delay = initial_delay
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                console.warning(
                    f"{action_name} failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay *= backoff_factor
            else:
                console.error(
                    f"{action_name} failed after {max_retries} attempts"
                )
    
    raise RecoverableError(
        f"{action_name} failed after {max_retries} retries",
        context={"attempts": max_retries},
        original_error=last_error,
        max_retries=max_retries,
    )


def validate_required_dependency(
    module_name: str,
    import_statement: str,
    feature_name: str,
    install_command: str,
) -> None:
    """
    Validate that a required dependency is installed.
    
    Args:
        module_name: Name of module (e.g., 'reportlab')
        import_statement: Python import statement for checking
        feature_name: Human-readable feature name
        install_command: pip install command to suggest
        
    Raises:
        FatalError: If dependency not available
    """
    try:
        __import__(module_name)
    except ImportError:
        raise FatalError(
            f"{feature_name} requires {module_name} but it's not installed.\n"
            f"Install with: {install_command}",
            context={
                "module": module_name,
                "feature": feature_name,
                "install_cmd": install_command,
            },
        )


def validate_optional_dependency(
    module_name: str,
    feature_name: str,
    install_command: str,
) -> bool:
    """
    Check if an optional dependency is available.
    
    Args:
        module_name: Name of module to check
        feature_name: Human-readable feature name
        install_command: pip install command to suggest
        
    Returns:
        True if available, False otherwise (with warning)
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        console.warning(
            f"Optional feature '{feature_name}' requires {module_name}.\n"
            f"Install with: {install_command}"
        )
        return False
