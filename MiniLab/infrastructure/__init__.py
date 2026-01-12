"""
MiniLab Infrastructure Layer.

Provides foundational systems for error handling, feature detection, and configuration.
This layer sits between the orchestrator and domain workflows.
"""

from .errors import (
    ErrorCategory,
    MiniLabError,
    FatalError,
    DegradedError,
    OptionalError,
    RecoverableError,
    handle_error,
    retry_with_backoff,
    validate_required_dependency,
    validate_optional_dependency,
)

from .features import (
    FeatureRegistry,
    Feature,
    get_feature_registry,
    is_feature_available,
    require_feature,
)

__all__ = [
    # Errors
    "ErrorCategory",
    "MiniLabError",
    "FatalError",
    "DegradedError",
    "OptionalError",
    "RecoverableError",
    "handle_error",
    "retry_with_backoff",
    "validate_required_dependency",
    "validate_optional_dependency",
    # Features
    "FeatureRegistry",
    "Feature",
    "get_feature_registry",
    "is_feature_available",
    "require_feature",
]
