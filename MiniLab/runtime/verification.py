"""
Verification: Output validation and quality checks.

Provides:
- VerificationReport: Results of verification checks
- Verifier: Configurable verification engine
- Schema validation, syntax checks, and custom rules
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import json

from pydantic import BaseModel, Field


class CheckResult(str, Enum):
    """Result of a verification check."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class VerificationCheck(BaseModel):
    """A single verification check and its result."""

    name: str = Field(..., description="Check name")
    description: str = Field(default="", description="What the check verifies")
    result: CheckResult = Field(..., description="Check result")
    message: str = Field(default="", description="Result message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")

    model_config = {"extra": "forbid"}


class VerificationReport(BaseModel):
    """
    Complete verification report for a task or artifact.
    
    Aggregates multiple checks with overall pass/fail status.
    """

    target_id: str = Field(..., description="ID of what was verified (task/artifact)")
    target_type: str = Field(default="task", description="Type of target")

    checks: list[VerificationCheck] = Field(default_factory=list, description="All checks run")

    overall_result: CheckResult = Field(
        default=CheckResult.PASSED,
        description="Overall result"
    )

    verified_at: datetime = Field(default_factory=datetime.now, description="When verified")
    verifier_name: str = Field(default="default", description="Which verifier was used")

    model_config = {"extra": "forbid"}

    def add_check(self, check: VerificationCheck) -> None:
        """Add a check result."""
        self.checks.append(check)
        self._update_overall_result()

    def _update_overall_result(self) -> None:
        """Update overall result based on checks."""
        if any(c.result == CheckResult.FAILED for c in self.checks):
            self.overall_result = CheckResult.FAILED
        elif any(c.result == CheckResult.WARNING for c in self.checks):
            self.overall_result = CheckResult.WARNING
        elif all(c.result == CheckResult.SKIPPED for c in self.checks):
            self.overall_result = CheckResult.SKIPPED
        else:
            self.overall_result = CheckResult.PASSED

    @property
    def passed(self) -> bool:
        """Check if verification passed."""
        return self.overall_result in (CheckResult.PASSED, CheckResult.WARNING)

    @property
    def failed_checks(self) -> list[VerificationCheck]:
        """Get all failed checks."""
        return [c for c in self.checks if c.result == CheckResult.FAILED]

    def summary(self) -> dict[str, Any]:
        """Get summary of verification."""
        return {
            "target_id": self.target_id,
            "overall_result": self.overall_result.value,
            "total_checks": len(self.checks),
            "passed": sum(1 for c in self.checks if c.result == CheckResult.PASSED),
            "failed": sum(1 for c in self.checks if c.result == CheckResult.FAILED),
            "warnings": sum(1 for c in self.checks if c.result == CheckResult.WARNING),
        }


class CheckSpec(BaseModel):
    """Specification for a verification check."""

    name: str = Field(..., description="Check name")
    check_type: str = Field(..., description="Type of check")
    params: dict[str, Any] = Field(default_factory=dict, description="Check parameters")
    required: bool = Field(default=True, description="Whether check is required")

    model_config = {"extra": "forbid"}


class Verifier(ABC):
    """
    Base class for verification engines.
    
    Subclasses implement specific verification logic.
    """

    name: str = "base"

    @abstractmethod
    def verify(
        self,
        target_id: str,
        target: Any,
        checks: Optional[list[CheckSpec]] = None
    ) -> VerificationReport:
        """
        Run verification on a target.
        
        Args:
            target_id: Identifier for the target
            target: The object/artifact to verify
            checks: Optional list of specific checks to run
            
        Returns:
            VerificationReport with results
        """
        pass


class SchemaVerifier(Verifier):
    """Verifier that checks output against JSON schema."""

    name: str = "schema"

    def __init__(self, schema: dict[str, Any]):
        """
        Initialize with expected schema.
        
        Args:
            schema: JSON schema to validate against
        """
        self.schema = schema

    def verify(
        self,
        target_id: str,
        target: Any,
        checks: Optional[list[CheckSpec]] = None
    ) -> VerificationReport:
        """Verify target against schema."""
        report = VerificationReport(
            target_id=target_id,
            target_type="schema",
            verifier_name=self.name,
        )

        # Basic type check
        if isinstance(target, dict):
            report.add_check(VerificationCheck(
                name="type_check",
                description="Check target is a dict",
                result=CheckResult.PASSED,
            ))
        else:
            report.add_check(VerificationCheck(
                name="type_check",
                description="Check target is a dict",
                result=CheckResult.FAILED,
                message=f"Expected dict, got {type(target).__name__}",
            ))
            return report

        # Check required fields
        required = self.schema.get("required", [])
        for field in required:
            if field in target:
                report.add_check(VerificationCheck(
                    name=f"required_{field}",
                    description=f"Check required field '{field}' exists",
                    result=CheckResult.PASSED,
                ))
            else:
                report.add_check(VerificationCheck(
                    name=f"required_{field}",
                    description=f"Check required field '{field}' exists",
                    result=CheckResult.FAILED,
                    message=f"Missing required field: {field}",
                ))

        return report


class FileVerifier(Verifier):
    """Verifier for file artifacts."""

    name: str = "file"

    def verify(
        self,
        target_id: str,
        target: Any,
        checks: Optional[list[CheckSpec]] = None
    ) -> VerificationReport:
        """
        Verify a file artifact.
        
        Args:
            target_id: File identifier
            target: Path to the file
            checks: Optional checks (e.g., size, extension)
        """
        report = VerificationReport(
            target_id=target_id,
            target_type="file",
            verifier_name=self.name,
        )

        path = Path(target) if isinstance(target, str) else target

        # Existence check
        if path.exists():
            report.add_check(VerificationCheck(
                name="exists",
                description="Check file exists",
                result=CheckResult.PASSED,
            ))
        else:
            report.add_check(VerificationCheck(
                name="exists",
                description="Check file exists",
                result=CheckResult.FAILED,
                message=f"File not found: {path}",
            ))
            return report

        # Size check (not empty)
        size = path.stat().st_size
        if size > 0:
            report.add_check(VerificationCheck(
                name="not_empty",
                description="Check file is not empty",
                result=CheckResult.PASSED,
                details={"size_bytes": size},
            ))
        else:
            report.add_check(VerificationCheck(
                name="not_empty",
                description="Check file is not empty",
                result=CheckResult.WARNING,
                message="File is empty",
            ))

        # Run custom checks if provided
        if checks:
            for spec in checks:
                result = self._run_check(path, spec)
                report.add_check(result)

        return report

    def _run_check(self, path: Path, spec: CheckSpec) -> VerificationCheck:
        """Run a specific check."""
        if spec.check_type == "extension":
            expected = spec.params.get("extension", "")
            actual = path.suffix
            if actual == expected:
                return VerificationCheck(
                    name=spec.name,
                    description=f"Check extension is {expected}",
                    result=CheckResult.PASSED,
                )
            else:
                return VerificationCheck(
                    name=spec.name,
                    description=f"Check extension is {expected}",
                    result=CheckResult.FAILED,
                    message=f"Expected {expected}, got {actual}",
                )

        elif spec.check_type == "min_size":
            min_bytes = spec.params.get("bytes", 0)
            actual = path.stat().st_size
            if actual >= min_bytes:
                return VerificationCheck(
                    name=spec.name,
                    description=f"Check size >= {min_bytes}",
                    result=CheckResult.PASSED,
                )
            else:
                return VerificationCheck(
                    name=spec.name,
                    description=f"Check size >= {min_bytes}",
                    result=CheckResult.FAILED,
                    message=f"Size {actual} < {min_bytes}",
                )

        return VerificationCheck(
            name=spec.name,
            description=f"Unknown check type: {spec.check_type}",
            result=CheckResult.SKIPPED,
        )


class CodeVerifier(Verifier):
    """Verifier for code artifacts (syntax, imports)."""

    name: str = "code"

    def verify(
        self,
        target_id: str,
        target: Any,
        checks: Optional[list[CheckSpec]] = None
    ) -> VerificationReport:
        """
        Verify code artifact.
        
        Args:
            target_id: Code identifier
            target: Code string or path
        """
        report = VerificationReport(
            target_id=target_id,
            target_type="code",
            verifier_name=self.name,
        )

        # Get code content
        if isinstance(target, Path) or (isinstance(target, str) and Path(target).exists()):
            path = Path(target)
            code = path.read_text()
            suffix = path.suffix
        else:
            code = str(target)
            suffix = ".py"  # Assume Python

        # Python syntax check
        if suffix == ".py":
            try:
                compile(code, "<string>", "exec")
                report.add_check(VerificationCheck(
                    name="python_syntax",
                    description="Check Python syntax is valid",
                    result=CheckResult.PASSED,
                ))
            except SyntaxError as e:
                report.add_check(VerificationCheck(
                    name="python_syntax",
                    description="Check Python syntax is valid",
                    result=CheckResult.FAILED,
                    message=str(e),
                    details={"line": e.lineno, "offset": e.offset},
                ))

        # JSON syntax check
        elif suffix == ".json":
            try:
                json.loads(code)
                report.add_check(VerificationCheck(
                    name="json_syntax",
                    description="Check JSON syntax is valid",
                    result=CheckResult.PASSED,
                ))
            except json.JSONDecodeError as e:
                report.add_check(VerificationCheck(
                    name="json_syntax",
                    description="Check JSON syntax is valid",
                    result=CheckResult.FAILED,
                    message=str(e),
                ))

        return report


class CompositeVerifier(Verifier):
    """Verifier that combines multiple verifiers."""

    name: str = "composite"

    def __init__(self, verifiers: list[Verifier]):
        """
        Initialize with list of verifiers.
        
        Args:
            verifiers: Verifiers to run in sequence
        """
        self.verifiers = verifiers

    def verify(
        self,
        target_id: str,
        target: Any,
        checks: Optional[list[CheckSpec]] = None
    ) -> VerificationReport:
        """Run all verifiers and combine results."""
        report = VerificationReport(
            target_id=target_id,
            target_type="composite",
            verifier_name=self.name,
        )

        for verifier in self.verifiers:
            sub_report = verifier.verify(target_id, target, checks)
            for check in sub_report.checks:
                report.add_check(check)

        return report
