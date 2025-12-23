"""
MiniBench: Evaluation Harness for MiniLab Agents.

Provides:
- Benchmark suite definition
- Test case management
- Scoring and metrics
- Result reporting
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Protocol


class BenchmarkCategory(str, Enum):
    """Categories of benchmarks."""

    # Code generation
    CODE_GENERATION = "code_generation"
    CODE_DEBUGGING = "code_debugging"
    CODE_REFACTORING = "code_refactoring"

    # Analysis
    DATA_ANALYSIS = "data_analysis"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    VISUALIZATION = "visualization"

    # Scientific
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EXPERIMENTAL_DESIGN = "experimental_design"

    # Reasoning
    PLANNING = "planning"
    PROBLEM_SOLVING = "problem_solving"
    MULTI_STEP_REASONING = "multi_step_reasoning"

    # Tool use
    TOOL_SELECTION = "tool_selection"
    TOOL_CHAINING = "tool_chaining"

    # Other
    GENERAL = "general"


class DifficultyLevel(str, Enum):
    """Difficulty levels for test cases."""

    TRIVIAL = "trivial"       # Single step, obvious solution
    EASY = "easy"             # Few steps, straightforward
    MEDIUM = "medium"         # Multiple steps, some complexity
    HARD = "hard"             # Many steps, significant complexity
    EXPERT = "expert"         # Requires deep domain knowledge


class BenchCaseStatus(str, Enum):
    """Status of a benchmark case execution."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


# Alias for backward compatibility
TestCaseStatus = BenchCaseStatus


@dataclass
class BenchCase:
    """
    A single benchmark case for evaluation.
    
    Benchmark cases define:
    - Input prompt/task
    - Expected outputs or validation criteria
    - Metadata for categorization
    """

    id: str
    name: str
    description: str

    # Input
    prompt: str
    context: dict[str, Any] = field(default_factory=dict)

    # Expected outputs
    expected_outputs: list[str] = field(default_factory=list)
    expected_artifacts: list[str] = field(default_factory=list)

    # Validation
    validators: list[str] = field(default_factory=list)  # Validator names

    # Metadata
    category: BenchmarkCategory = BenchmarkCategory.GENERAL
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    tags: list[str] = field(default_factory=list)

    # Limits
    max_tokens: int = 50000
    max_time_seconds: int = 300
    max_tool_calls: int = 50

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "context": self.context,
            "expected_outputs": self.expected_outputs,
            "expected_artifacts": self.expected_artifacts,
            "validators": self.validators,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "tags": self.tags,
            "max_tokens": self.max_tokens,
            "max_time_seconds": self.max_time_seconds,
            "max_tool_calls": self.max_tool_calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchCase":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            prompt=data["prompt"],
            context=data.get("context", {}),
            expected_outputs=data.get("expected_outputs", []),
            expected_artifacts=data.get("expected_artifacts", []),
            validators=data.get("validators", []),
            category=BenchmarkCategory(data.get("category", "general")),
            difficulty=DifficultyLevel(data.get("difficulty", "medium")),
            tags=data.get("tags", []),
            max_tokens=data.get("max_tokens", 50000),
            max_time_seconds=data.get("max_time_seconds", 300),
            max_tool_calls=data.get("max_tool_calls", 50),
        )


# Alias for backward compatibility
TestCase = BenchCase


@dataclass
class BenchResult:
    """Result of a single benchmark case execution."""

    test_case_id: str
    status: BenchCaseStatus

    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Resource usage
    tokens_used: int = 0
    tool_calls: int = 0

    # Outputs
    output: str = ""
    artifacts_created: list[str] = field(default_factory=list)

    # Validation
    validation_results: dict[str, bool] = field(default_factory=dict)
    validation_messages: dict[str, str] = field(default_factory=dict)

    # Scores (0.0 - 1.0)
    scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0

    # Error info
    error_message: Optional[str] = None
    traceback: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_case_id": self.test_case_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "tokens_used": self.tokens_used,
            "tool_calls": self.tool_calls,
            "output": self.output[:1000],  # Truncate for storage
            "artifacts_created": self.artifacts_created,
            "validation_results": self.validation_results,
            "validation_messages": self.validation_messages,
            "scores": self.scores,
            "overall_score": self.overall_score,
            "error_message": self.error_message,
        }


# Alias for backward compatibility
TestResult = BenchResult


class Validator(Protocol):
    """Protocol for benchmark case validators."""

    name: str

    def validate(
        self,
        test_case: BenchCase,
        result: BenchResult,
        output: str,
        artifacts: list[str],
    ) -> tuple[bool, str]:
        """
        Validate test output.
        
        Returns:
            Tuple of (passed, message)
        """
        ...


class ContainsValidator:
    """Validator that checks if output contains expected strings."""

    name = "contains"

    def __init__(self, required_strings: list[str], case_sensitive: bool = False):
        self.required_strings = required_strings
        self.case_sensitive = case_sensitive

    def validate(
        self,
        test_case: BenchCase,
        result: BenchResult,
        output: str,
        artifacts: list[str],
    ) -> tuple[bool, str]:
        check_output = output if self.case_sensitive else output.lower()

        missing = []
        for req in self.required_strings:
            check_req = req if self.case_sensitive else req.lower()
            if check_req not in check_output:
                missing.append(req)

        if missing:
            return False, f"Missing required strings: {missing}"
        return True, "All required strings found"


class ArtifactValidator:
    """Validator that checks for expected artifacts."""

    name = "artifacts"

    def __init__(self, required_types: list[str]):
        self.required_types = required_types

    def validate(
        self,
        test_case: BenchCase,
        result: BenchResult,
        output: str,
        artifacts: list[str],
    ) -> tuple[bool, str]:
        artifact_types = set()
        for art in artifacts:
            # Extract type from artifact ID (e.g., "figure_xxx" -> "figure")
            parts = art.split("_")
            if parts:
                artifact_types.add(parts[0])

        missing = []
        for req in self.required_types:
            if req not in artifact_types:
                missing.append(req)

        if missing:
            return False, f"Missing artifact types: {missing}"
        return True, "All required artifacts created"


class CodeExecutionValidator:
    """Validator that executes code and checks for errors."""

    name = "code_execution"

    def validate(
        self,
        test_case: BenchCase,
        result: BenchResult,
        output: str,
        artifacts: list[str],
    ) -> tuple[bool, str]:
        # Check for common error patterns in output
        error_patterns = [
            "Error:",
            "Exception:",
            "Traceback",
            "SyntaxError",
            "NameError",
            "TypeError",
            "ValueError",
        ]

        for pattern in error_patterns:
            if pattern in output:
                return False, f"Code execution error detected: {pattern}"

        return True, "No execution errors detected"


class ScoreValidator:
    """Validator that computes a numeric score."""

    name = "score"

    def __init__(
        self,
        scoring_fn: Callable[[str, list[str]], float],
        min_score: float = 0.5,
    ):
        self.scoring_fn = scoring_fn
        self.min_score = min_score

    def validate(
        self,
        test_case: BenchCase,
        result: BenchResult,
        output: str,
        artifacts: list[str],
    ) -> tuple[bool, str]:
        score = self.scoring_fn(output, artifacts)
        result.scores["custom"] = score

        if score >= self.min_score:
            return True, f"Score {score:.2f} >= threshold {self.min_score:.2f}"
        return False, f"Score {score:.2f} < threshold {self.min_score:.2f}"


@dataclass
class BenchmarkSuite:
    """
    A collection of test cases forming a benchmark.
    
    Suites group related test cases for evaluation.
    """

    id: str
    name: str
    description: str
    version: str = "1.0.0"

    test_cases: list[BenchCase] = field(default_factory=list)

    # Metadata
    categories: list[BenchmarkCategory] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def add_test_case(self, test_case: BenchCase) -> None:
        """Add a test case to the suite."""
        self.test_cases.append(test_case)
        if test_case.category not in self.categories:
            self.categories.append(test_case.category)

    def filter_by_category(
        self,
        category: BenchmarkCategory
    ) -> list[BenchCase]:
        """Get test cases by category."""
        return [tc for tc in self.test_cases if tc.category == category]

    def filter_by_difficulty(
        self,
        difficulty: DifficultyLevel
    ) -> list[BenchCase]:
        """Get test cases by difficulty."""
        return [tc for tc in self.test_cases if tc.difficulty == difficulty]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "categories": [c.value for c in self.categories],
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkSuite":
        """Create from dictionary."""
        suite = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
        )
        for tc_data in data.get("test_cases", []):
            suite.add_test_case(TestCase.from_dict(tc_data))
        return suite

    def save(self, path: Path) -> None:
        """Save suite to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BenchmarkSuite":
        """Load suite from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class BenchmarkRun:
    """A complete benchmark run with all results."""

    suite_id: str
    run_id: str

    # Configuration
    agent_id: str
    model_name: str

    # Timing
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Results
    results: list[BenchResult] = field(default_factory=list)

    # Summary statistics
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0

    # Aggregate metrics
    total_tokens: int = 0
    total_duration_seconds: float = 0.0
    average_score: float = 0.0

    def add_result(self, result: BenchResult) -> None:
        """Add a test result and update statistics."""
        self.results.append(result)
        self.total_tests += 1

        if result.status == BenchCaseStatus.PASSED:
            self.passed += 1
        elif result.status == BenchCaseStatus.FAILED:
            self.failed += 1
        elif result.status == BenchCaseStatus.ERROR:
            self.errors += 1
        elif result.status == BenchCaseStatus.SKIPPED:
            self.skipped += 1

        self.total_tokens += result.tokens_used
        self.total_duration_seconds += result.duration_seconds

        # Update average score
        scores = [r.overall_score for r in self.results if r.overall_score > 0]
        if scores:
            self.average_score = sum(scores) / len(scores)

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suite_id": self.suite_id,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "model_name": self.model_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results": [r.to_dict() for r in self.results],
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "total_tokens": self.total_tokens,
            "total_duration_seconds": self.total_duration_seconds,
            "average_score": self.average_score,
            "pass_rate": self.pass_rate,
        }

    def save(self, path: Path) -> None:
        """Save run to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class BenchmarkRunner:
    """
    Runner for executing benchmarks.
    
    Coordinates test execution, validation, and result collection.
    """

    def __init__(
        self,
        validators: Optional[dict[str, Validator]] = None,
    ):
        """
        Initialize benchmark runner.
        
        Args:
            validators: Mapping of validator name to validator instance
        """
        self.validators = validators or {}

        # Register default validators
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default validators."""
        if "contains" not in self.validators:
            self.validators["contains"] = ContainsValidator([])
        if "code_execution" not in self.validators:
            self.validators["code_execution"] = CodeExecutionValidator()
        if "artifacts" not in self.validators:
            self.validators["artifacts"] = ArtifactValidator([])

    def register_validator(self, name: str, validator: Validator) -> None:
        """Register a custom validator."""
        self.validators[name] = validator

    async def run_test(
        self,
        test_case: BenchCase,
        execute_fn: Callable[[str, dict[str, Any]], tuple[str, list[str], int]],
    ) -> TestResult:
        """
        Run a single test case.
        
        Args:
            test_case: The test case to run
            execute_fn: Function that executes the prompt and returns
                       (output, artifacts, tokens_used)
                       
        Returns:
            TestResult with execution details
        """
        result = TestResult(
            test_case_id=test_case.id,
            status=BenchCaseStatus.RUNNING,
            start_time=datetime.now(),
        )

        try:
            # Execute with timeout
            start = time.time()
            output, artifacts, tokens = await self._execute_with_timeout(
                execute_fn,
                test_case.prompt,
                test_case.context,
                test_case.max_time_seconds,
            )
            duration = time.time() - start

            result.output = output
            result.artifacts_created = artifacts
            result.tokens_used = tokens
            result.duration_seconds = duration

            # Run validators
            all_passed = True
            for validator_name in test_case.validators:
                if validator_name in self.validators:
                    validator = self.validators[validator_name]
                    passed, message = validator.validate(
                        test_case, result, output, artifacts
                    )
                    result.validation_results[validator_name] = passed
                    result.validation_messages[validator_name] = message
                    if not passed:
                        all_passed = False

            # Calculate overall score
            if result.validation_results:
                passed_count = sum(result.validation_results.values())
                result.overall_score = passed_count / len(result.validation_results)
            else:
                result.overall_score = 1.0 if output else 0.0

            result.status = BenchCaseStatus.PASSED if all_passed else BenchCaseStatus.FAILED

        except TimeoutError:
            result.status = BenchCaseStatus.TIMEOUT
            result.error_message = f"Execution exceeded {test_case.max_time_seconds}s"

        except Exception as e:
            result.status = BenchCaseStatus.ERROR
            result.error_message = str(e)
            import traceback
            result.traceback = traceback.format_exc()

        result.end_time = datetime.now()
        return result

    async def _execute_with_timeout(
        self,
        execute_fn: Callable,
        prompt: str,
        context: dict[str, Any],
        timeout: int,
    ) -> tuple[str, list[str], int]:
        """Execute with timeout."""
        import asyncio

        try:
            return await asyncio.wait_for(
                self._async_execute(execute_fn, prompt, context),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Execution timed out after {timeout}s")

    async def _async_execute(
        self,
        execute_fn: Callable,
        prompt: str,
        context: dict[str, Any],
    ) -> tuple[str, list[str], int]:
        """Wrap sync execution in async."""
        import asyncio

        # If execute_fn is async, await it directly
        result = execute_fn(prompt, context)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def run_suite(
        self,
        suite: BenchmarkSuite,
        execute_fn: Callable[[str, dict[str, Any]], tuple[str, list[str], int]],
        agent_id: str,
        model_name: str,
        filter_categories: Optional[list[BenchmarkCategory]] = None,
        filter_difficulty: Optional[DifficultyLevel] = None,
    ) -> BenchmarkRun:
        """
        Run a complete benchmark suite.
        
        Args:
            suite: The benchmark suite to run
            execute_fn: Function that executes prompts
            agent_id: ID of the agent being tested
            model_name: Name of the model
            filter_categories: Only run tests in these categories
            filter_difficulty: Only run tests at this difficulty
            
        Returns:
            BenchmarkRun with all results
        """
        run = BenchmarkRun(
            suite_id=suite.id,
            run_id=f"{suite.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            agent_id=agent_id,
            model_name=model_name,
            started_at=datetime.now(),
        )

        # Filter test cases
        test_cases = suite.test_cases
        if filter_categories:
            test_cases = [tc for tc in test_cases if tc.category in filter_categories]
        if filter_difficulty:
            test_cases = [tc for tc in test_cases if tc.difficulty == filter_difficulty]

        # Run each test
        for test_case in test_cases:
            result = await self.run_test(test_case, execute_fn)
            run.add_result(result)

        run.completed_at = datetime.now()
        return run


# Utility for creating standard test suites

def create_code_benchmark() -> BenchmarkSuite:
    """Create a benchmark suite for code generation capabilities."""
    suite = BenchmarkSuite(
        id="code_generation_v1",
        name="Code Generation Benchmark",
        description="Tests for Python code generation capabilities",
    )

    suite.add_test_case(TestCase(
        id="code_hello_world",
        name="Hello World",
        description="Generate a simple hello world program",
        prompt="Write a Python function that prints 'Hello, World!'",
        expected_outputs=["def", "print", "Hello"],
        validators=["contains"],
        category=BenchmarkCategory.CODE_GENERATION,
        difficulty=DifficultyLevel.TRIVIAL,
    ))

    suite.add_test_case(TestCase(
        id="code_fibonacci",
        name="Fibonacci",
        description="Generate Fibonacci sequence function",
        prompt="Write a Python function to compute the nth Fibonacci number",
        expected_outputs=["def", "fibonacci", "return"],
        validators=["contains", "code_execution"],
        category=BenchmarkCategory.CODE_GENERATION,
        difficulty=DifficultyLevel.EASY,
    ))

    suite.add_test_case(TestCase(
        id="code_sort",
        name="Sorting Algorithm",
        description="Implement a sorting algorithm",
        prompt="Write a Python implementation of quicksort",
        expected_outputs=["def", "pivot", "return"],
        validators=["contains", "code_execution"],
        category=BenchmarkCategory.CODE_GENERATION,
        difficulty=DifficultyLevel.MEDIUM,
    ))

    return suite


def create_analysis_benchmark() -> BenchmarkSuite:
    """Create a benchmark suite for data analysis capabilities."""
    suite = BenchmarkSuite(
        id="data_analysis_v1",
        name="Data Analysis Benchmark",
        description="Tests for data analysis capabilities",
    )

    suite.add_test_case(TestCase(
        id="analysis_descriptive",
        name="Descriptive Statistics",
        description="Compute basic descriptive statistics",
        prompt="Given a dataset, compute mean, median, and standard deviation",
        expected_outputs=["mean", "median", "std"],
        validators=["contains"],
        category=BenchmarkCategory.DATA_ANALYSIS,
        difficulty=DifficultyLevel.EASY,
    ))

    suite.add_test_case(TestCase(
        id="analysis_correlation",
        name="Correlation Analysis",
        description="Compute correlation matrix",
        prompt="Analyze correlations between variables in the dataset",
        expected_outputs=["correlation", "pearson"],
        validators=["contains"],
        category=BenchmarkCategory.STATISTICAL_ANALYSIS,
        difficulty=DifficultyLevel.MEDIUM,
    ))

    return suite
