"""
Scientific Analysis Patterns: Reusable workflow templates.

Provides:
- Standard analysis patterns (survival, classification, regression)
- Validation workflows
- Report generation templates
"""

from __future__ import annotations

__all__ = [
    "AnalysisType",
    "ValidationStrategy",
    "AnalysisConfig",
    "AnalysisResult",
    "AnalysisPattern",
    "DescriptivePattern",
    "SurvivalPattern",
    "ClassificationPattern",
    "ReportTemplate",
    "StandardReportTemplate",
    "ANALYSIS_PATTERNS",
    "get_pattern",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol


from .artifacts import (
    ArtifactStore,
    ArtifactType,
    ProvenanceRecord,
)


class AnalysisType(str, Enum):
    """Types of scientific analyses."""

    # Statistical
    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    CORRELATION = "correlation"
    REGRESSION = "regression"

    # Survival
    SURVIVAL_ANALYSIS = "survival_analysis"
    KAPLAN_MEIER = "kaplan_meier"
    COX_REGRESSION = "cox_regression"

    # Classification
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"

    # Clustering
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"

    # Other
    TIME_SERIES = "time_series"
    NETWORK_ANALYSIS = "network_analysis"
    CUSTOM = "custom"


class ValidationStrategy(str, Enum):
    """Validation strategies for analysis."""

    HOLDOUT = "holdout"
    KFOLD = "k_fold"
    LOO = "leave_one_out"
    BOOTSTRAP = "bootstrap"
    TIME_SERIES_CV = "time_series_cv"
    STRATIFIED_KFOLD = "stratified_k_fold"


@dataclass
class AnalysisConfig:
    """Configuration for an analysis."""

    analysis_type: AnalysisType
    name: str
    description: str = ""

    # Data configuration
    target_column: Optional[str] = None
    feature_columns: list[str] = field(default_factory=list)
    group_column: Optional[str] = None

    # Validation
    validation_strategy: ValidationStrategy = ValidationStrategy.KFOLD
    n_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42

    # Model configuration
    model_params: dict[str, Any] = field(default_factory=dict)

    # Output configuration
    generate_figures: bool = True
    generate_tables: bool = True
    generate_report: bool = True


@dataclass
class AnalysisResult:
    """Result from an analysis pattern."""

    success: bool
    analysis_type: AnalysisType

    # Core results
    metrics: dict[str, float] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)

    # Artifacts generated
    artifact_ids: list[str] = field(default_factory=list)
    figure_ids: list[str] = field(default_factory=list)
    table_ids: list[str] = field(default_factory=list)

    # Model info (if applicable)
    model_artifact_id: Optional[str] = None
    feature_importance: Optional[dict[str, float]] = None

    # Validation results
    validation_scores: dict[str, list[float]] = field(default_factory=dict)

    # Error info
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "analysis_type": self.analysis_type.value,
            "metrics": self.metrics,
            "statistics": self.statistics,
            "artifact_ids": self.artifact_ids,
            "figure_ids": self.figure_ids,
            "table_ids": self.table_ids,
            "model_artifact_id": self.model_artifact_id,
            "feature_importance": self.feature_importance,
            "validation_scores": self.validation_scores,
            "error_message": self.error_message,
        }


class AnalysisPattern(ABC):
    """
    Abstract base for analysis patterns.
    
    Each pattern encapsulates a complete analysis workflow:
    - Data preparation
    - Model fitting/analysis
    - Validation
    - Result generation
    - Artifact creation
    """

    analysis_type: AnalysisType
    name: str = "base_pattern"
    description: str = ""

    def __init__(self, config: AnalysisConfig, store: ArtifactStore):
        """
        Initialize analysis pattern.
        
        Args:
            config: Analysis configuration
            store: Artifact store for outputs
        """
        self.config = config
        self.store = store

    @abstractmethod
    async def prepare_data(
        self,
        data: Any,
        agent_id: str
    ) -> tuple[Any, Any]:
        """
        Prepare data for analysis.
        
        Returns:
            Tuple of (features, target)
        """
        ...

    @abstractmethod
    async def fit(
        self,
        features: Any,
        target: Any,
        agent_id: str,
    ) -> Any:
        """
        Fit model/run analysis.
        
        Returns:
            Fitted model or analysis results
        """
        ...

    @abstractmethod
    async def validate(
        self,
        model: Any,
        features: Any,
        target: Any,
        agent_id: str,
    ) -> dict[str, Any]:
        """
        Validate results.
        
        Returns:
            Validation metrics
        """
        ...

    @abstractmethod
    async def generate_outputs(
        self,
        model: Any,
        results: dict[str, Any],
        agent_id: str,
    ) -> list[str]:
        """
        Generate output artifacts.
        
        Returns:
            List of artifact IDs
        """
        ...

    async def run(
        self,
        data: Any,
        agent_id: str,
    ) -> AnalysisResult:
        """
        Run the complete analysis pattern.
        
        Args:
            data: Input data
            agent_id: ID of executing agent
            
        Returns:
            AnalysisResult with all outputs
        """
        try:
            # Prepare
            features, target = await self.prepare_data(data, agent_id)

            # Fit
            model = await self.fit(features, target, agent_id)

            # Validate
            validation = await self.validate(model, features, target, agent_id)

            # Generate outputs
            artifact_ids = await self.generate_outputs(model, validation, agent_id)

            return AnalysisResult(
                success=True,
                analysis_type=self.analysis_type,
                metrics=validation.get("metrics", {}),
                statistics=validation.get("statistics", {}),
                artifact_ids=artifact_ids,
                validation_scores=validation.get("scores", {}),
            )

        except Exception as e:
            return AnalysisResult(
                success=False,
                analysis_type=self.analysis_type,
                error_message=str(e),
            )


class DescriptivePattern(AnalysisPattern):
    """Pattern for descriptive statistics analysis."""

    analysis_type = AnalysisType.DESCRIPTIVE
    name = "descriptive_statistics"
    description = "Compute descriptive statistics for dataset"

    async def prepare_data(
        self,
        data: Any,
        agent_id: str
    ) -> tuple[Any, Any]:
        """Prepare data - return as-is for descriptive."""
        return data, None

    async def fit(
        self,
        features: Any,
        target: Any,
        agent_id: str,
    ) -> dict[str, Any]:
        """Compute descriptive statistics."""
        # This would integrate with pandas/numpy in real implementation
        return {
            "computed": True,
            "timestamp": datetime.now().isoformat(),
        }

    async def validate(
        self,
        model: Any,
        features: Any,
        target: Any,
        agent_id: str,
    ) -> dict[str, Any]:
        """No validation for descriptive - return summary."""
        return {
            "metrics": {},
            "statistics": model,
        }

    async def generate_outputs(
        self,
        model: Any,
        results: dict[str, Any],
        agent_id: str,
    ) -> list[str]:
        """Generate statistics artifact."""
        import json

        provenance = ProvenanceRecord(
            created_at=datetime.now(),
            created_by=agent_id,
            parameters={"config": self.config.name},
        )

        artifact = self.store.create_artifact(
            name="descriptive_statistics",
            artifact_type=ArtifactType.STATISTICS,
            content=json.dumps(results, default=str).encode(),
            provenance=provenance,
            description="Descriptive statistics summary",
            tags=["statistics", "descriptive"],
            metadata={"format": "json"},
        )

        return [artifact.id]


class SurvivalPattern(AnalysisPattern):
    """Pattern for survival analysis."""

    analysis_type = AnalysisType.SURVIVAL_ANALYSIS
    name = "survival_analysis"
    description = "Run survival analysis with Kaplan-Meier and Cox regression"

    async def prepare_data(
        self,
        data: Any,
        agent_id: str
    ) -> tuple[Any, Any]:
        """Prepare survival data."""
        # Extract time and event columns
        return data, None  # In real impl, extract T and E

    async def fit(
        self,
        features: Any,
        target: Any,
        agent_id: str,
    ) -> dict[str, Any]:
        """Fit survival models."""
        return {
            "kaplan_meier": {"median_survival": None},
            "cox_model": {"coefficients": {}},
            "computed": True,
        }

    async def validate(
        self,
        model: Any,
        features: Any,
        target: Any,
        agent_id: str,
    ) -> dict[str, Any]:
        """Validate survival model."""
        return {
            "metrics": {
                "concordance_index": 0.0,
                "log_likelihood": 0.0,
            },
            "statistics": model,
        }

    async def generate_outputs(
        self,
        model: Any,
        results: dict[str, Any],
        agent_id: str,
    ) -> list[str]:
        """Generate survival analysis outputs."""
        import json

        artifact_ids = []

        # Statistics artifact
        provenance = ProvenanceRecord(
            created_at=datetime.now(),
            created_by=agent_id,
            parameters={"analysis": "survival"},
        )

        stats_artifact = self.store.create_artifact(
            name="survival_statistics",
            artifact_type=ArtifactType.STATISTICS,
            content=json.dumps(results, default=str).encode(),
            provenance=provenance,
            description="Survival analysis results",
            tags=["survival", "statistics"],
            metadata={"format": "json"},
        )
        artifact_ids.append(stats_artifact.id)

        return artifact_ids


class ClassificationPattern(AnalysisPattern):
    """Pattern for classification analysis."""

    analysis_type = AnalysisType.BINARY_CLASSIFICATION
    name = "classification"
    description = "Run classification analysis with validation"

    async def prepare_data(
        self,
        data: Any,
        agent_id: str
    ) -> tuple[Any, Any]:
        """Prepare classification data."""
        return data, None  # In real impl, split X and y

    async def fit(
        self,
        features: Any,
        target: Any,
        agent_id: str,
    ) -> dict[str, Any]:
        """Fit classification model."""
        return {
            "model_type": "classifier",
            "trained": True,
        }

    async def validate(
        self,
        model: Any,
        features: Any,
        target: Any,
        agent_id: str,
    ) -> dict[str, Any]:
        """Validate classification model."""
        return {
            "metrics": {
                "accuracy": 0.0,
                "auc": 0.0,
                "f1": 0.0,
            },
            "scores": {
                "cv_accuracy": [],
                "cv_auc": [],
            },
        }

    async def generate_outputs(
        self,
        model: Any,
        results: dict[str, Any],
        agent_id: str,
    ) -> list[str]:
        """Generate classification outputs."""
        import json

        provenance = ProvenanceRecord(
            created_at=datetime.now(),
            created_by=agent_id,
            parameters={"analysis": "classification"},
        )

        artifact = self.store.create_artifact(
            name="classification_results",
            artifact_type=ArtifactType.RESULTS,
            content=json.dumps(results, default=str).encode(),
            provenance=provenance,
            description="Classification analysis results",
            tags=["classification", "results"],
            metadata={"format": "json"},
        )

        return [artifact.id]


# Report templates

class ReportTemplate(Protocol):
    """Protocol for report templates."""

    def render(
        self,
        results: AnalysisResult,
        config: AnalysisConfig,
        metadata: dict[str, Any],
    ) -> str:
        """Render the report as markdown."""
        ...


class StandardReportTemplate:
    """Standard scientific report template."""

    def render(
        self,
        results: AnalysisResult,
        config: AnalysisConfig,
        metadata: dict[str, Any],
    ) -> str:
        """Render standard report."""
        sections = []

        # Header
        sections.append(f"# {config.name}")
        sections.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

        # Description
        if config.description:
            sections.append(f"## Overview\n\n{config.description}\n")

        # Methods
        sections.append("## Methods\n")
        sections.append(f"- Analysis type: {config.analysis_type.value}")
        sections.append(f"- Validation: {config.validation_strategy.value}")
        if config.n_folds > 0:
            sections.append(f"- Folds: {config.n_folds}")
        sections.append("")

        # Results
        sections.append("## Results\n")

        if results.metrics:
            sections.append("### Metrics\n")
            for metric, value in results.metrics.items():
                if isinstance(value, float):
                    sections.append(f"- {metric}: {value:.4f}")
                else:
                    sections.append(f"- {metric}: {value}")
            sections.append("")

        if results.validation_scores:
            sections.append("### Cross-Validation Scores\n")
            for score_name, scores in results.validation_scores.items():
                if scores:
                    mean = sum(scores) / len(scores)
                    sections.append(f"- {score_name}: {mean:.4f} (±{max(scores) - min(scores):.4f})")
            sections.append("")

        # Artifacts
        if results.artifact_ids:
            sections.append("## Generated Artifacts\n")
            for aid in results.artifact_ids:
                sections.append(f"- `{aid}`")
            sections.append("")

        return "\n".join(sections)


# Pattern registry

ANALYSIS_PATTERNS: dict[AnalysisType, type[AnalysisPattern]] = {
    AnalysisType.DESCRIPTIVE: DescriptivePattern,
    AnalysisType.SURVIVAL_ANALYSIS: SurvivalPattern,
    AnalysisType.BINARY_CLASSIFICATION: ClassificationPattern,
}


def get_pattern(
    analysis_type: AnalysisType,
    config: AnalysisConfig,
    store: ArtifactStore,
) -> AnalysisPattern:
    """Get an analysis pattern by type."""
    pattern_class = ANALYSIS_PATTERNS.get(analysis_type)
    if pattern_class is None:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    return pattern_class(config, store)
