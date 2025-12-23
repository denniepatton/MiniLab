"""
Scientific Workflow Library: Artifact-first workflow patterns.

Provides:
- Structured artifact types (figures, tables, models, reports)
- Provenance tracking
- Reproducibility metadata
- Standard output formats for scientific workflows
"""

from __future__ import annotations

__all__ = [
    "ArtifactType",
    "ArtifactStatus",
    "ProvenanceRecord",
    "Artifact",
    "ArtifactManifest",
    "ArtifactStore",
    "WorkflowStep",
    "DataLoadStep",
    "AnalysisStep",
    "ReportStep",
]

import hashlib
import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Generic

from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    """Types of scientific artifacts."""

    # Figures
    FIGURE = "figure"
    PLOT = "plot"
    DIAGRAM = "diagram"

    # Data
    TABLE = "table"
    DATAFRAME = "dataframe"
    DATASET = "dataset"

    # Models
    MODEL = "model"
    MODEL_WEIGHTS = "model_weights"
    MODEL_CONFIG = "model_config"

    # Analysis
    STATISTICS = "statistics"
    RESULTS = "results"
    METRICS = "metrics"

    # Documents
    REPORT = "report"
    SUMMARY = "summary"
    NOTEBOOK = "notebook"

    # Code
    SCRIPT = "script"
    FUNCTION = "function"
    PIPELINE = "pipeline"

    # Other
    LOG = "log"
    CONFIG = "config"
    OTHER = "other"


class ArtifactStatus(str, Enum):
    """Status of artifact creation."""
    PENDING = "pending"
    CREATING = "creating"
    COMPLETED = "completed"
    FAILED = "failed"
    STALE = "stale"


@dataclass
class ProvenanceRecord:
    """Provenance information for an artifact."""

    # Creation context
    created_at: datetime
    created_by: str  # Agent ID
    task_id: Optional[str] = None
    workflow_id: Optional[str] = None

    # Input tracking
    input_artifacts: list[str] = field(default_factory=list)  # IDs
    input_data: dict[str, str] = field(default_factory=dict)  # name -> hash

    # Parameters
    parameters: dict[str, Any] = field(default_factory=dict)

    # Environment
    python_version: Optional[str] = None
    package_versions: dict[str, str] = field(default_factory=dict)

    # Git info
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "input_artifacts": self.input_artifacts,
            "input_data": self.input_data,
            "parameters": self.parameters,
            "python_version": self.python_version,
            "package_versions": self.package_versions,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceRecord":
        """Create from dictionary."""
        return cls(
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            task_id=data.get("task_id"),
            workflow_id=data.get("workflow_id"),
            input_artifacts=data.get("input_artifacts", []),
            input_data=data.get("input_data", {}),
            parameters=data.get("parameters", {}),
            python_version=data.get("python_version"),
            package_versions=data.get("package_versions", {}),
            git_commit=data.get("git_commit"),
            git_branch=data.get("git_branch"),
        )


class Artifact(BaseModel):
    """
    A scientific artifact with provenance tracking.
    
    Artifacts are the primary outputs of scientific workflows.
    Each artifact has:
    - A unique ID
    - A type (figure, table, model, etc.)
    - Content (file path or inline data)
    - Provenance information
    - Optional metadata
    """

    id: str = Field(..., description="Unique artifact identifier")
    name: str = Field(..., description="Human-readable name")
    artifact_type: ArtifactType = Field(..., description="Type of artifact")

    # Content location
    file_path: Optional[str] = Field(default=None, description="Path to artifact file")
    content_hash: Optional[str] = Field(default=None, description="SHA256 hash of content")

    # Status
    status: ArtifactStatus = Field(default=ArtifactStatus.PENDING)

    # Metadata
    description: str = Field(default="", description="Artifact description")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Provenance (stored separately for size)
    provenance_file: Optional[str] = Field(default=None, description="Path to provenance JSON")

    model_config = {"extra": "forbid"}

    def compute_hash(self, content: bytes) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content).hexdigest()


class ArtifactManifest(BaseModel):
    """Manifest of all artifacts in a workflow."""

    workflow_id: str = Field(..., description="Workflow identifier")
    workflow_name: str = Field(..., description="Workflow name")

    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    artifacts: dict[str, Artifact] = Field(
        default_factory=dict,
        description="Mapping of artifact ID to artifact"
    )

    # Dependency graph
    dependencies: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Artifact ID -> list of input artifact IDs"
    )

    model_config = {"extra": "forbid"}

    def add_artifact(
        self,
        artifact: Artifact,
        depends_on: Optional[list[str]] = None,
    ) -> None:
        """Add an artifact to the manifest."""
        self.artifacts[artifact.id] = artifact
        if depends_on:
            self.dependencies[artifact.id] = depends_on
        self.updated_at = datetime.now().isoformat()

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Get an artifact by ID."""
        return self.artifacts.get(artifact_id)

    def get_by_type(self, artifact_type: ArtifactType) -> list[Artifact]:
        """Get all artifacts of a specific type."""
        return [
            a for a in self.artifacts.values()
            if a.artifact_type == artifact_type
        ]

    def get_lineage(self, artifact_id: str) -> list[str]:
        """Get the full lineage of an artifact (all ancestors)."""
        visited = set()
        lineage = []

        def traverse(aid: str):
            if aid in visited:
                return
            visited.add(aid)
            for dep in self.dependencies.get(aid, []):
                traverse(dep)
            lineage.append(aid)

        traverse(artifact_id)
        return lineage


class ArtifactStore:
    """
    Storage manager for artifacts.
    
    Handles:
    - File storage with content-addressable naming
    - Manifest persistence
    - Provenance tracking
    """

    def __init__(self, base_path: Path | str):
        """
        Initialize artifact store.

        Args:
            base_path: Base directory for artifact storage
        """
        self.base_path = Path(base_path) if isinstance(base_path, str) else base_path
        self.artifacts_dir = self.base_path / "artifacts"
        self.provenance_dir = self.base_path / "provenance"
        self.manifest_path = self.base_path / "artifact_manifest.json"

        # Ensure directories exist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.provenance_dir.mkdir(parents=True, exist_ok=True)

        # Load or create manifest
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> ArtifactManifest:
        """Load manifest from disk or create new one."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                data = json.load(f)
            return ArtifactManifest(**data)
        return ArtifactManifest(
            workflow_id="unknown",
            workflow_name="unknown",
        )

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        with open(self.manifest_path, "w") as f:
            json.dump(self._manifest.model_dump(), f, indent=2)

    @property
    def manifest(self) -> ArtifactManifest:
        """Get the current manifest."""
        return self._manifest

    def create_artifact(
        self,
        name: str,
        artifact_type: ArtifactType,
        content: Optional[bytes] = None,
        source_file: Optional[Path] = None,
        provenance: Optional[ProvenanceRecord] = None,
        description: str = "",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        depends_on: Optional[list[str]] = None,
    ) -> Artifact:
        """
        Create a new artifact.
        
        Args:
            name: Human-readable name
            artifact_type: Type of artifact
            content: Raw bytes content (mutually exclusive with source_file)
            source_file: Path to existing file to import
            provenance: Provenance record
            description: Description
            tags: Searchable tags
            metadata: Additional metadata
            depends_on: List of input artifact IDs
            
        Returns:
            Created Artifact
        """
        # Generate ID
        artifact_id = f"{artifact_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name.replace(' ', '_')[:20]}"

        # Determine file path and hash
        file_path = None
        content_hash = None

        if content is not None:
            content_hash = hashlib.sha256(content).hexdigest()
            # Use content-addressable storage
            ext = self._get_extension(artifact_type, metadata)
            file_name = f"{content_hash[:16]}{ext}"
            file_path = self.artifacts_dir / file_name
            file_path.write_bytes(content)
        elif source_file is not None and source_file.exists():
            content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
            # Copy file to artifacts directory
            ext = source_file.suffix
            file_name = f"{content_hash[:16]}{ext}"
            file_path = self.artifacts_dir / file_name
            shutil.copy2(source_file, file_path)

        # Save provenance
        provenance_file = None
        if provenance:
            provenance_file = self.provenance_dir / f"{artifact_id}_provenance.json"
            with open(provenance_file, "w") as f:
                json.dump(provenance.to_dict(), f, indent=2)

        # Create artifact
        artifact = Artifact(
            id=artifact_id,
            name=name,
            artifact_type=artifact_type,
            file_path=str(file_path) if file_path else None,
            content_hash=content_hash,
            status=ArtifactStatus.COMPLETED if (content or source_file) else ArtifactStatus.PENDING,
            description=description,
            tags=tags or [],
            metadata=metadata or {},
            provenance_file=str(provenance_file) if provenance_file else None,
        )

        # Add to manifest
        self._manifest.add_artifact(artifact, depends_on)
        self._save_manifest()

        return artifact

    def _get_extension(
        self,
        artifact_type: ArtifactType,
        metadata: Optional[dict]
    ) -> str:
        """Get appropriate file extension for artifact type."""
        # Check metadata for explicit format
        if metadata and "format" in metadata:
            return f".{metadata['format']}"

        # Default extensions by type
        defaults = {
            ArtifactType.FIGURE: ".png",
            ArtifactType.PLOT: ".png",
            ArtifactType.DIAGRAM: ".svg",
            ArtifactType.TABLE: ".csv",
            ArtifactType.DATAFRAME: ".parquet",
            ArtifactType.DATASET: ".parquet",
            ArtifactType.MODEL: ".pkl",
            ArtifactType.MODEL_WEIGHTS: ".pt",
            ArtifactType.MODEL_CONFIG: ".yaml",
            ArtifactType.STATISTICS: ".json",
            ArtifactType.RESULTS: ".json",
            ArtifactType.METRICS: ".json",
            ArtifactType.REPORT: ".md",
            ArtifactType.SUMMARY: ".md",
            ArtifactType.NOTEBOOK: ".ipynb",
            ArtifactType.SCRIPT: ".py",
            ArtifactType.FUNCTION: ".py",
            ArtifactType.PIPELINE: ".yaml",
            ArtifactType.LOG: ".log",
            ArtifactType.CONFIG: ".yaml",
            ArtifactType.OTHER: ".bin",
        }
        return defaults.get(artifact_type, ".bin")

    def get_artifact_content(self, artifact_id: str) -> Optional[bytes]:
        """Get the content of an artifact."""
        artifact = self._manifest.get_artifact(artifact_id)
        if artifact and artifact.file_path:
            return Path(artifact.file_path).read_bytes()
        return None

    def get_provenance(self, artifact_id: str) -> Optional[ProvenanceRecord]:
        """Get provenance record for an artifact."""
        artifact = self._manifest.get_artifact(artifact_id)
        if artifact and artifact.provenance_file:
            with open(artifact.provenance_file) as f:
                return ProvenanceRecord.from_dict(json.load(f))
        return None

    def update_status(
        self,
        artifact_id: str,
        status: ArtifactStatus
    ) -> None:
        """Update artifact status."""
        artifact = self._manifest.get_artifact(artifact_id)
        if artifact:
            artifact.status = status
            self._save_manifest()


# Workflow step patterns

T = TypeVar("T")


class WorkflowStep(ABC, Generic[T]):
    """
    Abstract base for workflow steps.
    
    Each step:
    - Takes typed inputs
    - Produces typed outputs
    - Generates artifacts
    - Tracks provenance
    """

    name: str = "step"
    description: str = ""

    @abstractmethod
    async def execute(
        self,
        inputs: dict[str, Any],
        store: ArtifactStore,
        agent_id: str,
    ) -> T:
        """Execute the step and return results."""
        ...

    def create_provenance(
        self,
        agent_id: str,
        inputs: dict[str, Any],
        task_id: Optional[str] = None,
    ) -> ProvenanceRecord:
        """Create a provenance record for this step."""
        return ProvenanceRecord(
            created_at=datetime.now(),
            created_by=agent_id,
            task_id=task_id,
            parameters={k: str(v)[:100] for k, v in inputs.items()},
        )


class DataLoadStep(WorkflowStep[dict[str, Any]]):
    """Step for loading and validating data."""

    name = "data_load"
    description = "Load and validate input data"

    async def execute(
        self,
        inputs: dict[str, Any],
        store: ArtifactStore,
        agent_id: str,
    ) -> dict[str, Any]:
        """Load data and create dataset artifact."""
        data_path = inputs.get("data_path")
        if not data_path:
            raise ValueError("data_path required")

        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        # Create artifact
        provenance = self.create_provenance(agent_id, inputs)
        artifact = store.create_artifact(
            name=f"input_{path.stem}",
            artifact_type=ArtifactType.DATASET,
            source_file=path,
            provenance=provenance,
            description=f"Input data loaded from {path.name}",
            tags=["input", "data"],
        )

        return {
            "artifact_id": artifact.id,
            "file_path": str(path),
            "file_size": path.stat().st_size,
        }


class AnalysisStep(WorkflowStep[dict[str, Any]]):
    """Step for running analysis code."""

    name = "analysis"
    description = "Run analysis and generate results"

    def __init__(self, analysis_func: Callable[..., Any]):
        """
        Initialize with analysis function.
        
        Args:
            analysis_func: Function to run for analysis
        """
        self.analysis_func = analysis_func

    async def execute(
        self,
        inputs: dict[str, Any],
        store: ArtifactStore,
        agent_id: str,
    ) -> dict[str, Any]:
        """Run analysis and store results."""
        # Run analysis
        results = self.analysis_func(**inputs)

        # Create results artifact
        provenance = self.create_provenance(agent_id, inputs)
        artifact = store.create_artifact(
            name="analysis_results",
            artifact_type=ArtifactType.RESULTS,
            content=json.dumps(results, default=str).encode(),
            provenance=provenance,
            description="Analysis results",
            tags=["analysis", "results"],
            metadata={"format": "json"},
        )

        return {
            "artifact_id": artifact.id,
            "results": results,
        }


class ReportStep(WorkflowStep[str]):
    """Step for generating reports."""

    name = "report"
    description = "Generate analysis report"

    async def execute(
        self,
        inputs: dict[str, Any],
        store: ArtifactStore,
        agent_id: str,
    ) -> str:
        """Generate report and store as artifact."""
        content = inputs.get("content", "")
        title = inputs.get("title", "Analysis Report")

        # Create report artifact
        provenance = self.create_provenance(agent_id, inputs)

        # Get input artifact IDs for lineage
        input_artifacts = inputs.get("input_artifacts", [])

        artifact = store.create_artifact(
            name=title.lower().replace(" ", "_"),
            artifact_type=ArtifactType.REPORT,
            content=content.encode("utf-8"),
            provenance=provenance,
            description=title,
            tags=["report", "output"],
            metadata={"format": "md"},
            depends_on=input_artifacts,
        )

        return artifact.id
