"""
MiniLab Configuration System - SSOT (Single Source of Truth).

Consolidates all configuration from multiple YAML files into a unified system.
This is the authoritative source for:
- Project structure and paths
- Agent roster and specializations  
- Workflow definitions
- Budget allocations
- Feature flags
- Error handling policies

Loads from minilab_config.yaml (if present) with sensible defaults.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import os

from ..utils import console


@dataclass
class ProjectStructure:
    """Project directory layout configuration."""
    sandbox_root: str = "Sandbox"
    project_template: str = "{sandbox}/{project_name}"
    transcripts_dir: str = "Transcripts"
    archive_dir: str = "Archive"
    
    def resolve(self, project_name: Optional[str] = None) -> dict[str, Path]:
        """Resolve template variables to actual paths."""
        sandbox = Path(os.getenv("MINILAB_SANDBOX", self.sandbox_root))
        
        return {
            "sandbox_root": sandbox,
            "transcripts": Path(self.transcripts_dir),
            "archive": Path(self.archive_dir),
            "project": (
                sandbox / project_name
                if project_name
                else None
            ),
        }


@dataclass
class AgentConfig:
    """Configuration for an individual agent."""
    name: str
    role: str
    description: str
    specializations: List[str] = field(default_factory=list)
    can_execute: bool = True
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BudgetConfig:
    """Token budget configuration."""
    default_budget: int = 500_000
    min_budget: int = 50_000
    max_budget: int = 2_000_000
    
    # Phase allocation (learned from history if available)
    phase_allocations: Dict[str, float] = field(default_factory=lambda: {
        "discovery": 0.05,
        "planning": 0.15,
        "execution": 0.60,
        "synthesis": 0.15,
        "review": 0.05,
    })
    
    # Hard caps per phase (prevent early phases from consuming entire budget)
    phase_caps: Dict[str, int] = field(default_factory=dict)
    
    def calculate_phase_budget(self, total_budget: int, phase: str) -> int:
        """Calculate budget for a phase."""
        allocation = self.phase_allocations.get(phase, 0.1)
        budget = int(total_budget * allocation)
        
        # Apply hard cap if set
        if self.phase_caps:
            cap = self.phase_caps.get(phase)
            if cap:
                budget = min(budget, cap)
        
        return max(1000, budget)  # Minimum 1k tokens per phase


@dataclass
class WorkflowConfig:
    """Workflow execution configuration."""
    name: str
    description: str
    enabled: bool = True
    max_iterations: int = 3
    quality_threshold: float = 0.8
    timeout_seconds: Optional[int] = None


@dataclass
class FeatureConfig:
    """Feature availability and policy configuration."""
    name: str
    required: bool = False
    enabled: bool = True
    fallback_available: bool = False  # Can gracefully degrade
    

@dataclass
class ErrorHandlingPolicy:
    """Error handling rules for different scenarios."""
    bare_exceptions: str = "log_and_continue"  # log_and_continue, retry, fatal
    import_failures: str = "fatal"  # fatal, warn_and_continue
    network_errors: str = "retry"  # retry, degrade, fatal
    budget_exhaustion: str = "interactive"  # stop, degrade, interactive


class MiniLabConfig:
    """
    Unified configuration system for MiniLab.
    
    Acts as SSOT for all project settings. Loads from minilab_config.yaml
    if available, otherwise uses defaults.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent.parent / "minilab_config.yaml"
        
        # Load from file if exists, otherwise use defaults
        if self.config_path.exists():
            with open(self.config_path) as f:
                data = yaml.safe_load(f) or {}
            console.info(f"Loaded config from {self.config_path}")
        else:
            data = {}
            console.debug(f"No config file at {self.config_path}, using defaults")
        
        # Parse sections
        self.project_structure = self._parse_project_structure(data.get("project_structure", {}))
        self.agents = self._parse_agents(data.get("agents", {}))
        self.budget = self._parse_budget(data.get("budget", {}))
        self.workflows = self._parse_workflows(data.get("workflows", {}))
        self.features = self._parse_features(data.get("features", {}))
        self.error_handling = self._parse_error_handling(data.get("error_handling", {}))
        
        # Store raw data
        self._raw_config = data
    
    def _parse_project_structure(self, data: dict) -> ProjectStructure:
        """Parse project structure configuration."""
        return ProjectStructure(
            sandbox_root=data.get("sandbox_root", "Sandbox"),
            project_template=data.get("project_template", "{sandbox}/{project_name}"),
            transcripts_dir=data.get("transcripts_dir", "Transcripts"),
            archive_dir=data.get("archive_dir", "Archive"),
        )
    
    def _parse_agents(self, data: dict) -> Dict[str, AgentConfig]:
        """Parse agent configurations."""
        agents = {}
        
        # Default agents from design
        defaults = {
            "bohr": "Project Manager - high-level scientific planning, user communication, synthesis",
            "gould": "Science Writer - literature review, documentation, clear explanations",
            "farber": "Clinical Expert - experimental design, medical interpretation, protocols",
            "feynman": "Theoretician - physics, mathematics, theoretical analysis, first principles",
            "shannon": "Information Theorist - statistics, signal processing, information theory",
            "greider": "Molecular Biologist - genetics, cellular mechanisms, biological interpretation",
            "dayhoff": "Bioinformatician - sequence analysis, databases, computational biology",
            "hinton": "ML Expert - machine learning, neural networks, data analysis, modeling",
            "bayes": "Statistician - Bayesian inference, probability, uncertainty quantification",
        }
        
        # Load from config or use defaults
        for name, description in defaults.items():
            agent_data = data.get(name, {})
            agents[name] = AgentConfig(
                name=name,
                role=agent_data.get("role", name.title()),
                description=agent_data.get("description", description),
                specializations=agent_data.get("specializations", []),
                can_execute=agent_data.get("can_execute", True),
            )
        
        return agents
    
    def _parse_budget(self, data: dict) -> BudgetConfig:
        """Parse budget configuration."""
        allocations = data.get("phase_allocations", {
            "discovery": 0.05,
            "planning": 0.15,
            "execution": 0.60,
            "synthesis": 0.15,
            "review": 0.05,
        })
        
        return BudgetConfig(
            default_budget=data.get("default_budget", 500_000),
            min_budget=data.get("min_budget", 50_000),
            max_budget=data.get("max_budget", 2_000_000),
            phase_allocations=allocations,
            phase_caps=data.get("phase_caps", {}),
        )
    
    def _parse_workflows(self, data: dict) -> Dict[str, WorkflowConfig]:
        """Parse workflow configurations."""
        workflows = {}
        
        for name, cfg in data.items():
            workflows[name] = WorkflowConfig(
                name=name,
                description=cfg.get("description", ""),
                enabled=cfg.get("enabled", True),
                max_iterations=cfg.get("max_iterations", 3),
                quality_threshold=cfg.get("quality_threshold", 0.8),
                timeout_seconds=cfg.get("timeout_seconds"),
            )
        
        return workflows
    
    def _parse_features(self, data: dict) -> Dict[str, FeatureConfig]:
        """Parse feature configurations."""
        features = {}
        
        defaults = {
            "pdf_generation": {"required": True, "fallback_available": False},
            "prompt_caching": {"required": True, "fallback_available": False},
            "image_conversion": {"required": False, "fallback_available": True},
            "task_graph_visualization": {"required": False, "fallback_available": True},
            "rag_retrieval": {"required": False, "fallback_available": False},
        }
        
        for name, default_cfg in defaults.items():
            cfg = data.get(name, default_cfg)
            features[name] = FeatureConfig(
                name=name,
                required=cfg.get("required", False),
                enabled=cfg.get("enabled", True),
                fallback_available=cfg.get("fallback_available", False),
            )
        
        return features
    
    def _parse_error_handling(self, data: dict) -> ErrorHandlingPolicy:
        """Parse error handling policies."""
        return ErrorHandlingPolicy(
            bare_exceptions=data.get("bare_exceptions", "log_and_continue"),
            import_failures=data.get("import_failures", "fatal"),
            network_errors=data.get("network_errors", "retry"),
            budget_exhaustion=data.get("budget_exhaustion", "interactive"),
        )
    
    def get_agent_specializations(self, agent_name: str) -> List[str]:
        """Get specializations for an agent."""
        agent = self.agents.get(agent_name)
        return agent.specializations if agent else []
    
    def get_phase_budget(self, total_budget: int, phase: str) -> int:
        """Calculate budget for a phase."""
        return self.budget.calculate_phase_budget(total_budget, phase)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "project_structure": asdict(self.project_structure),
            "agents": {
                name: agent.to_dict()
                for name, agent in self.agents.items()
            },
            "budget": asdict(self.budget),
            "workflows": {
                name: asdict(wf)
                for name, wf in self.workflows.items()
            },
            "features": {
                name: asdict(feat)
                for name, feat in self.features.items()
            },
            "error_handling": asdict(self.error_handling),
        }


# Global singleton instance
_config: Optional[MiniLabConfig] = None


def get_config(config_path: Optional[Path] = None) -> MiniLabConfig:
    """Get or initialize the global configuration."""
    global _config
    if _config is None:
        _config = MiniLabConfig(config_path)
    return _config


def reload_config(config_path: Optional[Path] = None) -> MiniLabConfig:
    """Reload configuration."""
    global _config
    _config = MiniLabConfig(config_path)
    return _config
