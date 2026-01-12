"""
MiniLab Feature Registry.

Central detection and tracking of available features.
Replaces scattered try-except import guards throughout codebase.

Features are checked at startup and results cached.
"""

from dataclasses import dataclass
from typing import Dict
from ..utils import console
from .errors import validate_optional_dependency


@dataclass
class Feature:
    """A single feature with availability status."""
    name: str
    available: bool
    required: bool = False
    module: str = ""
    install_cmd: str = ""


class FeatureRegistry:
    """
    Central registry for MiniLab features.
    
    Detects available features at startup and provides consistent access.
    Replaces scattered `try: import X except: pass` patterns.
    """
    
    def __init__(self):
        self._features: Dict[str, Feature] = {}
        self._initialized = False
    
    def initialize(self) -> None:
        """Detect available features at startup."""
        if self._initialized:
            return
        
        # PDF generation
        pdf_available = validate_optional_dependency(
            "reportlab",
            "PDF generation",
            "pip install reportlab>=4.0"
        )
        self._features["pdf_generation"] = Feature(
            name="pdf_generation",
            available=pdf_available,
            required=True,  # Nature outputs mandate this
            module="reportlab",
            install_cmd="pip install reportlab>=4.0",
        )
        
        # Image handling
        image_available = validate_optional_dependency(
            "pdf2image",
            "Image conversion",
            "pip install pdf2image"
        )
        self._features["image_conversion"] = Feature(
            name="image_conversion",
            available=image_available,
            required=False,
            module="pdf2image",
            install_cmd="pip install pdf2image",
        )
        
        # Graphviz (for task graph visualization)
        graphviz_available = validate_optional_dependency(
            "graphviz",
            "Task graph visualization",
            "pip install graphviz"
        )
        self._features["task_graph_visualization"] = Feature(
            name="task_graph_visualization",
            available=graphviz_available,
            required=False,
            module="graphviz",
            install_cmd="pip install graphviz",
        )
        
        # Prompt caching (should always be available with Anthropic SDK)
        try:
            import anthropic
            cache_available = hasattr(anthropic.Anthropic, "beta") or True  # Always available with SDK
        except ImportError:
            cache_available = False
        
        self._features["prompt_caching"] = Feature(
            name="prompt_caching",
            available=cache_available,
            required=True,  # Core efficiency feature
            module="anthropic",
            install_cmd="pip install anthropic>=0.20.0",
        )
        
        # Response caching (SQLite-based, should always work)
        self._features["response_caching"] = Feature(
            name="response_caching",
            available=True,
            required=False,
            module="sqlite3",
            install_cmd="Built-in to Python",
        )
        
        # RAG context retrieval
        self._features["rag_retrieval"] = Feature(
            name="rag_retrieval",
            available=True,  # Our implementation is always available
            required=False,
            module="MiniLab.context",
            install_cmd="Part of MiniLab",
        )
        
        self._initialized = True
        self._report_status()
    
    def _report_status(self) -> None:
        """Report feature availability to user."""
        console.info("Feature Registry:")
        for name, feature in self._features.items():
            status = "✓" if feature.available else "✗"
            required = " (required)" if feature.required else " (optional)"
            console.info(f"  {status} {name}{required}")
    
    def is_available(self, feature_name: str) -> bool:
        """Check if a feature is available."""
        if not self._initialized:
            self.initialize()
        return self._features.get(feature_name, Feature(feature_name, False)).available
    
    def require_feature(self, feature_name: str) -> None:
        """
        Require a feature to be available.
        
        Raises FatalError if not available.
        """
        if not self._initialized:
            self.initialize()
        
        feature = self._features.get(feature_name)
        if not feature or not feature.available:
            from .errors import FatalError
            raise FatalError(
                f"Required feature '{feature_name}' is not available.\n"
                f"Install with: {feature.install_cmd if feature else 'Unknown'}",
                context={"feature": feature_name},
            )
    
    def get_feature(self, feature_name: str) -> Feature:
        """Get feature object."""
        if not self._initialized:
            self.initialize()
        return self._features.get(
            feature_name,
            Feature(feature_name, False)
        )


# Global singleton instance
_registry: FeatureRegistry = FeatureRegistry()


def get_feature_registry() -> FeatureRegistry:
    """Get the global feature registry."""
    return _registry


def is_feature_available(feature_name: str) -> bool:
    """Check if a feature is available."""
    return get_feature_registry().is_available(feature_name)


def require_feature(feature_name: str) -> None:
    """Require a feature to be available (raises FatalError if not)."""
    get_feature_registry().require_feature(feature_name)
