"""
Universal Taxonomy for MiniLab Token Accounting.

This module defines a universal, project-agnostic taxonomy for tracking
token usage. Unlike "phases" which are project-specific, this taxonomy
captures stable categories that enable meaningful learning across runs.

Design Philosophy:
- Universal: Works across any project type (lit review, analysis, etc.)
- Learnable: Categories are stable enough to accumulate statistics
- Actionable: Agents can use learned estimates to plan work
- Composable: Fine-grained dimensions combine into useful aggregates

Three Axes of Attribution:
1. MODULE - What workflow module is executing (consultation, literature_review, etc.)
2. OP_KIND - What type of operation is happening (retrieve, write, analyze, coordinate)
3. TOOL_FAMILY - What tool category is being used (search, filesystem, llm, terminal)

These replace the old "phase3.planning_committee" style labels which were:
- Project-specific (not learnable across projects)
- Mixed abstraction levels (phase + module in one string)
- Hard to aggregate meaningfully
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Module(str, Enum):
    """
    Workflow modules - stable across all projects.
    
    These represent the major workflow stages that MiniLab can execute.
    Each module has consistent characteristics (tool usage patterns,
    typical token consumption, agent involvement) that enable learning.
    """
    CONSULTATION = "consultation"           # Initial scoping and planning with user
    LITERATURE_REVIEW = "literature_review" # Research and background gathering
    PLANNING_COMMITTEE = "planning_committee"  # Multi-agent deliberation
    EXECUTE_ANALYSIS = "execute_analysis"   # Code execution and data processing
    WRITEUP_RESULTS = "writeup_results"     # Documentation and report generation
    CRITICAL_REVIEW = "critical_review"     # Quality assessment and iteration
    ORCHESTRATION = "orchestration"         # Bohr's coordination (summaries, routing)
    
    @classmethod
    def from_string(cls, s: str) -> Optional["Module"]:
        """Parse module from string, handling legacy formats."""
        if not s:
            return None
        normalized = s.lower().strip()
        # Handle legacy "phaseN.module_name" format
        if "." in normalized:
            parts = normalized.split(".")
            normalized = parts[-1]  # Take last part
        # Direct match
        for m in cls:
            if m.value == normalized:
                return m
        # Fuzzy match
        for m in cls:
            if normalized in m.value or m.value in normalized:
                return m
        return None


class OpKind(str, Enum):
    """
    Operation kinds - what type of work is happening.
    
    These categorize operations by their nature, independent of which
    tool performs them. This enables learning about operation costs
    regardless of implementation details.
    """
    RETRIEVE = "retrieve"       # Getting information (search, read, fetch)
    WRITE = "write"             # Creating/modifying files
    ANALYZE = "analyze"         # Processing, computation, inference
    COORDINATE = "coordinate"   # Agent communication, delegation
    EXECUTE = "execute"         # Running code, shell commands
    VALIDATE = "validate"       # Checking, testing, reviewing
    SYNTHESIZE = "synthesize"   # Combining information, summarizing
    
    @classmethod
    def from_tool_action(cls, tool: str, action: str) -> "OpKind":
        """Infer op_kind from tool and action."""
        tool_lower = tool.lower()
        action_lower = action.lower()
        
        # Retrieve operations
        if action_lower in {"search", "read", "fetch", "head", "tail", "list", "exists", "stats", "get"}:
            return cls.RETRIEVE
        
        # Write operations
        if action_lower in {"write", "append", "create", "create_dir", "save"}:
            return cls.WRITE
        
        # Execute operations
        if action_lower in {"execute", "run", "run_script"}:
            return cls.EXECUTE
        
        # Tool-specific inference
        if tool_lower in {"pubmed", "arxiv", "web_search", "citation"}:
            return cls.RETRIEVE
        if tool_lower == "code_editor":
            if action_lower in {"run", "execute"}:
                return cls.EXECUTE
            if action_lower in {"create", "edit"}:
                return cls.WRITE
            if action_lower == "syntax_check":
                return cls.VALIDATE
        if tool_lower == "terminal":
            return cls.EXECUTE
        if tool_lower == "environment":
            return cls.EXECUTE
        
        # Default based on common patterns
        if "search" in action_lower or "find" in action_lower:
            return cls.RETRIEVE
        if "write" in action_lower or "create" in action_lower:
            return cls.WRITE
        
        return cls.ANALYZE  # Default fallback


class ToolFamily(str, Enum):
    """
    Tool families - categories of tools with similar cost characteristics.
    
    Tools within a family tend to have similar token costs per operation,
    making this useful for budgeting and learning.
    """
    SEARCH = "search"           # PubMed, arXiv, web search
    FILESYSTEM = "filesystem"   # File read/write operations
    TERMINAL = "terminal"       # Shell command execution
    CODE = "code"               # Code editing and execution
    CITATION = "citation"       # Citation fetching and formatting
    ENVIRONMENT = "environment" # Package installation, env management
    LLM = "llm"                 # Direct LLM calls (agent reasoning)
    USER = "user"               # User input/interaction
    RAG = "rag"                 # Vector store and embedding operations
    
    @classmethod
    def from_tool_name(cls, tool: str) -> "ToolFamily":
        """Map tool name to family."""
        mappings = {
            "pubmed": cls.SEARCH,
            "arxiv": cls.SEARCH,
            "web_search": cls.SEARCH,
            "filesystem": cls.FILESYSTEM,
            "terminal": cls.TERMINAL,
            "code_editor": cls.CODE,
            "citation": cls.CITATION,
            "environment": cls.ENVIRONMENT,
            "user_input": cls.USER,
            "vector_store": cls.RAG,
            "embeddings": cls.RAG,
        }
        return mappings.get(tool.lower(), cls.LLM)


@dataclass
class TaxonomyContext:
    """
    Complete taxonomy context for a token transaction.
    
    This is the structured attribution that replaces the old
    workflow/trigger strings. All fields are optional to support
    gradual migration and varied contexts.
    """
    module: Optional[Module] = None
    op_kind: Optional[OpKind] = None
    tool_family: Optional[ToolFamily] = None
    
    # Additional context (not used for aggregation but useful for debugging)
    agent_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_action: Optional[str] = None
    
    # Legacy compatibility - store original workflow/trigger if provided
    legacy_workflow: Optional[str] = None
    legacy_trigger: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "module": self.module.value if self.module else None,
            "op_kind": self.op_kind.value if self.op_kind else None,
            "tool_family": self.tool_family.value if self.tool_family else None,
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "tool_action": self.tool_action,
            "legacy_workflow": self.legacy_workflow,
            "legacy_trigger": self.legacy_trigger,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaxonomyContext":
        """Deserialize from dictionary."""
        return cls(
            module=Module(data["module"]) if data.get("module") else None,
            op_kind=OpKind(data["op_kind"]) if data.get("op_kind") else None,
            tool_family=ToolFamily(data["tool_family"]) if data.get("tool_family") else None,
            agent_id=data.get("agent_id"),
            tool_name=data.get("tool_name"),
            tool_action=data.get("tool_action"),
            legacy_workflow=data.get("legacy_workflow"),
            legacy_trigger=data.get("legacy_trigger"),
        )
    
    @classmethod
    def from_legacy(cls, workflow: Optional[str], trigger: Optional[str]) -> "TaxonomyContext":
        """
        Create TaxonomyContext from legacy workflow/trigger strings.
        
        This enables backward compatibility while migrating to the new system.
        """
        ctx = cls(legacy_workflow=workflow, legacy_trigger=trigger)
        
        # Parse module from workflow string
        if workflow:
            ctx.module = Module.from_string(workflow)
        
        # Parse op_kind from trigger string
        if trigger:
            trigger_lower = trigger.lower()
            if "after_tool:" in trigger_lower:
                # Extract tool.action from trigger
                tool_action = trigger_lower.replace("after_tool:", "").strip()
                if "." in tool_action:
                    tool, action = tool_action.split(".", 1)
                    ctx.tool_name = tool
                    ctx.tool_action = action
                    ctx.op_kind = OpKind.from_tool_action(tool, action)
                    ctx.tool_family = ToolFamily.from_tool_name(tool)
            elif "after_colleague:" in trigger_lower:
                ctx.op_kind = OpKind.COORDINATE
            elif trigger_lower in {"initial", "react_loop"}:
                ctx.op_kind = OpKind.ANALYZE
        
        return ctx
    
    def aggregation_key(self) -> str:
        """
        Generate a stable key for aggregating statistics.
        
        Format: module.op_kind.tool_family (with 'unknown' for missing parts)
        """
        parts = [
            self.module.value if self.module else "unknown",
            self.op_kind.value if self.op_kind else "unknown",
            self.tool_family.value if self.tool_family else "unknown",
        ]
        return ".".join(parts)
    
    def module_key(self) -> str:
        """Key for module-level aggregation."""
        return self.module.value if self.module else "unknown"
    
    def op_key(self) -> str:
        """Key for op_kind-level aggregation."""
        return self.op_kind.value if self.op_kind else "unknown"
    
    def tool_key(self) -> str:
        """Key for tool_family-level aggregation."""
        return self.tool_family.value if self.tool_family else "unknown"


@dataclass
class TaxonomyStats:
    """
    Statistics for a single taxonomy category.
    
    Uses Welford's algorithm for online mean/variance calculation.
    Tracks both completed and incomplete runs separately.
    """
    key: str
    count: int = 0
    total_tokens: int = 0
    mean_tokens: float = 0.0
    variance: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    
    # Incomplete run tracking (for learning lower bounds)
    incomplete_count: int = 0
    max_incomplete_tokens: int = 0
    
    def record(self, tokens: int) -> None:
        """Record a completed transaction."""
        self.count += 1
        self.total_tokens += tokens
        
        if self.count == 1:
            self.min_tokens = tokens
            self.max_tokens = tokens
        else:
            self.min_tokens = min(self.min_tokens, tokens)
            self.max_tokens = max(self.max_tokens, tokens)
        
        # Welford's algorithm
        delta = tokens - self.mean_tokens
        self.mean_tokens += delta / self.count
        delta2 = tokens - self.mean_tokens
        self.variance += delta * delta2
    
    def record_incomplete(self, tokens: int) -> None:
        """Record an incomplete (censored) observation."""
        self.incomplete_count += 1
        self.max_incomplete_tokens = max(self.max_incomplete_tokens, tokens)
    
    @property
    def std_tokens(self) -> float:
        """Standard deviation of token usage."""
        if self.count < 2:
            return 0.0
        return (self.variance / (self.count - 1)) ** 0.5
    
    def estimate(self, prior_mean: float = 10000, prior_weight: float = 2.0) -> tuple[float, float]:
        """
        Bayesian estimate with confidence interval.
        
        Returns (mean_estimate, ci_width).
        """
        if self.count < 1:
            return prior_mean, prior_mean * 0.5
        
        # Bayesian posterior mean
        n = self.count
        posterior_mean = (prior_weight * prior_mean + n * self.mean_tokens) / (prior_weight + n)
        
        # Confidence interval width (80% CI)
        std = self.std_tokens if self.count >= 2 else prior_mean * 0.3
        ci_width = 1.28 * std / (n ** 0.5) if n > 0 else prior_mean * 0.5
        
        return posterior_mean, ci_width
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "key": self.key,
            "count": self.count,
            "total_tokens": self.total_tokens,
            "mean_tokens": self.mean_tokens,
            "variance": self.variance,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "incomplete_count": self.incomplete_count,
            "max_incomplete_tokens": self.max_incomplete_tokens,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaxonomyStats":
        """Deserialize from dictionary."""
        stats = cls(key=data["key"])
        stats.count = data.get("count", 0)
        stats.total_tokens = data.get("total_tokens", 0)
        stats.mean_tokens = data.get("mean_tokens", 0.0)
        stats.variance = data.get("variance", 0.0)
        stats.min_tokens = data.get("min_tokens", 0)
        stats.max_tokens = data.get("max_tokens", 0)
        stats.incomplete_count = data.get("incomplete_count", 0)
        stats.max_incomplete_tokens = data.get("max_incomplete_tokens", 0)
        return stats


# Tool cost heuristics - empirical estimates for planning
# Format: {tool_family: {op_kind: tokens_per_call}}
TOOL_COST_HEURISTICS: dict[str, dict[str, int]] = {
    "search": {
        "retrieve": 500,    # Search query + parsing results
    },
    "filesystem": {
        "retrieve": 200,    # File read (depends on size, this is typical)
        "write": 50,        # File write (output doesn't feed back to LLM)
    },
    "terminal": {
        "execute": 800,     # Command execution + output parsing
    },
    "code": {
        "write": 300,       # Code creation
        "execute": 600,     # Code execution + output
        "validate": 100,    # Syntax check
    },
    "citation": {
        "retrieve": 400,    # Citation fetch + formatting
    },
    "llm": {
        "analyze": 2000,    # Typical agent reasoning turn
        "coordinate": 1500, # Agent consultation
        "synthesize": 3000, # Summary generation
    },
}


def estimate_tool_cost(tool_family: ToolFamily, op_kind: OpKind) -> int:
    """
    Estimate token cost for a tool operation.
    
    This provides default estimates when no historical data is available.
    """
    family_costs = TOOL_COST_HEURISTICS.get(tool_family.value, {})
    return family_costs.get(op_kind.value, 500)  # Default 500 tokens


# Module cost heuristics - empirical estimates per complete module execution
MODULE_COST_HEURISTICS: dict[str, dict[str, int]] = {
    "consultation": {
        "mean": 15000,
        "min": 5000,
        "max": 40000,
    },
    "literature_review": {
        "mean": 80000,
        "min": 30000,
        "max": 200000,
    },
    "planning_committee": {
        "mean": 60000,
        "min": 20000,
        "max": 150000,
    },
    "execute_analysis": {
        "mean": 120000,
        "min": 30000,
        "max": 400000,
    },
    "writeup_results": {
        "mean": 50000,
        "min": 15000,
        "max": 120000,
    },
    "critical_review": {
        "mean": 30000,
        "min": 10000,
        "max": 80000,
    },
}


def estimate_module_cost(module: Module, complexity: float = 0.5) -> dict[str, int]:
    """
    Estimate token cost for a module execution.
    
    Args:
        module: The module to estimate
        complexity: Project complexity 0.0-1.0
        
    Returns:
        Dict with mean, min, max estimates
    """
    base = MODULE_COST_HEURISTICS.get(module.value, {"mean": 50000, "min": 20000, "max": 100000})
    
    # Scale by complexity (0.7x for simple, 1.3x for complex)
    scale = 0.7 + complexity * 0.6
    
    return {
        "mean": int(base["mean"] * scale),
        "min": int(base["min"] * scale),
        "max": int(base["max"] * scale),
    }
