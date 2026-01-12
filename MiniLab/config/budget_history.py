"""
Budget History: Taxonomy-based token learning system.

Provides:
- Persistent storage of token usage organized by universal taxonomy
- Bayesian-updated predictions that improve with each run
- Per-module, per-operation, and per-tool tracking
- Global living markdown document for agent context

Design Philosophy:
- Universal taxonomy enables learning across different project types
- Statistics accumulated at multiple granularities (module, op_kind, tool_family)
- Agents receive this context as guidance for self-regulation (not hard limits)
- The living document (token_usage_learnings.md) is the single source for agents

CRITICAL: This is global learning that improves over time across ALL projects.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..core.taxonomy import (
    Module,
    OpKind,
    ToolFamily,
    TaxonomyStats,
    MODULE_COST_HEURISTICS,
    TOOL_COST_HEURISTICS,
)


def _coerce_complexity(value: Any, default: float = 0.5) -> float:
    """Coerce a complexity value into a float in [0, 1]."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(min(1.0, max(0.0, value)))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"simple", "low", "easy", "small"}:
            return 0.2
        if s in {"moderate", "medium", "normal"}:
            return 0.5
        if s in {"complex", "high", "hard", "large"}:
            return 0.8
        try:
            f = float(s)
            return float(min(1.0, max(0.0, f)))
        except Exception:
            return default
    return default


@dataclass
class ModuleStats:
    """
    Statistics for a workflow module.
    
    Uses Welford's algorithm for online mean/variance calculation.
    Tracks completed and incomplete runs separately.
    """
    module: str
    run_count: int = 0
    total_tokens: int = 0
    mean_tokens: float = 0.0
    variance: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    last_run: Optional[str] = None
    
    # Incomplete tracking
    incomplete_count: int = 0
    budget_exhausted_count: int = 0
    user_stopped_count: int = 0
    max_incomplete_tokens: int = 0
    
    # Per-complexity buckets
    by_complexity: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    def record(self, tokens: int, complexity: float = 0.5) -> None:
        """Record a completed run."""
        self.run_count += 1
        self.total_tokens += tokens
        self.last_run = datetime.now().isoformat()
        
        if self.run_count == 1:
            self.min_tokens = tokens
            self.max_tokens = tokens
        else:
            self.min_tokens = min(self.min_tokens, tokens)
            self.max_tokens = max(self.max_tokens, tokens)
        
        # Welford's algorithm
        delta = tokens - self.mean_tokens
        self.mean_tokens += delta / self.run_count
        delta2 = tokens - self.mean_tokens
        self.variance += delta * delta2
        
        # Track by complexity bucket
        bucket = self._complexity_bucket(complexity)
        if bucket not in self.by_complexity:
            self.by_complexity[bucket] = {"count": 0, "mean": 0.0, "variance": 0.0}
        stats = self.by_complexity[bucket]
        stats["count"] += 1
        c_delta = tokens - stats["mean"]
        stats["mean"] += c_delta / stats["count"]
        c_delta2 = tokens - stats["mean"]
        stats["variance"] += c_delta * c_delta2
    
    def record_incomplete(self, tokens: int, reason: str = "unknown") -> None:
        """Record an incomplete run (censored observation)."""
        self.incomplete_count += 1
        self.max_incomplete_tokens = max(self.max_incomplete_tokens, tokens)
        if reason == "budget_exhausted":
            self.budget_exhausted_count += 1
        elif reason == "user_stopped":
            self.user_stopped_count += 1
    
    def _complexity_bucket(self, complexity: float) -> str:
        if complexity < 0.33:
            return "low"
        elif complexity < 0.67:
            return "mid"
        return "high"
    
    @property
    def std_tokens(self) -> float:
        if self.run_count < 2:
            return 0.0
        return math.sqrt(self.variance / (self.run_count - 1))
    
    def estimate(self, complexity: float = 0.5) -> dict[str, Any]:
        """Get Bayesian estimate for this module."""
        # Get prior from heuristics
        prior = MODULE_COST_HEURISTICS.get(self.module, {"mean": 50000})["mean"]
        prior_weight = 2.0
        
        # Try complexity-specific first
        bucket = self._complexity_bucket(complexity)
        bucket_stats = self.by_complexity.get(bucket)
        
        if bucket_stats and bucket_stats["count"] >= 3:
            n = bucket_stats["count"]
            obs_mean = bucket_stats["mean"]
            obs_var = bucket_stats["variance"] / max(1, n - 1) if n > 1 else prior * 0.3
        elif self.run_count >= 3:
            n = self.run_count
            obs_mean = self.mean_tokens
            obs_var = self.variance / max(1, n - 1) if n > 1 else prior * 0.3
        else:
            # Not enough data
            adjustment = 0.7 + complexity * 0.6
            return {
                "estimated_tokens": int(prior * adjustment),
                "confidence_low": int(prior * adjustment * 0.5),
                "confidence_high": int(prior * adjustment * 1.5),
                "data_points": self.run_count,
                "source": "prior",
            }
        
        # Bayesian posterior
        posterior_mean = (prior_weight * prior + n * obs_mean) / (prior_weight + n)
        std = math.sqrt(obs_var) if obs_var > 0 else posterior_mean * 0.2
        ci_width = 1.28 * std / math.sqrt(max(1, n))
        
        return {
            "estimated_tokens": int(posterior_mean),
            "confidence_low": int(max(0, posterior_mean - ci_width)),
            "confidence_high": int(posterior_mean + ci_width),
            "data_points": self.run_count,
            "source": "history",
        }
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "run_count": self.run_count,
            "total_tokens": self.total_tokens,
            "mean_tokens": self.mean_tokens,
            "variance": self.variance,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "last_run": self.last_run,
            "incomplete_count": self.incomplete_count,
            "budget_exhausted_count": self.budget_exhausted_count,
            "user_stopped_count": self.user_stopped_count,
            "max_incomplete_tokens": self.max_incomplete_tokens,
            "by_complexity": self.by_complexity,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModuleStats":
        stats = cls(module=data["module"])
        stats.run_count = data.get("run_count", 0)
        stats.total_tokens = data.get("total_tokens", 0)
        stats.mean_tokens = data.get("mean_tokens", 0.0)
        stats.variance = data.get("variance", 0.0)
        stats.min_tokens = data.get("min_tokens", 0)
        stats.max_tokens = data.get("max_tokens", 0)
        stats.last_run = data.get("last_run")
        stats.incomplete_count = data.get("incomplete_count", 0)
        stats.budget_exhausted_count = data.get("budget_exhausted_count", 0)
        stats.user_stopped_count = data.get("user_stopped_count", 0)
        stats.max_incomplete_tokens = data.get("max_incomplete_tokens", 0)
        stats.by_complexity = data.get("by_complexity", {})
        return stats


@dataclass
class ToolFamilyStats:
    """Statistics for a tool family (search, filesystem, etc.)."""
    family: str
    call_count: int = 0
    total_tokens: int = 0
    mean_tokens_per_call: float = 0.0
    variance: float = 0.0
    max_tokens_per_call: int = 0
    
    def record(self, tokens: int) -> None:
        """Record a tool call."""
        self.call_count += 1
        self.total_tokens += tokens
        self.max_tokens_per_call = max(self.max_tokens_per_call, tokens)
        
        delta = tokens - self.mean_tokens_per_call
        self.mean_tokens_per_call += delta / self.call_count
        delta2 = tokens - self.mean_tokens_per_call
        self.variance += delta * delta2
    
    @property
    def std_tokens(self) -> float:
        if self.call_count < 2:
            return 0.0
        return math.sqrt(self.variance / (self.call_count - 1))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "mean_tokens_per_call": self.mean_tokens_per_call,
            "variance": self.variance,
            "max_tokens_per_call": self.max_tokens_per_call,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolFamilyStats":
        stats = cls(family=data["family"])
        stats.call_count = data.get("call_count", 0)
        stats.total_tokens = data.get("total_tokens", 0)
        stats.mean_tokens_per_call = data.get("mean_tokens_per_call", 0.0)
        stats.variance = data.get("variance", 0.0)
        stats.max_tokens_per_call = data.get("max_tokens_per_call", 0)
        return stats


class BudgetHistory:
    """
    Persistent budget history with taxonomy-based learning.
    
    Stores actual token usage organized by universal taxonomy:
    - Per-module statistics (consultation, literature_review, etc.)
    - Per-tool-family statistics (search, filesystem, terminal, etc.)
    - Combined taxonomy statistics (module + op_kind + tool_family)
    
    This data feeds into the living document that agents use for planning.
    """
    
    _instance: Optional["BudgetHistory"] = None
    
    # Name of the global learning document
    VISIBLE_DOC_NAME = "token_usage_learnings.md"
    
    def __init__(self, history_path: Optional[Path] = None):
        if history_path is None:
            history_path = Path.home() / ".minilab" / "budget_history.json"
        
        self.history_path = history_path
        self._modules: dict[str, ModuleStats] = {}
        self._tool_families: dict[str, ToolFamilyStats] = {}
        self._taxonomy_stats: dict[str, TaxonomyStats] = {}  # full taxonomy keys
        self._loaded = False
        self._workspace_root: Optional[Path] = None
        
        # Legacy support
        self._workflows: dict[str, Any] = {}  # Maps to _modules for compatibility
    
    @classmethod
    def get_instance(cls, history_path: Optional[Path] = None) -> "BudgetHistory":
        if cls._instance is None:
            cls._instance = cls(history_path)
            cls._instance.load()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        cls._instance = None
    
    def set_workspace_root(self, workspace_root: Path) -> None:
        """Set workspace root for writing the global learning document."""
        self._workspace_root = Path(workspace_root)
    
    # Legacy alias
    def set_project_root(self, project_root: Path) -> None:
        if project_root.parent.name == "Sandbox":
            self._workspace_root = project_root.parent.parent
        else:
            self._workspace_root = project_root.parent.parent
    
    def get_visible_doc_path(self) -> Optional[Path]:
        if self._workspace_root:
            return self._workspace_root / self.VISIBLE_DOC_NAME
        return None
    
    def load(self) -> None:
        """Load history from disk."""
        if self._loaded:
            return
        
        if self.history_path.exists():
            try:
                with open(self.history_path) as f:
                    data = json.load(f)
                
                # Load module stats
                for name, mod_data in data.get("modules", {}).items():
                    self._modules[name] = ModuleStats.from_dict(mod_data)
                
                # Load tool family stats
                for name, tf_data in data.get("tool_families", {}).items():
                    self._tool_families[name] = ToolFamilyStats.from_dict(tf_data)
                
                # Load taxonomy stats
                for key, tax_data in data.get("taxonomy", {}).items():
                    self._taxonomy_stats[key] = TaxonomyStats.from_dict(tax_data)
                
                # Legacy: load old workflow format into modules
                for name, wf_data in data.get("workflows", {}).items():
                    if name not in self._modules:
                        # Convert old format
                        stats = ModuleStats(module=name)
                        stats.run_count = wf_data.get("run_count", 0)
                        stats.total_tokens = wf_data.get("total_tokens", 0)
                        stats.mean_tokens = wf_data.get("mean_tokens", 0.0)
                        stats.variance = wf_data.get("variance", 0.0)
                        stats.min_tokens = wf_data.get("min_tokens", 0)
                        stats.max_tokens = wf_data.get("max_tokens", 0)
                        self._modules[name] = stats
            except (json.JSONDecodeError, KeyError):
                pass
        
        self._loaded = True
        # Set up legacy alias
        self._workflows = self._modules
    
    def save(self) -> None:
        """Save history to disk."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": 3,
            "updated": datetime.now().isoformat(),
            "modules": {name: stats.to_dict() for name, stats in self._modules.items()},
            "tool_families": {name: stats.to_dict() for name, stats in self._tool_families.items()},
            "taxonomy": {key: stats.to_dict() for key, stats in self._taxonomy_stats.items()},
        }
        
        with open(self.history_path, "w") as f:
            json.dump(data, f, indent=2)
        
        self._update_living_document()
    
    def record_run(
        self,
        workflow_name: str,
        tokens_used: int,
        status: str = "completed",
        stop_reason: Optional[str] = None,
        complexity: Any = 0.5,
        breakdown: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Record a workflow run with token usage.
        
        Args:
            workflow_name: Module name (e.g., "literature_review")
            tokens_used: Total tokens consumed
            status: "completed", "budget_exhausted", "user_stopped", "failed"
            stop_reason: Optional reason for stopping
            complexity: Project complexity 0.0-1.0
            breakdown: Optional breakdown by taxonomy keys
        """
        # Normalize to module name
        module = Module.from_string(workflow_name)
        module_name = module.value if module else workflow_name
        
        if module_name not in self._modules:
            self._modules[module_name] = ModuleStats(module=module_name)
        
        c = _coerce_complexity(complexity)
        
        if status.lower() == "completed":
            self._modules[module_name].record(tokens_used, c)
        else:
            self._modules[module_name].record_incomplete(tokens_used, status)
        
        # Record taxonomy breakdown if provided
        if breakdown:
            for key, tokens in breakdown.items():
                if key not in self._taxonomy_stats:
                    self._taxonomy_stats[key] = TaxonomyStats(key=key)
                self._taxonomy_stats[key].record(tokens)
        
        self.save()
    
    def record_tool_usage(
        self,
        tool_family: str,
        tokens: int,
    ) -> None:
        """Record tool family usage for learning."""
        if tool_family not in self._tool_families:
            self._tool_families[tool_family] = ToolFamilyStats(family=tool_family)
        self._tool_families[tool_family].record(tokens)
    
    def record_taxonomy_usage(
        self,
        module: Optional[str],
        op_kind: Optional[str],
        tool_family: Optional[str],
        tokens: int,
    ) -> None:
        """Record fine-grained taxonomy usage."""
        key = f"{module or 'unknown'}.{op_kind or 'unknown'}.{tool_family or 'unknown'}"
        if key not in self._taxonomy_stats:
            self._taxonomy_stats[key] = TaxonomyStats(key=key)
        self._taxonomy_stats[key].record(tokens)
    
    # Legacy alias
    def record(self, workflow_name: str, tokens_used: int, complexity: float = 0.5) -> None:
        self.record_run(workflow_name, tokens_used, "completed", None, complexity)
    
    def estimate(self, workflow_name: str, complexity: float = 0.5) -> dict[str, Any]:
        """Estimate tokens for a workflow module."""
        module = Module.from_string(workflow_name)
        module_name = module.value if module else workflow_name
        
        if module_name in self._modules:
            return self._modules[module_name].estimate(complexity)
        
        # No history, use heuristics
        heuristics = MODULE_COST_HEURISTICS.get(module_name, {"mean": 50000, "min": 20000, "max": 100000})
        adjustment = 0.7 + complexity * 0.6
        
        return {
            "estimated_tokens": int(heuristics["mean"] * adjustment),
            "confidence_low": int(heuristics["min"] * adjustment),
            "confidence_high": int(heuristics["max"] * adjustment),
            "data_points": 0,
            "source": "heuristic",
        }
    
    def estimate_session(self, workflows: list[str], complexity: float = 0.5) -> dict[str, Any]:
        """Estimate total tokens for a set of workflows."""
        total_est = 0
        total_low = 0
        total_high = 0
        breakdown = {}
        
        for wf in workflows:
            est = self.estimate(wf, complexity)
            total_est += est["estimated_tokens"]
            total_low += est["confidence_low"]
            total_high += est["confidence_high"]
            breakdown[wf] = est
        
        return {
            "total_estimated": total_est,
            "total_low": total_low,
            "total_high": total_high,
            "breakdown": breakdown,
            "complexity": complexity,
        }
    
    def estimate_tool_call(self, tool_family: str, op_kind: str = "retrieve") -> int:
        """Estimate tokens for a tool call."""
        # Check learned stats first
        if tool_family in self._tool_families:
            stats = self._tool_families[tool_family]
            if stats.call_count >= 3:
                return int(stats.mean_tokens_per_call)
        
        # Fall back to heuristics
        family_costs = TOOL_COST_HEURISTICS.get(tool_family, {})
        return family_costs.get(op_kind, 500)
    
    def get_learnings_context(self) -> str:
        """Get the learning document content for agent context."""
        if self._workspace_root:
            visible_path = self._workspace_root / self.VISIBLE_DOC_NAME
            if visible_path.exists():
                return visible_path.read_text()
        
        return self._generate_document_content()
    
    def get_context_for_agents(self) -> str:
        """Get formatted budget context for agent prompts."""
        return self.get_learnings_context()
    
    def _update_living_document(self) -> None:
        """Update the global token learnings document."""
        content = self._generate_document_content()
        
        if self._workspace_root:
            visible_path = self._workspace_root / self.VISIBLE_DOC_NAME
            visible_path.parent.mkdir(parents=True, exist_ok=True)
            visible_path.write_text(content)
        
        # Backup to system location
        system_path = self.history_path.parent / "token_learnings.md"
        system_path.parent.mkdir(parents=True, exist_ok=True)
        system_path.write_text(content)
    
    def _generate_document_content(self) -> str:
        """Generate the markdown content for the learning document."""
        now = datetime.now()
        
        total_runs = sum(s.run_count for s in self._modules.values())
        total_incomplete = sum(s.incomplete_count for s in self._modules.values())
        total_tokens = sum(s.total_tokens for s in self._modules.values())
        
        lines = [
            "# MiniLab Token Usage Learnings",
            "",
            f"**Last Updated:** {now.strftime('%Y-%m-%d %H:%M')}",
            f"**Total Completed Runs:** {total_runs}",
            f"**Total Incomplete Runs:** {total_incomplete}",
            f"**Total Tokens All Time:** {total_tokens:,}",
            "",
            "---",
            "",
            "## How Agents Should Use This Document",
            "",
            "This document provides historical token usage data to help you plan work.",
            "Use these estimates as **guidance** for self-regulation, not hard limits.",
            "",
            "**Key principles:**",
            "- Check remaining budget frequently during work",
            "- Prioritize core deliverables when budget is constrained",
            "- Tool calls (especially terminal output) can be expensive",
            "- Literature review and analysis are typically the most token-intensive",
            "",
            "---",
            "",
            "## Module Statistics (Completed Runs)",
            "",
            "| Module | Runs | Mean | Min | Max | Std | Incomplete | Reliability |",
            "|--------|------|------|-----|-----|-----|------------|-------------|",
        ]
        
        sorted_modules = sorted(
            self._modules.values(),
            key=lambda s: s.mean_tokens,
            reverse=True
        )
        
        for stats in sorted_modules:
            reliability = "🟢 High" if stats.run_count >= 5 else "🟡 Medium" if stats.run_count >= 2 else "🔴 Low"
            std = int(stats.std_tokens) if stats.run_count >= 2 else 0
            lines.append(
                f"| {stats.module} | {stats.run_count} | {int(stats.mean_tokens):,} | "
                f"{stats.min_tokens:,} | {stats.max_tokens:,} | ±{std:,} | "
                f"{stats.incomplete_count} | {reliability} |"
            )
        
        # Tool family section
        if self._tool_families:
            lines.extend([
                "",
                "---",
                "",
                "## Tool Family Statistics",
                "",
                "| Tool Family | Calls | Total Tokens | Mean/Call | Max/Call |",
                "|-------------|-------|--------------|-----------|----------|",
            ])
            
            sorted_tools = sorted(
                self._tool_families.values(),
                key=lambda s: s.total_tokens,
                reverse=True
            )
            
            for stats in sorted_tools:
                lines.append(
                    f"| {stats.family} | {stats.call_count} | {stats.total_tokens:,} | "
                    f"{int(stats.mean_tokens_per_call):,} | {stats.max_tokens_per_call:,} |"
                )
        
        # Planning recommendations
        lines.extend([
            "",
            "---",
            "",
            "## Planning Recommendations",
            "",
        ])
        
        if "execute_analysis" in self._modules:
            ea = self._modules["execute_analysis"]
            base = int(ea.mean_tokens + 2 * ea.std_tokens) if ea.run_count >= 2 else 150000
            if ea.budget_exhausted_count > 0 and ea.max_incomplete_tokens > 0:
                base = max(base, int(ea.max_incomplete_tokens * 1.15))
            lines.extend([
                "### Execute Analysis",
                f"- **Recommended budget:** {base:,} tokens (mean + 2σ)",
                f"- **Observed range:** {ea.min_tokens:,} – {ea.max_tokens:,}",
                f"- **Budget exhaustions:** {ea.budget_exhausted_count}",
                "- **Note:** Highly variable based on data complexity",
                "",
            ])
        
        if "literature_review" in self._modules:
            lr = self._modules["literature_review"]
            base = int(lr.mean_tokens * 1.2) if lr.run_count >= 2 else 80000
            lines.extend([
                "### Literature Review",
                f"- **Recommended budget:** {base:,} tokens",
                f"- **Mean usage:** {int(lr.mean_tokens):,}",
                "- **Note:** Scales with number of sources requested",
                "",
            ])
        
        # Token optimization tips
        lines.extend([
            "---",
            "",
            "## Token Optimization Tips",
            "",
            "### High-Cost Operations",
            "- **Terminal output:** Commands that print large outputs consume many tokens",
            "  - Use `head -50`, `tail -50`, or `grep` to limit output",
            "  - Redirect verbose output to files",
            "  - Check size first: `wc -l file.txt`",
            "",
            "- **File reading:** Reading entire large files is expensive",
            "  - Use `head` to preview structure",
            "  - Read specific line ranges when possible",
            "",
            "- **Literature searches:** Multiple search queries add up",
            "  - Plan queries before executing",
            "  - Use specific, targeted queries",
            "",
            "### Low-Cost Operations",
            "- Creating directories and writing files",
            "- Targeted searches with specific queries",
            "- Concise agent consultations",
            "",
            "---",
            "",
            "## Complexity Adjustments",
            "",
            "| Complexity | Multiplier | When to Use |",
            "|------------|------------|-------------|",
            "| Simple | 0.7x | Single file, clear objective |",
            "| Medium | 1.0x | Multiple files, some iteration |",
            "| Complex | 1.3x | Multi-modal data, novel analysis |",
            "",
            "---",
            "",
            "*This document is auto-generated after each run.*",
            "*Statistics improve with more completed runs.*",
        ])
        
        return "\n".join(lines)
    
    def get_stats(self, workflow_name: str) -> Optional[ModuleStats]:
        """Get raw statistics for a module."""
        module = Module.from_string(workflow_name)
        module_name = module.value if module else workflow_name
        return self._modules.get(module_name)
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of all historical data."""
        return {
            "total_modules_tracked": len(self._modules),
            "total_runs": sum(s.run_count for s in self._modules.values()),
            "total_tokens_all_time": sum(s.total_tokens for s in self._modules.values()),
            "modules": {
                name: {
                    "runs": s.run_count,
                    "mean": int(s.mean_tokens),
                    "std": int(s.std_tokens),
                    "min": s.min_tokens,
                    "max": s.max_tokens,
                }
                for name, s in self._modules.items()
            },
            "tool_families": {
                name: {
                    "calls": s.call_count,
                    "mean_per_call": int(s.mean_tokens_per_call),
                    "total": s.total_tokens,
                }
                for name, s in self._tool_families.items()
            },
        }


def get_budget_history() -> BudgetHistory:
    """Get the global BudgetHistory singleton."""
    return BudgetHistory.get_instance()


# Legacy compatibility: WorkflowStats is now ModuleStats
WorkflowStats = ModuleStats
