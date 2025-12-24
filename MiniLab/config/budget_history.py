"""
Budget History: Historical token usage tracking with Bayesian estimation.

Provides:
- Persistent storage of actual token usage per workflow
- Bayesian-updated predictions that improve with each run
- Exponential decay weighting for recent runs
- Per-complexity-level tracking
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class WorkflowStats:
    """Statistics for a single workflow."""
    workflow_name: str
    run_count: int = 0
    total_tokens: int = 0
    mean_tokens: float = 0.0
    variance: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    last_run: Optional[str] = None
    # Per-complexity tracking (complexity as 0.0-1.0 buckets: low/mid/high)
    by_complexity: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    def record(self, tokens: int, complexity: float = 0.5) -> None:
        """Record a new run with Welford's online algorithm for mean/variance."""
        self.run_count += 1
        self.total_tokens += tokens
        self.last_run = datetime.now().isoformat()
        
        # Update min/max
        if self.run_count == 1:
            self.min_tokens = tokens
            self.max_tokens = tokens
        else:
            self.min_tokens = min(self.min_tokens, tokens)
            self.max_tokens = max(self.max_tokens, tokens)
        
        # Welford's online algorithm for mean and variance
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
    
    def _complexity_bucket(self, complexity: float) -> str:
        """Map complexity score to bucket."""
        if complexity < 0.33:
            return "low"
        elif complexity < 0.67:
            return "mid"
        return "high"
    
    @property
    def std_tokens(self) -> float:
        """Standard deviation of token usage."""
        if self.run_count < 2:
            return 0.0
        return math.sqrt(self.variance / (self.run_count - 1))
    
    def estimate(self, complexity: float = 0.5, prior_mean: float = 0.0, prior_weight: float = 2.0) -> tuple[float, float]:
        """
        Bayesian estimate combining historical data with prior.
        
        Args:
            complexity: Current project complexity (0.0-1.0)
            prior_mean: Prior expectation (from config, if no history)
            prior_weight: How much to weight the prior (equivalent sample size)
            
        Returns:
            (estimated_tokens, confidence_interval_width)
        """
        # Check complexity-specific data first
        bucket = self._complexity_bucket(complexity)
        bucket_stats = self.by_complexity.get(bucket)
        
        n: int = 0
        obs_mean: float = 0.0
        obs_var: float = prior_mean * 0.3
        
        if bucket_stats and int(bucket_stats.get("count", 0)) >= 3:
            # Use complexity-specific estimate
            n = int(bucket_stats["count"])
            obs_mean = float(bucket_stats["mean"])
            obs_var = float(bucket_stats["variance"]) / max(1, n - 1) if n > 1 else prior_mean * 0.3
        elif self.run_count >= 3:
            # Use overall estimate
            n = self.run_count
            obs_mean = self.mean_tokens
            obs_var = self.variance / max(1, n - 1) if n > 1 else prior_mean * 0.3
        else:
            # Not enough data, use prior with adjustment
            adjustment = 0.7 + complexity * 0.6  # Scale 0.7-1.3 based on complexity
            return prior_mean * adjustment, prior_mean * 0.5
        
        # Bayesian update: posterior mean is weighted average
        posterior_mean = float((prior_weight * prior_mean + n * obs_mean) / (prior_weight + n))
        
        # Confidence interval (approximate 80% CI)
        std = float(math.sqrt(obs_var)) if obs_var > 0 else posterior_mean * 0.2
        ci_width = float(1.28 * std / math.sqrt(max(1, n)))  # 80% CI
        
        return posterior_mean, ci_width
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "run_count": self.run_count,
            "total_tokens": self.total_tokens,
            "mean_tokens": self.mean_tokens,
            "variance": self.variance,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "last_run": self.last_run,
            "by_complexity": self.by_complexity,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowStats:
        return cls(
            workflow_name=str(data["workflow_name"]),
            run_count=int(data.get("run_count", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            mean_tokens=float(data.get("mean_tokens", 0.0)),
            variance=float(data.get("variance", 0.0)),
            min_tokens=int(data.get("min_tokens", 0)),
            max_tokens=int(data.get("max_tokens", 0)),
            last_run=data.get("last_run"),
            by_complexity=dict(data.get("by_complexity", {})),
        )


class BudgetHistory:
    """
    Persistent budget history with learning capabilities.
    
    Stores actual token usage per workflow and uses Bayesian updating
    to improve future estimates. Data is stored in ~/.minilab/budget_history.json.
    """
    
    _instance: Optional[BudgetHistory] = None
    
    # Default priors (from budgets.yaml cost_estimates, roughly scaled)
    DEFAULT_PRIORS = {
        "consultation": 15_000,
        "literature_review": 80_000,
        "planning_committee": 60_000,
        "execute_analysis": 150_000,
        "writeup_results": 60_000,
        "critical_review": 40_000,
    }
    
    def __init__(self, history_path: Optional[Path] = None):
        if history_path is None:
            history_path = Path.home() / ".minilab" / "budget_history.json"
        
        self.history_path = history_path
        self._workflows: dict[str, WorkflowStats] = {}
        self._loaded = False
        self._decay_lambda = 0.9  # Exponential decay for old data
    
    @classmethod
    def get_instance(cls, history_path: Optional[Path] = None) -> BudgetHistory:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(history_path)
            cls._instance.load()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
    
    def load(self) -> None:
        """Load history from disk."""
        if self._loaded:
            return
        
        if self.history_path.exists():
            try:
                with open(self.history_path) as f:
                    data = json.load(f)
                
                for wf_name, wf_data in data.get("workflows", {}).items():
                    self._workflows[wf_name] = WorkflowStats.from_dict(wf_data)
            except (json.JSONDecodeError, KeyError):
                pass  # Start fresh if corrupted
        
        self._loaded = True
    
    def save(self) -> None:
        """Save history to disk."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        
        data: dict[str, Any] = {
            "version": 1,
            "updated": datetime.now().isoformat(),
            "workflows": {
                name: stats.to_dict()
                for name, stats in self._workflows.items()
            },
        }
        
        with open(self.history_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def record(self, workflow_name: str, tokens_used: int, complexity: float = 0.5) -> None:
        """
        Record actual token usage for a workflow run.
        
        Args:
            workflow_name: Name of the workflow
            tokens_used: Actual tokens consumed
            complexity: Project complexity score (0.0-1.0)
        """
        if workflow_name not in self._workflows:
            self._workflows[workflow_name] = WorkflowStats(workflow_name=workflow_name)
        
        self._workflows[workflow_name].record(tokens_used, complexity)
        self.save()
    
    def estimate(self, workflow_name: str, complexity: float = 0.5) -> dict[str, Any]:
        """
        Estimate tokens needed for a workflow.
        
        Returns dict with:
        - estimated_tokens: Best estimate
        - confidence_low: Lower bound (80% CI)
        - confidence_high: Upper bound (80% CI)
        - data_points: Number of historical runs used
        - source: "history" or "prior"
        """
        prior = self.DEFAULT_PRIORS.get(workflow_name, 50_000)
        
        if workflow_name in self._workflows:
            stats = self._workflows[workflow_name]
            mean, ci_width = stats.estimate(complexity, prior_mean=prior)
            
            return {
                "estimated_tokens": int(mean),
                "confidence_low": int(max(0, mean - ci_width)),
                "confidence_high": int(mean + ci_width),
                "data_points": stats.run_count,
                "source": "history" if stats.run_count >= 3 else "prior+history",
            }
        
        # No history, use scaled prior
        adjustment = 0.7 + complexity * 0.6
        scaled = int(prior * adjustment)
        
        return {
            "estimated_tokens": scaled,
            "confidence_low": int(scaled * 0.6),
            "confidence_high": int(scaled * 1.5),
            "data_points": 0,
            "source": "prior",
        }
    
    def estimate_session(self, workflows: list[str], complexity: float = 0.5) -> dict[str, Any]:
        """
        Estimate total tokens for a full session.
        
        Args:
            workflows: List of workflow names to include
            complexity: Project complexity score
            
        Returns:
            Combined estimate with breakdown
        """
        total_est = 0
        total_low = 0
        total_high = 0
        breakdown: dict[str, Any] = {}
        
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
    
    def get_stats(self, workflow_name: str) -> Optional[WorkflowStats]:
        """Get raw statistics for a workflow."""
        return self._workflows.get(workflow_name)
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of all historical data."""
        return {
            "total_workflows_tracked": len(self._workflows),
            "total_runs": sum(s.run_count for s in self._workflows.values()),
            "total_tokens_all_time": sum(s.total_tokens for s in self._workflows.values()),
            "workflows": {
                name: {
                    "runs": s.run_count,
                    "mean": int(s.mean_tokens),
                    "std": int(s.std_tokens),
                    "min": s.min_tokens,
                    "max": s.max_tokens,
                }
                for name, s in self._workflows.items()
            },
        }


def get_budget_history() -> BudgetHistory:
    """Get the global BudgetHistory singleton."""
    return BudgetHistory.get_instance()

