"""
Token Learning System.

Tracks token usage across runs to enable learning and better estimation.

Components:
- token_model.md: Human-readable model documentation
- token_runs_recent.jsonl: Recent run data for learning
- TokenLearner: Statistical learning from historical data
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class TokenRunRecord:
    """Record of a single module execution's token usage."""
    
    run_id: str
    timestamp: str
    module_name: str
    agent_id: str
    
    # Token counts
    estimated_tokens: int
    actual_input_tokens: int
    actual_output_tokens: int
    actual_total_tokens: int
    
    # Context
    project_name: str = ""
    task_description: str = ""
    
    # Outcome
    success: bool = True
    error: Optional[str] = None
    duration_seconds: float = 0.0
    
    # Tool usage
    tools_used: list[str] = field(default_factory=list)
    tool_calls: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "module_name": self.module_name,
            "agent_id": self.agent_id,
            "estimated_tokens": self.estimated_tokens,
            "actual_input_tokens": self.actual_input_tokens,
            "actual_output_tokens": self.actual_output_tokens,
            "actual_total_tokens": self.actual_total_tokens,
            "project_name": self.project_name,
            "task_description": self.task_description,
            "success": self.success,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "tools_used": self.tools_used,
            "tool_calls": self.tool_calls,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenRunRecord":
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            module_name=data["module_name"],
            agent_id=data["agent_id"],
            estimated_tokens=data.get("estimated_tokens", 0),
            actual_input_tokens=data.get("actual_input_tokens", 0),
            actual_output_tokens=data.get("actual_output_tokens", 0),
            actual_total_tokens=data.get("actual_total_tokens", 0),
            project_name=data.get("project_name", ""),
            task_description=data.get("task_description", ""),
            success=data.get("success", True),
            error=data.get("error"),
            duration_seconds=data.get("duration_seconds", 0.0),
            tools_used=data.get("tools_used", []),
            tool_calls=data.get("tool_calls", 0),
        )
    
    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        return json.dumps(self.to_dict())


@dataclass
class ModuleTokenStats:
    """Aggregated statistics for a module's token usage."""
    
    module_name: str
    sample_count: int = 0
    
    # Estimation accuracy
    mean_estimate: float = 0.0
    mean_actual: float = 0.0
    estimation_ratio: float = 1.0  # actual / estimated
    
    # Distribution stats
    min_tokens: int = 0
    max_tokens: int = 0
    std_dev: float = 0.0
    median_tokens: float = 0.0
    
    # Success rate
    success_rate: float = 1.0
    
    # Timing
    mean_duration: float = 0.0


class TokenLearner:
    """
    Learn token usage patterns from historical data.
    
    Maintains running statistics and provides improved estimates
    based on past performance.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the token learner.
        
        Args:
            data_path: Path to the token_runs_recent.jsonl file
        """
        self.data_path = data_path
        self._records: list[TokenRunRecord] = []
        self._stats_by_module: dict[str, ModuleTokenStats] = {}
        
        if data_path and data_path.exists():
            self._load_records()
    
    def _load_records(self) -> None:
        """Load records from JSONL file."""
        if not self.data_path or not self.data_path.exists():
            return
        
        self._records = []
        with open(self.data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        self._records.append(TokenRunRecord.from_dict(data))
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        self._compute_stats()
    
    def _compute_stats(self) -> None:
        """Compute statistics by module."""
        by_module: dict[str, list[TokenRunRecord]] = {}
        
        for record in self._records:
            if record.module_name not in by_module:
                by_module[record.module_name] = []
            by_module[record.module_name].append(record)
        
        self._stats_by_module = {}
        
        for module_name, records in by_module.items():
            actuals = [r.actual_total_tokens for r in records]
            estimates = [r.estimated_tokens for r in records if r.estimated_tokens > 0]
            durations = [r.duration_seconds for r in records if r.duration_seconds > 0]
            successes = [r for r in records if r.success]
            
            stats = ModuleTokenStats(module_name=module_name)
            stats.sample_count = len(records)
            
            if actuals:
                stats.mean_actual = statistics.mean(actuals)
                stats.min_tokens = min(actuals)
                stats.max_tokens = max(actuals)
                stats.median_tokens = statistics.median(actuals)
                if len(actuals) > 1:
                    stats.std_dev = statistics.stdev(actuals)
            
            if estimates:
                stats.mean_estimate = statistics.mean(estimates)
                if stats.mean_estimate > 0:
                    stats.estimation_ratio = stats.mean_actual / stats.mean_estimate
            
            stats.success_rate = len(successes) / len(records) if records else 1.0
            
            if durations:
                stats.mean_duration = statistics.mean(durations)
            
            self._stats_by_module[module_name] = stats
    
    def record_run(self, record: TokenRunRecord) -> None:
        """
        Record a new run.
        
        Args:
            record: The run record to add
        """
        self._records.append(record)
        
        # Append to file
        if self.data_path:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_path, "a") as f:
                f.write(record.to_jsonl() + "\n")
        
        # Update stats
        self._compute_stats()
    
    def get_estimate(self, module_name: str, default: int = 50000) -> int:
        """
        Get a token estimate for a module based on historical data.
        
        Args:
            module_name: The module to estimate
            default: Default estimate if no history
            
        Returns:
            Estimated token count
        """
        stats = self._stats_by_module.get(module_name)
        
        if not stats or stats.sample_count < 3:
            return default
        
        # Use median + 1 std dev as a conservative estimate
        estimate = int(stats.median_tokens + stats.std_dev)
        
        # Clamp to reasonable bounds
        return max(1000, min(estimate, 500000))
    
    def get_module_stats(self, module_name: str) -> Optional[ModuleTokenStats]:
        """Get statistics for a specific module."""
        return self._stats_by_module.get(module_name)
    
    def get_all_stats(self) -> dict[str, ModuleTokenStats]:
        """Get statistics for all modules."""
        return self._stats_by_module.copy()
    
    def get_estimation_accuracy(self, module_name: str) -> Optional[float]:
        """
        Get how accurate estimates have been for a module.
        
        Returns:
            Ratio of actual/estimated (1.0 = perfect, >1 = underestimate)
        """
        stats = self._stats_by_module.get(module_name)
        return stats.estimation_ratio if stats else None
    
    def generate_model_markdown(self) -> str:
        """
        Generate a human-readable markdown document of the token model.
        
        Returns:
            Markdown string documenting the learned model
        """
        lines = [
            "# Token Usage Model",
            "",
            f"Generated: {datetime.now().isoformat()}",
            f"Total runs analyzed: {len(self._records)}",
            "",
            "## Module Statistics",
            "",
        ]
        
        for module_name, stats in sorted(self._stats_by_module.items()):
            lines.extend([
                f"### {module_name}",
                "",
                f"- **Sample count**: {stats.sample_count}",
                f"- **Mean actual tokens**: {stats.mean_actual:,.0f}",
                f"- **Median tokens**: {stats.median_tokens:,.0f}",
                f"- **Std deviation**: {stats.std_dev:,.0f}",
                f"- **Range**: {stats.min_tokens:,} - {stats.max_tokens:,}",
                f"- **Estimation accuracy**: {stats.estimation_ratio:.2f}x",
                f"- **Success rate**: {stats.success_rate:.1%}",
                f"- **Mean duration**: {stats.mean_duration:.1f}s",
                "",
            ])
        
        lines.extend([
            "## Recommendations",
            "",
            "Based on the data:",
            "",
        ])
        
        # Add recommendations
        for module_name, stats in self._stats_by_module.items():
            if stats.estimation_ratio > 1.5:
                lines.append(
                    f"- **{module_name}**: Estimates are too low. "
                    f"Recommended estimate: {int(stats.median_tokens + stats.std_dev):,}"
                )
            elif stats.estimation_ratio < 0.5:
                lines.append(
                    f"- **{module_name}**: Estimates are too high. "
                    f"Recommended estimate: {int(stats.median_tokens + stats.std_dev):,}"
                )
        
        return "\n".join(lines)
    
    def save_model(self, model_path: Path) -> None:
        """Save the model documentation to a markdown file."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(self.generate_model_markdown())
    
    def prune_old_records(self, keep_last: int = 1000) -> int:
        """
        Prune old records, keeping only the most recent.
        
        Args:
            keep_last: Number of records to keep
            
        Returns:
            Number of records removed
        """
        if len(self._records) <= keep_last:
            return 0
        
        removed = len(self._records) - keep_last
        self._records = self._records[-keep_last:]
        
        # Rewrite file
        if self.data_path:
            with open(self.data_path, "w") as f:
                for record in self._records:
                    f.write(record.to_jsonl() + "\n")
        
        self._compute_stats()
        return removed


# Global instance
_learner: Optional[TokenLearner] = None


def get_token_learner(data_path: Optional[Path] = None) -> TokenLearner:
    """Get the global token learner instance."""
    global _learner
    
    if _learner is None:
        # Default path
        if data_path is None:
            data_path = Path(__file__).parent.parent / "config" / "token_runs_recent.jsonl"
        _learner = TokenLearner(data_path)
    
    return _learner


def record_token_usage(
    module_name: str,
    agent_id: str,
    estimated_tokens: int,
    actual_input_tokens: int,
    actual_output_tokens: int,
    project_name: str = "",
    task_description: str = "",
    success: bool = True,
    error: Optional[str] = None,
    duration_seconds: float = 0.0,
    tools_used: Optional[list[str]] = None,
    tool_calls: int = 0,
) -> TokenRunRecord:
    """
    Convenience function to record token usage.
    
    Returns the created record.
    """
    from uuid import uuid4
    
    record = TokenRunRecord(
        run_id=str(uuid4()),
        timestamp=datetime.now().isoformat(),
        module_name=module_name,
        agent_id=agent_id,
        estimated_tokens=estimated_tokens,
        actual_input_tokens=actual_input_tokens,
        actual_output_tokens=actual_output_tokens,
        actual_total_tokens=actual_input_tokens + actual_output_tokens,
        project_name=project_name,
        task_description=task_description,
        success=success,
        error=error,
        duration_seconds=duration_seconds,
        tools_used=tools_used or [],
        tool_calls=tool_calls,
    )
    
    get_token_learner().record_run(record)
    return record
