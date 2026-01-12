"""
TokenAccount: Real-time token budget tracking with universal taxonomy.

Provides:
- Single shared instance across all LLM calls
- Real-time balance tracking and reporting
- Automatic mode degradation warnings
- Guidance-based budgeting (not hard stops)
- Graceful shutdown reserve for completion
- Accurate per-model pricing with input/output/cache differentiation
- Universal taxonomy for meaningful aggregation and learning

The taxonomy system replaces the old "phase3.planning_committee" style labels
with structured attributes (module, op_kind, tool_family) that enable:
- Stable aggregation across projects
- Meaningful learning from historical runs
- Fine-grained cost attribution
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .taxonomy import TaxonomyContext, Module, OpKind, ToolFamily


# Pricing per million tokens (as of January 2026)
# Format: {model_pattern: {input, output, cache_write_5m, cache_write_1h, cache_read}}
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Claude Opus 4.5
    "claude-opus-4-5": {
        "input": 5.00,
        "output": 25.00,
        "cache_write_5m": 6.25,    # 1.25x input
        "cache_write_1h": 10.00,   # 2x input
        "cache_read": 0.50,        # 0.1x input
    },
    # Claude Opus 4/4.1
    "claude-opus-4": {
        "input": 15.00,
        "output": 75.00,
        "cache_write_5m": 18.75,
        "cache_write_1h": 30.00,
        "cache_read": 1.50,
    },
    # Claude Sonnet 4/4.5
    "claude-sonnet-4": {
        "input": 3.00,
        "output": 15.00,
        "cache_write_5m": 3.75,
        "cache_write_1h": 6.00,
        "cache_read": 0.30,
    },
    # Claude Haiku 4.5
    "claude-haiku-4-5": {
        "input": 1.00,
        "output": 5.00,
        "cache_write_5m": 1.25,
        "cache_write_1h": 2.00,
        "cache_read": 0.10,
    },
    # Claude Haiku 3.5
    "claude-haiku-3-5": {
        "input": 0.80,
        "output": 4.00,
        "cache_write_5m": 1.00,
        "cache_write_1h": 1.60,
        "cache_read": 0.08,
    },
    # Default fallback (Sonnet pricing as baseline)
    "default": {
        "input": 3.00,
        "output": 15.00,
        "cache_write_5m": 3.75,
        "cache_write_1h": 6.00,
        "cache_read": 0.30,
    },
}


def get_model_pricing(model: str) -> dict[str, float]:
    """Get pricing for a model, with pattern matching for variants."""
    model_lower = model.lower()
    
    # Check exact matches and patterns
    for pattern, pricing in MODEL_PRICING.items():
        if pattern in model_lower:
            return pricing
    
    return MODEL_PRICING["default"]


class BudgetExceededError(Exception):
    """Raised when token budget has been exceeded."""
    
    def __init__(self, used: int, budget: int, message: str = None):
        self.used = used
        self.budget = budget
        self.percentage = (used / budget * 100) if budget > 0 else 0
        msg = message or f"Token budget exceeded: {used:,}/{budget:,} ({self.percentage:.1f}%)"
        super().__init__(msg)


@dataclass
class TokenTransaction:
    """Record of a single token debit with taxonomy attribution."""
    timestamp: datetime
    agent_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    balance_after: int
    
    # Legacy fields (kept for backward compatibility)
    workflow: Optional[str] = None
    trigger: Optional[str] = None
    operation: str = ""  # e.g., "pubmed.search", "llm.complete"
    
    # New taxonomy fields
    module: Optional[str] = None       # e.g., "literature_review"
    op_kind: Optional[str] = None      # e.g., "retrieve"
    tool_family: Optional[str] = None  # e.g., "search"
    
    def taxonomy_key(self) -> str:
        """Generate taxonomy aggregation key."""
        parts = [
            self.module or "unknown",
            self.op_kind or "unknown",
            self.tool_family or "unknown",
        ]
        return ".".join(parts)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "agent_id": self.agent_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "balance_after": self.balance_after,
            "workflow": self.workflow,
            "trigger": self.trigger,
            "operation": self.operation,
            "module": self.module,
            "op_kind": self.op_kind,
            "tool_family": self.tool_family,
        }


@dataclass
class ToolUsageStats:
    """Aggregated statistics for a single tool."""
    tool_name: str
    call_count: int = 0
    total_output_chars: int = 0  # Characters injected back into the LLM (proxy for token cost)
    total_raw_output_chars: int = 0  # Raw tool output size (for observability)
    estimated_tokens: int = 0  # ~4 chars per token estimate (based on injected chars)
    mean_output_chars: float = 0.0
    max_output_chars: int = 0
    max_raw_output_chars: int = 0
    error_count: int = 0
    
    def record_call(self, output_chars: int, success: bool = True, raw_output_chars: Optional[int] = None) -> None:
        """Record a tool call.

        Args:
            output_chars: Approximate chars injected back into the LLM.
            raw_output_chars: Size of the full raw tool output (optional).
        """
        self.call_count += 1
        if not success:
            self.error_count += 1
        self.total_output_chars += output_chars
        if raw_output_chars is None:
            raw_output_chars = output_chars
        self.total_raw_output_chars += raw_output_chars
        self.estimated_tokens = self.total_output_chars // 4  # ~4 chars per token
        self.mean_output_chars = self.total_output_chars / self.call_count
        self.max_output_chars = max(self.max_output_chars, output_chars)
        self.max_raw_output_chars = max(self.max_raw_output_chars, raw_output_chars)


class TokenAccount:
    """
    Singleton token budget manager.
    
    All LLM calls debit from this shared account, enabling:
    - Real-time tracking across all agents
    - Budget warnings at configurable thresholds
    - Hard stops when budget is exceeded
    - Transaction history for auditing
    
    Usage:
        account = get_token_account()
        account.set_budget(100_000)
        
        # Before each LLM call, check if allowed
        if not account.can_spend(estimated_tokens):
            # Handle budget exceeded
            ...
        
        # After call, record actual usage
        account.debit(actual_input, actual_output, agent_id, operation)
    """
    
    _instance: Optional[TokenAccount] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._budget: Optional[int] = None
        self._total_input: int = 0
        self._total_output: int = 0
        self._total_cache_creation: int = 0
        self._total_cache_read: int = 0
        self._transactions: list[TokenTransaction] = []
        
        # Model tracking for accurate pricing
        self._model: str = "claude-sonnet-4-5"  # Default model
        
        # Callbacks
        self._on_warning: Optional[Callable[[int, int, float], None]] = None
        self._on_budget_exceeded: Optional[Callable[[int, int], None]] = None
        
        # Warning thresholds (percentages)
        self._warning_thresholds = [60, 80, 95]
        self._warnings_issued: set[int] = set()
        
        # Per-tool tracking: tool_name -> ToolUsageStats
        self._tool_usage: dict[str, ToolUsageStats] = {}

        # Optional per-workflow token caps (absolute tokens).
        # Keys are workflow prefixes (e.g., "phase2.consultation").
        self._workflow_caps: dict[str, int] = {}
        
        self._initialized = True
    
    def reset(self) -> None:
        """Reset account state for a new session."""
        self._budget = None
        self._total_input = 0
        self._total_output = 0
        self._total_cache_creation = 0
        self._total_cache_read = 0
        self._transactions = []
        self._warnings_issued = set()
        self._model = "claude-sonnet-4-5"
        self._tool_usage = {}
        self._workflow_caps = {}
    
    def set_budget(self, budget: int, model: str = None) -> None:
        """Set the token budget for this session."""
        self._budget = budget
        self._warnings_issued = set()
        if model:
            self._model = model

    def set_workflow_caps(self, caps: dict[str, int]) -> None:
        """Set/replace per-workflow caps.

        Args:
            caps: Mapping of workflow prefix -> max tokens allowed for that prefix.
        """
        self._workflow_caps = {str(k): int(v) for k, v in (caps or {}).items() if v is not None}

    def get_workflow_usage(self, workflow_prefix: str) -> int:
        """Return total tokens attributed to a workflow prefix."""
        prefix = (workflow_prefix or "").strip()
        if not prefix:
            return 0
        total = 0
        for t in self._transactions:
            if t.workflow and str(t.workflow).startswith(prefix):
                total += int(t.total_tokens)
        return total

    def get_workflow_cap(self, workflow: str | None) -> Optional[int]:
        """Return the matching cap for a workflow, if any (longest-prefix match)."""
        if not workflow:
            return None
        w = str(workflow)
        best_key = None
        for key in self._workflow_caps.keys():
            if w.startswith(key) and (best_key is None or len(key) > len(best_key)):
                best_key = key
        return self._workflow_caps.get(best_key) if best_key else None

    def enforce_workflow_cap(self, workflow: str | None, planned_tokens: int = 0) -> None:
        """Raise BudgetExceededError if the workflow would exceed its cap."""
        cap = self.get_workflow_cap(workflow)
        if cap is None:
            return
        used = self.get_workflow_usage(str(workflow))
        if used + int(planned_tokens) > int(cap):
            raise BudgetExceededError(
                used=used,
                budget=int(cap),
                message=(
                    f"Workflow cap exceeded for '{workflow}': "
                    f"{used:,}/{cap:,} used (planned +{int(planned_tokens):,}). "
                    "This phase is capped to preserve budget for execution."
                ),
            )
    
    @property
    def model(self) -> str:
        """Get the current model."""
        return self._model
    
    @property
    def budget(self) -> Optional[int]:
        """Get the current budget."""
        return self._budget
    
    @property
    def total_used(self) -> int:
        """Total tokens used (input + output)."""
        return self._total_input + self._total_output
    
    @property
    def remaining(self) -> int:
        """Tokens remaining in budget."""
        if self._budget is None:
            return float('inf')
        return max(0, self._budget - self.total_used)
    
    @property
    def percentage_used(self) -> float:
        """Percentage of budget used."""
        if self._budget is None or self._budget == 0:
            return 0.0
        return (self.total_used / self._budget) * 100
    
    @property
    def usage_summary(self) -> dict[str, Any]:
        """Get comprehensive usage summary."""
        return {
            "budget": self._budget,
            "total_input": self._total_input,
            "total_output": self._total_output,
            "total_used": self.total_used,
            "remaining": self.remaining if self._budget else None,
            "percentage_used": self.percentage_used,
            "cache_creation": self._total_cache_creation,
            "cache_read": self._total_cache_read,
            "transaction_count": len(self._transactions),
            "estimated_cost": self._estimate_cost(),
            "cost_breakdown": self._cost_breakdown(),
            "cache_efficiency": self.cache_efficiency,
        }
    
    def _estimate_cost(self, model: str = None) -> float:
        """
        Estimate cost using accurate per-model pricing.
        
        Calculates:
        - Input tokens at input rate
        - Output tokens at output rate (typically higher)
        - Cache creation at 1.25x input rate (5-minute TTL default)
        - Cache reads at 0.1x input rate
        """
        pricing = get_model_pricing(model or self._model)
        
        # Regular input tokens (excluding cache reads which are cheaper)
        regular_input = max(0, self._total_input - self._total_cache_read)
        
        cost = 0.0
        cost += (regular_input / 1_000_000) * pricing["input"]
        cost += (self._total_output / 1_000_000) * pricing["output"]
        cost += (self._total_cache_creation / 1_000_000) * pricing["cache_write_5m"]
        cost += (self._total_cache_read / 1_000_000) * pricing["cache_read"]
        
        return round(cost, 4)
    
    def _cost_breakdown(self, model: str = None) -> dict[str, float]:
        """Get detailed cost breakdown by category."""
        pricing = get_model_pricing(model or self._model)
        regular_input = max(0, self._total_input - self._total_cache_read)
        
        return {
            "input": round((regular_input / 1_000_000) * pricing["input"], 4),
            "output": round((self._total_output / 1_000_000) * pricing["output"], 4),
            "cache_write": round((self._total_cache_creation / 1_000_000) * pricing["cache_write_5m"], 4),
            "cache_read": round((self._total_cache_read / 1_000_000) * pricing["cache_read"], 4),
        }
    
    @property
    def cache_efficiency(self) -> dict[str, Any]:
        """Calculate cache efficiency metrics."""
        total_cacheable = self._total_cache_creation + self._total_cache_read
        if total_cacheable == 0:
            return {"hit_rate": 0.0, "tokens_saved": 0, "cost_saved": 0.0}
        
        hit_rate = (self._total_cache_read / total_cacheable) * 100
        
        # Cost saved = difference between full input price and cache read price
        pricing = get_model_pricing(self._model)
        savings_per_token = (pricing["input"] - pricing["cache_read"]) / 1_000_000
        cost_saved = self._total_cache_read * savings_per_token
        
        return {
            "hit_rate": round(hit_rate, 1),
            "tokens_cached": self._total_cache_creation,
            "tokens_from_cache": self._total_cache_read,
            "cost_saved": round(cost_saved, 4),
        }
    
    def estimate_task_cost(self, estimated_input: int, estimated_output: int, 
                          cache_hit_rate: float = 0.5, model: str = None) -> float:
        """
        Estimate cost for a planned task.
        
        Args:
            estimated_input: Expected input tokens
            estimated_output: Expected output tokens
            cache_hit_rate: Expected cache hit rate (0.0-1.0)
            model: Model to use for pricing (defaults to current)
            
        Returns:
            Estimated cost in USD
        """
        pricing = get_model_pricing(model or self._model)
        
        # Split input between cached and uncached based on expected hit rate
        cached_input = int(estimated_input * cache_hit_rate)
        uncached_input = estimated_input - cached_input
        
        cost = 0.0
        cost += (uncached_input / 1_000_000) * pricing["input"]
        cost += (cached_input / 1_000_000) * pricing["cache_read"]
        cost += (estimated_output / 1_000_000) * pricing["output"]
        
        return round(cost, 4)
    
    def can_spend(self, estimated_tokens: int) -> bool:
        """Check if spending estimated tokens would exceed budget."""
        if self._budget is None:
            return True
        return (self.total_used + estimated_tokens) <= self._budget
    
    def is_budget_exceeded(self) -> bool:
        """Check if budget has been exceeded."""
        if self._budget is None:
            return False
        return self.total_used >= self._budget
    
    def debit(
        self,
        input_tokens: int,
        output_tokens: int,
        agent_id: str = "unknown",
        operation: str = "",
        workflow: Optional[str] = None,
        trigger: Optional[str] = None,
        cache_creation: int = 0,
        cache_read: int = 0,
        taxonomy: Optional["TaxonomyContext"] = None,
    ) -> TokenTransaction:
        """
        Record token usage with taxonomy attribution.
        
        Args:
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            agent_id: Which agent made this call
            operation: What operation (e.g., "pubmed.search")
            workflow: Legacy workflow string (deprecated, use taxonomy)
            trigger: Legacy trigger string (deprecated, use taxonomy)
            cache_creation: Cache creation tokens
            cache_read: Cache read tokens
            taxonomy: TaxonomyContext for structured attribution
        
        Returns:
            TokenTransaction record
        """
        self._total_input += input_tokens
        self._total_output += output_tokens
        self._total_cache_creation += cache_creation
        self._total_cache_read += cache_read
        
        total = input_tokens + output_tokens
        
        # Extract taxonomy fields if provided
        module_str: Optional[str] = None
        op_kind_str: Optional[str] = None
        tool_family_str: Optional[str] = None
        
        if taxonomy:
            module_str = taxonomy.module.value if taxonomy.module else None
            op_kind_str = taxonomy.op_kind.value if taxonomy.op_kind else None
            tool_family_str = taxonomy.tool_family.value if taxonomy.tool_family else None
            # Use taxonomy's legacy fields if not provided directly
            if workflow is None:
                workflow = taxonomy.legacy_workflow
            if trigger is None:
                trigger = taxonomy.legacy_trigger
            if agent_id == "unknown" and taxonomy.agent_id:
                agent_id = taxonomy.agent_id
        
        # If no taxonomy but we have workflow/trigger, try to parse them
        if not taxonomy and (workflow or trigger):
            from .taxonomy import TaxonomyContext
            parsed = TaxonomyContext.from_legacy(workflow, trigger)
            module_str = parsed.module.value if parsed.module else None
            op_kind_str = parsed.op_kind.value if parsed.op_kind else None
            tool_family_str = parsed.tool_family.value if parsed.tool_family else None
        
        transaction = TokenTransaction(
            timestamp=datetime.now(),
            agent_id=agent_id,
            workflow=workflow,
            trigger=trigger,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            balance_after=self.remaining if self._budget else -1,
            operation=operation,
            module=module_str,
            op_kind=op_kind_str,
            tool_family=tool_family_str,
        )
        self._transactions.append(transaction)
        
        # Check warnings (guidance, not hard stops)
        self._check_warnings()
        
        return transaction
    
    def _check_warnings(self) -> None:
        """Check and issue budget warnings."""
        if self._budget is None:
            return
        
        pct = self.percentage_used
        
        for threshold in self._warning_thresholds:
            if pct >= threshold and threshold not in self._warnings_issued:
                self._warnings_issued.add(threshold)
                if self._on_warning:
                    self._on_warning(self.total_used, self._budget, pct)
    
    def set_warning_callback(
        self, 
        callback: Callable[[int, int, float], None]
    ) -> None:
        """
        Set callback for budget warnings.
        
        Callback receives: (total_used, budget, percentage)
        """
        self._on_warning = callback
    
    def set_exceeded_callback(
        self,
        callback: Callable[[int, int], None]
    ) -> None:
        """
        Set callback for budget exceeded.
        
        Callback receives: (total_used, budget)
        """
        self._on_budget_exceeded = callback
    
    def get_agent_usage(self, agent_id: str) -> dict[str, int]:
        """Get usage for a specific agent."""
        input_total = sum(t.input_tokens for t in self._transactions if t.agent_id == agent_id)
        output_total = sum(t.output_tokens for t in self._transactions if t.agent_id == agent_id)
        return {
            "input_tokens": input_total,
            "output_tokens": output_total,
            "total_tokens": input_total + output_total,
            "call_count": len([t for t in self._transactions if t.agent_id == agent_id]),
        }

    def iter_transactions(self) -> list[TokenTransaction]:
        """Return a copy of all transactions (authoritative accounting)."""
        return list(self._transactions)

    def aggregate(
        self,
        keys: tuple[str, ...] = ("module", "agent_id"),
    ) -> list[dict[str, Any]]:
        """Aggregate token usage by selected fields.

        Keys may include: module, op_kind, tool_family, agent_id, operation, workflow, trigger.
        
        Prefer taxonomy keys (module, op_kind, tool_family) over legacy keys for
        meaningful aggregation.
        """
        allowed = {"module", "op_kind", "tool_family", "workflow", "agent_id", "operation", "trigger"}
        for k in keys:
            if k not in allowed:
                raise ValueError(f"Unsupported aggregation key: {k}")

        buckets: dict[tuple[Any, ...], dict[str, Any]] = {}
        for t in self._transactions:
            bucket_key = tuple(getattr(t, k) for k in keys)
            b = buckets.get(bucket_key)
            if b is None:
                b = {k: getattr(t, k) for k in keys}
                b.update({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "call_count": 0})
                buckets[bucket_key] = b
            b["input_tokens"] += t.input_tokens
            b["output_tokens"] += t.output_tokens
            b["total_tokens"] += t.total_tokens
            b["call_count"] += 1

        rows = list(buckets.values())
        rows.sort(key=lambda r: r.get("total_tokens", 0), reverse=True)
        return rows
    
    def aggregate_by_taxonomy(self) -> dict[str, dict[str, Any]]:
        """
        Aggregate token usage by full taxonomy key.
        
        Returns dict mapping taxonomy_key -> aggregated stats.
        This is the preferred method for learning and reporting.
        """
        result: dict[str, dict[str, Any]] = {}
        
        for t in self._transactions:
            key = t.taxonomy_key()
            if key not in result:
                result[key] = {
                    "module": t.module,
                    "op_kind": t.op_kind,
                    "tool_family": t.tool_family,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                }
            result[key]["input_tokens"] += t.input_tokens
            result[key]["output_tokens"] += t.output_tokens
            result[key]["total_tokens"] += t.total_tokens
            result[key]["call_count"] += 1
        
        return result
    
    def aggregate_by_module(self) -> dict[str, dict[str, Any]]:
        """
        Aggregate token usage by module only.
        
        Returns dict mapping module_name -> aggregated stats.
        Useful for high-level budget tracking and reporting.
        """
        result: dict[str, dict[str, Any]] = {}
        
        for t in self._transactions:
            key = t.module or "unknown"
            if key not in result:
                result[key] = {
                    "module": key,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                }
            result[key]["input_tokens"] += t.input_tokens
            result[key]["output_tokens"] += t.output_tokens
            result[key]["total_tokens"] += t.total_tokens
            result[key]["call_count"] += 1
        
        return result
    
    def aggregate_by_tool_family(self) -> dict[str, dict[str, Any]]:
        """
        Aggregate token usage by tool family.
        
        Returns dict mapping tool_family -> aggregated stats.
        Useful for understanding which tool categories cost most.
        """
        result: dict[str, dict[str, Any]] = {}
        
        for t in self._transactions:
            key = t.tool_family or "llm"  # Default to llm for agent reasoning
            if key not in result:
                result[key] = {
                    "tool_family": key,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                }
            result[key]["input_tokens"] += t.input_tokens
            result[key]["output_tokens"] += t.output_tokens
            result[key]["total_tokens"] += t.total_tokens
            result[key]["call_count"] += 1
        
        return result
    
    # =========================================================================
    # Per-Tool Tracking
    # =========================================================================
    
    def record_tool_usage(
        self,
        tool_name: str,
        action: str,
        output_chars: int,
        injected_chars: Optional[int] = None,
        success: bool = True,
    ) -> None:
        """
        Record a tool invocation for tracking.
        
        This tracks tool output sizes which become input tokens on subsequent
        LLM calls. Helps identify which tools contribute most to token usage.
        
        Args:
            tool_name: Name of the tool (e.g., "terminal", "filesystem")
            action: The action performed (e.g., "execute", "read")
            output_chars: Size of the raw tool output (chars)
            injected_chars: Approximate chars injected back into the LLM (if different)
            success: Whether the tool call succeeded
        """
        # Use tool_name.action as the key for granular tracking
        key = f"{tool_name}.{action}"
        
        if key not in self._tool_usage:
            self._tool_usage[key] = ToolUsageStats(tool_name=key)
        
        injected = injected_chars if injected_chars is not None else output_chars
        self._tool_usage[key].record_call(injected, success, raw_output_chars=output_chars)
    
    def get_tool_usage(self) -> dict[str, ToolUsageStats]:
        """Get all tool usage statistics."""
        return dict(self._tool_usage)
    
    def get_tool_usage_summary(self) -> list[dict[str, Any]]:
        """
        Get sorted tool usage summary for reporting.
        
        Returns list of tool stats sorted by estimated token consumption.
        """
        if not self._tool_usage:
            return []
        
        summary = []
        for key, stats in self._tool_usage.items():
            summary.append({
                "tool": key,
                "calls": stats.call_count,
                "total_chars": stats.total_output_chars,
                "total_raw_chars": stats.total_raw_output_chars,
                "estimated_tokens": stats.estimated_tokens,
                "mean_chars": int(stats.mean_output_chars),
                "max_chars": stats.max_output_chars,
                "max_raw_chars": stats.max_raw_output_chars,
                "errors": stats.error_count,
            })
        
        # Sort by estimated tokens (highest first)
        summary.sort(key=lambda x: x["estimated_tokens"], reverse=True)
        return summary
    
    def get_tool_context_for_agents(self) -> str:
        """
        Get formatted tool usage context for agent system prompts.
        
        This helps agents understand which tools are expensive and
        make informed decisions about tool usage.
        """
        summary = self.get_tool_usage_summary()
        if not summary:
            return ""
        
        total_tool_tokens = sum(s["estimated_tokens"] for s in summary)
        
        lines = ["## Tool Usage This Session"]
        lines.append("")
        lines.append(f"**Total tool output tokens (est.):** {total_tool_tokens:,}")
        lines.append("")
        
        if total_tool_tokens > 0:
            lines.append("| Tool | Calls | Est. Tokens | Mean Output | Max Output |")
            lines.append("|------|-------|-------------|-------------|------------|")
            
            for s in summary[:10]:  # Top 10 tools
                pct = (s["estimated_tokens"] / total_tool_tokens * 100) if total_tool_tokens > 0 else 0
                lines.append(
                    f"| {s['tool']} | {s['calls']} | {s['estimated_tokens']:,} ({pct:.0f}%) | "
                    f"{s['mean_chars']:,} chars | {s['max_chars']:,} chars |"
                )
            
            lines.append("")
            
            # Add guidance if terminal is high
            terminal_tokens = sum(s["estimated_tokens"] for s in summary if "terminal" in s["tool"])
            if terminal_tokens > total_tool_tokens * 0.3:
                lines.append("⚠️ **Terminal output is a major token consumer.** Consider:")
                lines.append("- Using `head`, `tail`, or `grep` to limit output")
                lines.append("- Redirecting verbose output to files")
                lines.append("- Using `wc -l` to check output size before displaying")
                lines.append("")
        
        return "\n".join(lines)
    
    def format_status(self) -> str:
        """Format a human-readable status string."""
        if self._budget is None:
            return f"Tokens used: {self.total_used:,} (no budget set)"
        
        return (
            f"Tokens: {self.total_used:,} / {self._budget:,} "
            f"({self.percentage_used:.1f}% used, {self.remaining:,} remaining)"
        )
    
    def format_balance_brief(self) -> str:
        """Format brief balance for inline display."""
        if self._budget is None:
            return f"({self.total_used:,} tokens)"
        return f"({self.remaining:,} tokens remaining)"


# Module-level singleton accessor
_account: Optional[TokenAccount] = None


def get_token_account() -> TokenAccount:
    """Get the global TokenAccount singleton."""
    global _account
    if _account is None:
        _account = TokenAccount()
    return _account
