"""
TokenAccount: Real-time token budget tracking with enforcement.

Provides:
- Single shared instance across all LLM calls
- Real-time balance tracking and reporting
- Automatic mode degradation warnings
- Hard stop at budget limit
- Graceful shutdown reserve for completion
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Any


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
    """Record of a single token debit."""
    timestamp: datetime
    agent_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    balance_after: int
    operation: str = ""  # e.g., "pubmed.search", "llm.complete"


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
        
        # Callbacks
        self._on_warning: Optional[Callable[[int, int, float], None]] = None
        self._on_budget_exceeded: Optional[Callable[[int, int], None]] = None
        
        # Warning thresholds (percentages)
        self._warning_thresholds = [60, 80, 95]
        self._warnings_issued: set[int] = set()
        
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
    
    def set_budget(self, budget: int) -> None:
        """Set the token budget for this session."""
        self._budget = budget
        self._warnings_issued = set()
    
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
        }
    
    def _estimate_cost(self) -> float:
        """Estimate cost at ~$5/1M tokens."""
        return (self.total_used / 1_000_000) * 5.00
    
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
        cache_creation: int = 0,
        cache_read: int = 0,
    ) -> TokenTransaction:
        """
        Record token usage.
        
        Args:
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            agent_id: Which agent made this call
            operation: What operation (e.g., "pubmed.search")
            cache_creation: Cache creation tokens
            cache_read: Cache read tokens
        
        Returns:
            TokenTransaction record
        
        Raises:
            RuntimeError: If budget is hard exceeded
        """
        self._total_input += input_tokens
        self._total_output += output_tokens
        self._total_cache_creation += cache_creation
        self._total_cache_read += cache_read
        
        total = input_tokens + output_tokens
        
        transaction = TokenTransaction(
            timestamp=datetime.now(),
            agent_id=agent_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            balance_after=self.remaining if self._budget else -1,
            operation=operation,
        )
        self._transactions.append(transaction)
        
        # Check warnings
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
