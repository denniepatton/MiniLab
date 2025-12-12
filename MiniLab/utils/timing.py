"""
Timing utilities for MiniLab performance monitoring.

Provides optional instrumentation for measuring operation durations.
Enable via environment variable: MINILAB_TIMING=1
"""

from __future__ import annotations

import asyncio
import functools
import os
import sys
import time
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, TypeVar
import threading

# ParamSpec is available in Python 3.10+, use typing_extensions for older versions
if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

P = ParamSpec('P')
T = TypeVar('T')


def _timing_enabled() -> bool:
    """Check if timing is enabled via environment variable."""
    return os.environ.get("MINILAB_TIMING", "0").lower() in ("1", "true", "yes", "on")


@dataclass
class TimingRecord:
    """Record of a single timed operation."""
    operation: str
    category: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "category": self.category,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class TimingCollector:
    """
    Collects and reports timing data for MiniLab operations.
    
    Thread-safe singleton for collecting timing records across
    all components.
    
    Categories:
        - llm: LLM API calls
        - embedding: Embedding generation
        - tool: Tool execution
        - workflow: Workflow phases
        - io: File I/O operations
    """
    
    _instance: Optional["TimingCollector"] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._records: list[TimingRecord] = []
        self._enabled = _timing_enabled()
        self._record_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> "TimingCollector":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton (clears all records)."""
        with cls._lock:
            cls._instance = None
    
    @property
    def enabled(self) -> bool:
        """Check if timing collection is enabled."""
        return self._enabled
    
    def enable(self) -> None:
        """Enable timing collection."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable timing collection."""
        self._enabled = False
    
    def record(
        self,
        operation: str,
        category: str,
        duration_ms: float,
        **metadata: Any,
    ) -> None:
        """
        Record a timing measurement.
        
        Args:
            operation: Name of the operation (e.g., "acomplete", "embed")
            category: Category (e.g., "llm", "embedding", "tool")
            duration_ms: Duration in milliseconds
            **metadata: Additional metadata (model, tool_name, etc.)
        """
        if not self._enabled:
            return
        
        record = TimingRecord(
            operation=operation,
            category=category,
            duration_ms=duration_ms,
            metadata=metadata,
        )
        
        with self._record_lock:
            self._records.append(record)
        
        # Print if verbose timing
        if os.environ.get("MINILAB_TIMING_VERBOSE", "0").lower() in ("1", "true"):
            print(f"  ⏱ [{category}] {operation}: {duration_ms:.1f}ms")
    
    def get_records(
        self,
        category: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> list[TimingRecord]:
        """Get timing records, optionally filtered."""
        with self._record_lock:
            records = self._records.copy()
        
        if category:
            records = [r for r in records if r.category == category]
        if operation:
            records = [r for r in records if r.operation == operation]
        
        return records
    
    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics for all timing records.
        
        Returns:
            Dict with statistics by category and operation.
        """
        records = self.get_records()
        
        if not records:
            return {"total_records": 0}
        
        # Group by category
        by_category: dict[str, list[float]] = {}
        by_operation: dict[str, list[float]] = {}
        
        for r in records:
            by_category.setdefault(r.category, []).append(r.duration_ms)
            op_key = f"{r.category}.{r.operation}"
            by_operation.setdefault(op_key, []).append(r.duration_ms)
        
        def stats(durations: list[float]) -> dict:
            if not durations:
                return {}
            return {
                "count": len(durations),
                "total_ms": round(sum(durations), 2),
                "avg_ms": round(sum(durations) / len(durations), 2),
                "min_ms": round(min(durations), 2),
                "max_ms": round(max(durations), 2),
            }
        
        return {
            "total_records": len(records),
            "total_time_ms": round(sum(r.duration_ms for r in records), 2),
            "by_category": {cat: stats(durs) for cat, durs in by_category.items()},
            "by_operation": {op: stats(durs) for op, durs in by_operation.items()},
        }
    
    def print_summary(self) -> None:
        """Print a formatted timing summary."""
        summary = self.get_summary()
        
        if summary["total_records"] == 0:
            print("\n⏱ No timing records collected.")
            return
        
        print("\n" + "=" * 60)
        print("  ⏱  TIMING SUMMARY")
        print("=" * 60)
        print(f"  Total operations: {summary['total_records']}")
        print(f"  Total time: {summary['total_time_ms']:.1f}ms ({summary['total_time_ms']/1000:.2f}s)")
        
        print("\n  By Category:")
        for cat, stats in summary.get("by_category", {}).items():
            print(f"    {cat}: {stats['count']} ops, {stats['total_ms']:.1f}ms total, {stats['avg_ms']:.1f}ms avg")
        
        print("\n  Slowest Operations:")
        by_op = summary.get("by_operation", {})
        slowest = sorted(by_op.items(), key=lambda x: x[1].get("avg_ms", 0), reverse=True)[:5]
        for op, stats in slowest:
            print(f"    {op}: {stats['avg_ms']:.1f}ms avg ({stats['count']} calls)")
        
        print("=" * 60 + "\n")
    
    def clear(self) -> None:
        """Clear all timing records."""
        with self._record_lock:
            self._records.clear()


# Convenience accessor
timing = TimingCollector.get_instance


@contextmanager
def timed_operation(operation: str, category: str, **metadata):
    """
    Context manager for timing a synchronous operation.
    
    Usage:
        with timed_operation("read_file", "io", path="/foo/bar"):
            content = file.read()
    """
    collector = timing()
    if not collector.enabled:
        yield
        return
    
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        collector.record(operation, category, duration_ms, **metadata)


@asynccontextmanager
async def async_timed_operation(operation: str, category: str, **metadata):
    """
    Async context manager for timing an async operation.
    
    Usage:
        async with async_timed_operation("acomplete", "llm", model="claude"):
            response = await llm.acomplete(messages)
    """
    collector = timing()
    if not collector.enabled:
        yield
        return
    
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        collector.record(operation, category, duration_ms, **metadata)


def timed(category: str, operation: Optional[str] = None):
    """
    Decorator for timing function calls.
    
    Usage:
        @timed("llm")
        async def acomplete(self, messages):
            ...
        
        @timed("tool", "execute")
        async def execute(self, action, params):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        op_name = operation or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                collector = timing()
                if not collector.enabled:
                    return await func(*args, **kwargs)
                
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    collector.record(op_name, category, duration_ms)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                collector = timing()
                if not collector.enabled:
                    return func(*args, **kwargs)
                
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    collector.record(op_name, category, duration_ms)
            return sync_wrapper
    
    return decorator


class TimingContext:
    """
    High-level timing context for a workflow or session.
    
    Usage:
        ctx = TimingContext("analysis_session")
        ctx.start()
        
        # ... do work ...
        
        ctx.stop()
        ctx.print_summary()
    """
    
    def __init__(self, name: str):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._collector = timing()
        self._start_record_count = 0
    
    def start(self) -> None:
        """Start timing context."""
        self.start_time = time.perf_counter()
        self._start_record_count = len(self._collector.get_records())
    
    def stop(self) -> None:
        """Stop timing context."""
        self.end_time = time.perf_counter()
    
    @property
    def duration_s(self) -> float:
        """Get duration in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.perf_counter()
        return end - self.start_time
    
    def get_records(self) -> list[TimingRecord]:
        """Get timing records collected during this context."""
        all_records = self._collector.get_records()
        return all_records[self._start_record_count:]
    
    def print_summary(self) -> None:
        """Print summary for this context."""
        if not self._collector.enabled:
            return
        
        records = self.get_records()
        
        print(f"\n⏱ Timing: {self.name}")
        print(f"  Wall time: {self.duration_s:.2f}s")
        print(f"  Operations: {len(records)}")
        
        if records:
            total_op_time = sum(r.duration_ms for r in records) / 1000
            print(f"  Operation time: {total_op_time:.2f}s")
            
            # Breakdown by category
            by_cat: dict[str, float] = {}
            for r in records:
                by_cat[r.category] = by_cat.get(r.category, 0) + r.duration_ms
            
            for cat, ms in sorted(by_cat.items(), key=lambda x: -x[1]):
                print(f"    {cat}: {ms:.1f}ms")


def enable_timing() -> None:
    """Enable timing collection."""
    timing().enable()


def disable_timing() -> None:
    """Disable timing collection."""
    timing().disable()


def print_timing_summary() -> None:
    """Print timing summary if timing was enabled."""
    timing().print_summary()
