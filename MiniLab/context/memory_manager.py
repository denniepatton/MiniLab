"""
MemoryManager: Context compaction and rolling memory.

Provides:
- Automatic context window management
- Rolling summarization of conversation history
- Artifact-based memory (persist outputs, forget intermediate states)
- Compression strategies for different content types
"""

from __future__ import annotations

__all__ = [
    "ContentType",
    "CompressionStrategy",
    "MemoryConfig",
    "MemoryEntry",
    "MemoryCompressor",
    "SimpleCompressor",
    "MemoryManager",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import json
import hashlib

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Types of content in memory."""
    CONVERSATION = "conversation"
    TOOL_OUTPUT = "tool_output"
    CODE = "code"
    DOCUMENT = "document"
    ARTIFACT = "artifact"
    SUMMARY = "summary"


class CompressionStrategy(str, Enum):
    """Strategies for compressing content."""
    NONE = "none"
    TRUNCATE = "truncate"
    SUMMARIZE = "summarize"
    SAMPLE = "sample"  # Keep representative samples
    REFERENCE = "reference"  # Replace with reference to stored artifact


@dataclass
class MemoryEntry:
    """A single entry in memory."""

    id: str
    content: str
    content_type: ContentType
    timestamp: datetime

    # Token tracking
    original_tokens: int = 0
    current_tokens: int = 0

    # Compression state
    compressed: bool = False
    compression_strategy: CompressionStrategy = CompressionStrategy.NONE

    # Importance scoring (0-1)
    importance: float = 0.5

    # Source tracking
    source: str = ""  # e.g., agent name, tool name
    task_id: Optional[str] = None

    # Reference to stored artifact (if compressed to reference)
    artifact_path: Optional[str] = None

    def to_context(self) -> str:
        """Render entry for context."""
        if self.artifact_path:
            return f"[Reference: {self.artifact_path}]"
        return self.content


class MemoryConfig(BaseModel):
    """Configuration for memory management."""

    # Target sizes (in tokens)
    target_context_tokens: int = Field(default=8000, description="Target context window")
    max_conversation_tokens: int = Field(default=2000, description="Max for conversation history")
    max_tool_output_tokens: int = Field(default=1000, description="Max per tool output")

    # Rolling window sizes
    conversation_window_size: int = Field(default=10, description="Keep last N messages")
    tool_output_window_size: int = Field(default=5, description="Keep last N tool outputs")

    # Compression thresholds
    compress_after_tokens: int = Field(default=500, description="Compress entries > this")
    summarize_threshold: float = Field(default=0.3, description="Summarize when importance < this")

    # Artifact settings
    artifact_dir: Optional[Path] = Field(default=None, description="Directory for stored artifacts")

    model_config = {"extra": "forbid"}


class MemoryCompressor(ABC):
    """Base class for content compressors."""

    @abstractmethod
    def compress(
        self,
        content: str,
        target_tokens: int,
        content_type: ContentType,
    ) -> tuple[str, CompressionStrategy]:
        """
        Compress content to target size.
        
        Args:
            content: Original content
            target_tokens: Target token count
            content_type: Type of content
            
        Returns:
            Tuple of (compressed content, strategy used)
        """
        pass


class SimpleCompressor(MemoryCompressor):
    """Simple truncation-based compressor."""

    # Rough estimate: 1 token ≈ 4 chars
    CHARS_PER_TOKEN = 4

    def compress(
        self,
        content: str,
        target_tokens: int,
        content_type: ContentType,
    ) -> tuple[str, CompressionStrategy]:
        """Compress by truncation."""
        target_chars = target_tokens * self.CHARS_PER_TOKEN

        if len(content) <= target_chars:
            return content, CompressionStrategy.NONE

        # For code, keep beginning and end
        if content_type == ContentType.CODE:
            half = target_chars // 2 - 30
            return (
                content[:half] +
                "\n# ... [truncated] ...\n" +
                content[-half:]
            ), CompressionStrategy.TRUNCATE

        # For tool outputs, keep beginning
        if content_type == ContentType.TOOL_OUTPUT:
            return (
                content[:target_chars - 30] +
                "\n[truncated...]"
            ), CompressionStrategy.TRUNCATE

        # For conversation, summarize would be ideal but fallback to truncate
        return (
            content[:target_chars - 30] +
            "\n[truncated...]"
        ), CompressionStrategy.TRUNCATE


class MemoryManager:
    """
    Manages context memory with automatic compaction.
    
    Features:
    - Rolling conversation history
    - Automatic compression of large entries
    - Artifact-based persistence for important outputs
    - Importance-based eviction
    """

    def __init__(
        self,
        config: MemoryConfig,
        compressor: Optional[MemoryCompressor] = None,
    ):
        """
        Initialize memory manager.
        
        Args:
            config: Memory configuration
            compressor: Optional custom compressor
        """
        self.config = config
        self.compressor = compressor or SimpleCompressor()

        # Memory storage by type
        self._entries: dict[str, MemoryEntry] = {}
        self._by_type: dict[ContentType, list[str]] = {ct: [] for ct in ContentType}

        # Total token tracking
        self._total_tokens: int = 0

        # Artifact storage
        if config.artifact_dir:
            config.artifact_dir.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        content: str,
        content_type: ContentType,
        source: str = "",
        task_id: Optional[str] = None,
        importance: float = 0.5,
    ) -> MemoryEntry:
        """
        Add an entry to memory.
        
        Args:
            content: Content to store
            content_type: Type of content
            source: Source of content (agent, tool, etc.)
            task_id: Optional task association
            importance: Importance score (0-1)
            
        Returns:
            Created MemoryEntry
        """
        # Generate ID
        entry_id = hashlib.sha256(
            f"{content_type}:{source}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Estimate tokens
        original_tokens = len(content) // 4

        # Create entry
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            content_type=content_type,
            timestamp=datetime.now(),
            original_tokens=original_tokens,
            current_tokens=original_tokens,
            importance=importance,
            source=source,
            task_id=task_id,
        )

        # Store
        self._entries[entry_id] = entry
        self._by_type[content_type].append(entry_id)
        self._total_tokens += original_tokens

        # Trigger compaction if needed
        if self._total_tokens > self.config.target_context_tokens:
            self._compact()

        return entry

    def get_context(
        self,
        max_tokens: Optional[int] = None,
        include_types: Optional[list[ContentType]] = None,
        task_id: Optional[str] = None,
    ) -> list[MemoryEntry]:
        """
        Get memory entries for context.
        
        Args:
            max_tokens: Maximum tokens to return
            include_types: Types to include (all if None)
            task_id: Filter by task
            
        Returns:
            List of memory entries
        """
        max_tokens = max_tokens or self.config.target_context_tokens
        include_types = include_types or list(ContentType)

        # Collect eligible entries
        eligible = []
        for entry_id, entry in self._entries.items():
            if entry.content_type not in include_types:
                continue
            if task_id and entry.task_id != task_id:
                continue
            eligible.append(entry)

        # Sort by timestamp (most recent first) and importance
        eligible.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)

        # Select within budget
        result = []
        total = 0
        for entry in eligible:
            if total + entry.current_tokens <= max_tokens:
                result.append(entry)
                total += entry.current_tokens

        return result

    def _compact(self) -> None:
        """Compact memory to fit within target."""
        target = self.config.target_context_tokens

        if self._total_tokens <= target:
            return

        # Get all entries sorted by eviction priority
        # (low importance, old, already compressed)
        entries = list(self._entries.values())
        entries.sort(key=lambda e: (
            e.importance,
            1 if not e.compressed else 0,
            e.timestamp.timestamp(),
        ))

        # Compress/evict until under target
        for entry in entries:
            if self._total_tokens <= target:
                break

            # Try compression first
            if not entry.compressed:
                old_tokens = entry.current_tokens

                if entry.importance < self.config.summarize_threshold:
                    # Low importance - aggressive compression
                    target_tokens = entry.current_tokens // 4
                else:
                    # Normal compression
                    target_tokens = min(
                        entry.current_tokens // 2,
                        self.config.compress_after_tokens
                    )

                compressed, strategy = self.compressor.compress(
                    entry.content,
                    target_tokens,
                    entry.content_type,
                )

                entry.content = compressed
                entry.current_tokens = len(compressed) // 4
                entry.compressed = True
                entry.compression_strategy = strategy

                self._total_tokens -= (old_tokens - entry.current_tokens)

            # If still over budget and entry is low importance, evict
            if self._total_tokens > target and entry.importance < 0.2:
                self._evict(entry.id)

    def _evict(self, entry_id: str) -> None:
        """Evict an entry from memory."""
        if entry_id not in self._entries:
            return

        entry = self._entries[entry_id]

        # Store as artifact if important enough
        if entry.importance >= 0.5 and self.config.artifact_dir:
            artifact_path = self._store_artifact(entry)
            # Keep a reference entry
            entry.content = ""
            entry.artifact_path = str(artifact_path)
            entry.current_tokens = 10  # Small reference
            self._total_tokens -= entry.original_tokens - 10
        else:
            # Full eviction
            self._total_tokens -= entry.current_tokens
            del self._entries[entry_id]
            self._by_type[entry.content_type].remove(entry_id)

    def _store_artifact(self, entry: MemoryEntry) -> Path:
        """Store entry as artifact on disk."""
        if not self.config.artifact_dir:
            raise RuntimeError("No artifact directory configured")

        artifact_path = self.config.artifact_dir / f"{entry.id}.json"

        data = {
            "id": entry.id,
            "content": entry.content,
            "content_type": entry.content_type.value,
            "timestamp": entry.timestamp.isoformat(),
            "source": entry.source,
            "task_id": entry.task_id,
            "original_tokens": entry.original_tokens,
        }

        with open(artifact_path, "w") as f:
            json.dump(data, f, indent=2)

        return artifact_path

    def load_artifact(self, artifact_path: str) -> Optional[str]:
        """Load artifact content from disk."""
        path = Path(artifact_path)
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return data.get("content")

    def get_conversation_history(
        self,
        max_messages: Optional[int] = None
    ) -> list[MemoryEntry]:
        """Get recent conversation history."""
        max_messages = max_messages or self.config.conversation_window_size

        conv_ids = self._by_type[ContentType.CONVERSATION]
        entries = [self._entries[eid] for eid in conv_ids if eid in self._entries]

        # Sort by timestamp
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        return entries[:max_messages]

    def clear_task(self, task_id: str) -> None:
        """Clear memory entries for a completed task."""
        to_remove = [
            eid for eid, entry in self._entries.items()
            if entry.task_id == task_id and entry.importance < 0.5
        ]

        for entry_id in to_remove:
            self._evict(entry_id)

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        by_type_counts = {
            ct.value: len(ids) for ct, ids in self._by_type.items()
        }
        by_type_tokens = {}
        for ct, ids in self._by_type.items():
            tokens = sum(
                self._entries[eid].current_tokens
                for eid in ids
                if eid in self._entries
            )
            by_type_tokens[ct.value] = tokens

        compressed_count = sum(
            1 for e in self._entries.values() if e.compressed
        )

        return {
            "total_entries": len(self._entries),
            "total_tokens": self._total_tokens,
            "target_tokens": self.config.target_context_tokens,
            "by_type_counts": by_type_counts,
            "by_type_tokens": by_type_tokens,
            "compressed_count": compressed_count,
            "usage_pct": (self._total_tokens / self.config.target_context_tokens) * 100,
        }
