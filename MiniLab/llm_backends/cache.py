"""
LLM Response Cache: Hash-based caching for LLM responses.

Provides:
- Content-addressable caching using SHA256 hashes
- SQLite storage for persistence and efficient lookup
- TTL-based expiration with configurable duration (default 1 hour)
- Token savings tracking for efficiency metrics
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import threading


@dataclass
class CacheEntry:
    """A cached LLM response."""
    cache_key: str
    response: str
    model: str
    input_tokens: int
    output_tokens: int
    created_at: float
    expires_at: float
    hit_count: int = 0


class LLMCache:
    """
    SQLite-backed LLM response cache.
    
    Caches LLM responses by hashing the full request (system prompt + messages).
    Uses SQLite for persistence and efficient key lookup.
    
    Usage:
        cache = LLMCache()
        
        # Check cache before API call
        key = cache.make_key(messages, model)
        cached = cache.get(key)
        if cached:
            return cached.response
        
        # After API call, store result
        cache.set(key, response, model, input_tokens, output_tokens)
    """
    
    _instance: Optional[LLMCache] = None
    _lock = threading.Lock()
    
    # Default TTL: 1 hour (aligned with Anthropic prompt cache TTL for agentic workflows)
    # Tasks typically take >5 min, so 1h TTL provides good value
    DEFAULT_TTL = 60 * 60  # 1 hour
    
    def __init__(
        self,
        cache_path: Optional[Path] = None,
        ttl: int = DEFAULT_TTL,
        max_entries: int = 10000,
    ):
        if cache_path is None:
            cache_path = Path.home() / ".minilab" / "llm_cache.db"
        
        self.cache_path = cache_path
        self.ttl = ttl
        self.max_entries = max_entries
        
        # Stats tracking
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        
        self._init_db()
    
    @classmethod
    def get_instance(cls, cache_path: Optional[Path] = None) -> LLMCache:
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cache_path)
            return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None
    
    def _init_db(self) -> None:
        """Initialize SQLite database."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
            conn.commit()
    
    def make_key(
        self,
        messages: List[dict],
        model: str,
        temperature: float = 0.2,
    ) -> str:
        """
        Generate cache key from request parameters.
        
        Uses SHA256 hash of normalized content for deterministic keys.
        """
        # Normalize messages to consistent JSON
        normalized = json.dumps({
            "messages": messages,
            "model": model,
            "temperature": round(temperature, 2),
        }, sort_keys=True)
        
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Get cached response if valid.
        
        Returns None if not found or expired.
        """
        now = time.time()
        
        with sqlite3.connect(self.cache_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM cache WHERE cache_key = ? AND expires_at > ?",
                (key, now)
            ).fetchone()
            
            if row:
                # Update hit count
                conn.execute(
                    "UPDATE cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                    (key,)
                )
                conn.commit()
                
                self._hits += 1
                self._tokens_saved += row["input_tokens"] + row["output_tokens"]
                
                return CacheEntry(
                    cache_key=row["cache_key"],
                    response=row["response"],
                    model=row["model"],
                    input_tokens=row["input_tokens"],
                    output_tokens=row["output_tokens"],
                    created_at=row["created_at"],
                    expires_at=row["expires_at"],
                    hit_count=row["hit_count"] + 1,
                )
        
        self._misses += 1
        return None
    
    def set(
        self,
        key: str,
        response: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        ttl: Optional[int] = None,
    ) -> None:
        """Store response in cache."""
        now = time.time()
        expires = now + (ttl or self.ttl)
        
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache 
                (cache_key, response, model, input_tokens, output_tokens, created_at, expires_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """, (key, response, model, input_tokens, output_tokens, now, expires))
            conn.commit()
        
        # Periodically cleanup old entries
        if self._misses % 100 == 0:
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Remove expired entries and enforce max size."""
        now = time.time()
        
        with sqlite3.connect(self.cache_path) as conn:
            # Remove expired
            conn.execute("DELETE FROM cache WHERE expires_at < ?", (now,))
            
            # Enforce max entries (keep most recently accessed)
            count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            if count > self.max_entries:
                excess = count - self.max_entries
                conn.execute("""
                    DELETE FROM cache WHERE cache_key IN (
                        SELECT cache_key FROM cache 
                        ORDER BY created_at ASC 
                        LIMIT ?
                    )
                """, (excess,))
            
            conn.commit()
    
    def invalidate(self, key: str) -> bool:
        """Remove specific entry from cache."""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE cache_key = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0
    
    def clear(self) -> int:
        """Clear all cached entries. Returns count deleted."""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute("DELETE FROM cache")
            conn.commit()
            return cursor.rowcount
    
    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        with sqlite3.connect(self.cache_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            total_hits = conn.execute("SELECT SUM(hit_count) FROM cache").fetchone()[0] or 0
        
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "entries": count,
            "session_hits": self._hits,
            "session_misses": self._misses,
            "session_hit_rate": f"{hit_rate:.1f}%",
            "all_time_hits": total_hits,
            "tokens_saved_session": self._tokens_saved,
        }


def get_llm_cache() -> LLMCache:
    """Get the global LLMCache singleton."""
    return LLMCache.get_instance()
