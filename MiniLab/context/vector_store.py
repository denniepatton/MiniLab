"""
Vector Store for RAG retrieval.

Uses FAISS for efficient similarity search with semantic + recency scoring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import numpy as np


@dataclass
class Document:
    """A document in the vector store."""
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Document:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class SearchResult:
    """Result from a vector search."""
    document: Document
    score: float  # Combined semantic + recency score
    semantic_score: float
    recency_score: float


class VectorStore:
    """
    Vector store with FAISS backend.
    
    Features:
    - Fast similarity search
    - Semantic + recency biased scoring
    - Persistence to disk
    - Metadata filtering
    """
    
    def __init__(
        self,
        embedding_dim: int,
        persist_dir: Optional[Path] = None,
        recency_weight: float = 0.2,  # Weight for recency in final score
        recency_half_life: float = 86400.0,  # Half-life in seconds (1 day)
    ):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            persist_dir: Directory for persistence
            recency_weight: Weight for recency score (0-1)
            recency_half_life: Time in seconds for recency score to halve
        """
        self.embedding_dim = embedding_dim
        self.persist_dir = persist_dir
        self.recency_weight = recency_weight
        self.recency_half_life = recency_half_life
        
        self._index = None
        self._documents: dict[str, Document] = {}
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        
        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._load()
    
    @property
    def index(self):
        """Lazy initialize FAISS index."""
        if self._index is None:
            try:
                import faiss
                self._index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine sim
            except ImportError:
                raise ImportError(
                    "faiss not installed. "
                    "Run: pip install faiss-cpu"
                )
        return self._index
    
    def _load(self) -> None:
        """Load store from disk."""
        if not self.persist_dir:
            return
        
        docs_file = self.persist_dir / "documents.json"
        index_file = self.persist_dir / "index.faiss"
        
        if docs_file.exists():
            with open(docs_file) as f:
                docs_data = json.load(f)
            
            for doc_data in docs_data:
                doc = Document.from_dict(doc_data)
                self._documents[doc.id] = doc
        
        if index_file.exists() and self._documents:
            try:
                import faiss
                self._index = faiss.read_index(str(index_file))
                
                # Rebuild mappings
                for idx, doc_id in enumerate(self._documents.keys()):
                    self._id_to_idx[doc_id] = idx
                    self._idx_to_id[idx] = doc_id
            except Exception:
                # Index corrupted, will rebuild
                self._index = None
    
    def _save(self) -> None:
        """Save store to disk."""
        if not self.persist_dir:
            return
        
        # Save documents
        docs_file = self.persist_dir / "documents.json"
        docs_data = [doc.to_dict() for doc in self._documents.values()]
        with open(docs_file, "w") as f:
            json.dump(docs_data, f, indent=2)
        
        # Save index
        if self._index is not None and self._index.ntotal > 0:
            import faiss
            index_file = self.persist_dir / "index.faiss"
            faiss.write_index(self._index, str(index_file))
    
    def add(self, document: Document) -> None:
        """
        Add a document to the store.
        
        Args:
            document: Document with embedding to add
        """
        if document.embedding is None:
            raise ValueError("Document must have an embedding")
        
        # Remove existing if updating
        if document.id in self._documents:
            self.remove(document.id)
        
        # Normalize embedding for cosine similarity
        embedding = document.embedding / np.linalg.norm(document.embedding)
        embedding = embedding.astype(np.float32).reshape(1, -1)
        
        # Add to index
        idx = self.index.ntotal
        self.index.add(embedding)
        
        # Update mappings
        self._documents[document.id] = document
        self._id_to_idx[document.id] = idx
        self._idx_to_id[idx] = document.id
        
        # Auto-save periodically
        if len(self._documents) % 50 == 0:
            self._save()
    
    def add_batch(self, documents: list[Document]) -> None:
        """
        Add multiple documents to the store.
        
        Args:
            documents: List of documents with embeddings
        """
        for doc in documents:
            self.add(doc)
        self._save()
    
    def remove(self, doc_id: str) -> bool:
        """
        Remove a document from the store.
        
        Note: FAISS doesn't support efficient removal, so we mark as removed
        and rebuild periodically.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if document was removed
        """
        if doc_id not in self._documents:
            return False
        
        del self._documents[doc_id]
        # Note: Index will be stale, rebuild on next search if needed
        return True
    
    def get(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self._documents.get(doc_id)
    
    def _compute_recency_score(self, timestamp: float) -> float:
        """
        Compute recency score with exponential decay.
        
        Args:
            timestamp: Document timestamp
            
        Returns:
            Recency score (0 to 1)
        """
        age = time.time() - timestamp
        if age < 0:
            age = 0
        
        # Exponential decay with half-life
        return np.exp(-0.693 * age / self.recency_half_life)
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Only return docs matching this metadata
            min_score: Minimum combined score threshold
            
        Returns:
            List of SearchResults sorted by combined score
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search more than k to account for filtering
        search_k = min(k * 3, self.index.ntotal)
        
        distances, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            doc_id = self._idx_to_id.get(idx)
            if doc_id is None or doc_id not in self._documents:
                continue
            
            doc = self._documents[doc_id]
            
            # Apply metadata filter
            if filter_metadata:
                match = True
                for key, value in filter_metadata.items():
                    if doc.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Compute scores
            semantic_score = float(dist)  # Already normalized, so this is cosine sim
            recency_score = self._compute_recency_score(doc.timestamp)
            
            # Combined score with weighting
            combined_score = (
                (1 - self.recency_weight) * semantic_score +
                self.recency_weight * recency_score
            )
            
            if combined_score >= min_score:
                results.append(SearchResult(
                    document=doc,
                    score=combined_score,
                    semantic_score=semantic_score,
                    recency_score=recency_score,
                ))
        
        # Sort by combined score and return top k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
    
    def search_by_metadata(
        self,
        metadata: dict[str, Any],
        limit: int = 100,
    ) -> list[Document]:
        """
        Search documents by metadata only.
        
        Args:
            metadata: Metadata to match
            limit: Maximum results to return
            
        Returns:
            List of matching documents
        """
        results = []
        for doc in self._documents.values():
            match = True
            for key, value in metadata.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                results.append(doc)
                if len(results) >= limit:
                    break
        return results
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        self._documents.clear()
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._index = None
        
        if self.persist_dir:
            for f in self.persist_dir.glob("*"):
                f.unlink()
    
    def save(self) -> None:
        """Explicitly save to disk."""
        self._save()
    
    def __len__(self) -> int:
        return len(self._documents)
