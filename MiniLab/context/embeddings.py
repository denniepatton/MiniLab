"""
Embedding Manager for RAG system.

Uses sentence-transformers for local embeddings.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional
import numpy as np

# Workaround for pyarrow compatibility issues with newer versions
# https://github.com/huggingface/datasets/issues/6444
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class EmbeddingManager:
    """
    Manages text embeddings using sentence-transformers.
    
    Features:
    - Local embedding generation (no API calls)
    - Embedding cache for efficiency
    - Batch processing support
    """
    
    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Good balance of speed/quality, ~90MB
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize embedding manager.
        
        Args:
            model_name: sentence-transformers model to use
            cache_dir: Directory for caching embeddings
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
    
    @property
    def model(self):
        """Lazy load the model with minimal console output."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import logging
                import warnings
                
                # Suppress verbose logging from transformers/sentence-transformers
                logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
                logging.getLogger("transformers").setLevel(logging.ERROR)
                logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
                warnings.filterwarnings("ignore", category=FutureWarning)
                
                # Set environment to reduce progress bar verbosity
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                os.environ["TRANSFORMERS_VERBOSITY"] = "error"
                
                # Load silently - the calling code should handle progress display
                self._model = SentenceTransformer(self.model_name, device="cpu")
                
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._model
        return self._model
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def _text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / "embeddings_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                
                for key, embedding_list in data.items():
                    self._cache[key] = np.array(embedding_list, dtype=np.float32)
            except Exception:
                pass  # Cache corruption, start fresh
    
    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / "embeddings_cache.json"
        
        # Convert numpy arrays to lists for JSON serialization
        data = {k: v.tolist() for k, v in self._cache.items()}
        
        with open(cache_file, "w") as f:
            json.dump(data, f)
    
    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector as numpy array
        """
        if use_cache:
            cache_key = self._text_hash(text)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        if use_cache:
            self._cache[cache_key] = embedding
            # Periodically save cache
            if len(self._cache) % 100 == 0:
                self._save_cache()
        
        return embedding
    
    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            use_cache: Whether to use cached embeddings
            show_progress: Show progress bar
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Check cache for already embedded texts
        embeddings = []
        texts_to_embed = []
        text_indices = []
        
        for i, text in enumerate(texts):
            if use_cache:
                cache_key = self._text_hash(text)
                if cache_key in self._cache:
                    embeddings.append((i, self._cache[cache_key]))
                    continue
            
            texts_to_embed.append(text)
            text_indices.append(i)
        
        # Embed remaining texts
        if texts_to_embed:
            new_embeddings = self.model.encode(
                texts_to_embed,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )
            
            for i, (text_idx, embedding) in enumerate(zip(text_indices, new_embeddings)):
                embeddings.append((text_idx, embedding))
                
                if use_cache:
                    cache_key = self._text_hash(texts_to_embed[i])
                    self._cache[cache_key] = embedding
            
            if use_cache:
                self._save_cache()
        
        # Sort by original index and return
        embeddings.sort(key=lambda x: x[0])
        return np.array([e[1] for e in embeddings])
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        if self.cache_dir:
            cache_file = self.cache_dir / "embeddings_cache.json"
            if cache_file.exists():
                cache_file.unlink()
