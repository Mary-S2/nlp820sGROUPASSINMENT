"""
DocuSense Vector Index Module
Encapsulates FAISS indexing (IndexFlatIP) with persistent disk serialization,
metadata alignment, and an exact NumPy cosine similarity fallback.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import config
except ImportError:
    from .. import config

logger = logging.getLogger("DocuSense.VectorIndex")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS not installed. Using high-performance NumPy cosine similarity fallback.")


class VectorIndex:
    """
    Vector search index for dense sentence embeddings.
    Uses FAISS IndexFlatIP for high-speed inner product (cosine similarity on unit vectors),
    with an exact NumPy dot-product fallback when FAISS is unavailable.
    """

    def __init__(
        self,
        dimension: int = config.EMBEDDING_DIM,
        index_path: Optional[Path] = None,
    ):
        self.dimension = dimension
        self.index_path = index_path or config.FAISS_INDEX_PATH
        self.faiss_index = None
        self.embeddings: Optional[np.ndarray] = None  # Raw matrix for fallback or inspection
        self.metadata: List[Dict[str, Any]] = []
        self._is_indexed = False

    def build_or_load(
        self,
        embeddings: Optional[np.ndarray] = None,
        chunks_df: Optional[pd.DataFrame] = None,
        force_rebuild: bool = False,
    ) -> "VectorIndex":
        """
        Build the FAISS index or load it from disk along with chunk metadata.
        """
        if self._is_indexed and not force_rebuild:
            return self

        # 1. Try loading pre-built FAISS index from disk
        if HAS_FAISS and self.index_path.exists() and not force_rebuild:
            try:
                logger.info(f"Loading persistent FAISS index from: {self.index_path}")
                self.faiss_index = faiss.read_index(str(self.index_path))
                if chunks_df is not None:
                    self._bind_metadata(chunks_df)
                self._is_indexed = True
                return self
            except Exception as e:
                logger.warning(f"Failed to load FAISS index from disk ({e}). Rebuilding.")

        # 2. Build index from embeddings matrix
        if embeddings is None:
            if config.EMBEDDINGS_PATH.exists():
                logger.info(f"Loading precomputed embeddings from: {config.EMBEDDINGS_PATH}")
                embeddings = np.load(str(config.EMBEDDINGS_PATH))
            else:
                raise FileNotFoundError(f"Embeddings file not found: {config.EMBEDDINGS_PATH}")

        # Ensure float32 and correct 2D shape
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")

        self.dimension = embeddings.shape[1]
        self.embeddings = embeddings

        if chunks_df is not None:
            self._bind_metadata(chunks_df)

        if HAS_FAISS:
            logger.info(f"Building FAISS IndexFlatIP for {len(embeddings)} vectors (dim={self.dimension})...")
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
            self.faiss_index.add(self.embeddings)
            
            # Save to disk
            try:
                faiss.write_index(self.faiss_index, str(self.index_path))
                logger.info(f"Saved FAISS index to {self.index_path}")
            except Exception as e:
                logger.warning(f"Could not persist FAISS index to disk: {e}")
        else:
            logger.info(f"Indexed {len(embeddings)} vectors into NumPy fallback matrix.")

        self._is_indexed = True
        return self

    def _bind_metadata(self, chunks_df: pd.DataFrame) -> None:
        """Store passage metadata mapped 1-to-1 with index row positions."""
        self.metadata = chunks_df.to_dict(orient="records")

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        query_vector: 1D array of shape (dimension,) or 2D of shape (1, dimension).
        Returns list of structured hit dictionaries.
        """
        if not self._is_indexed:
            self.build_or_load()

        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        # L2-normalize query vector for true cosine similarity
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        q = q / norm

        top_k = min(k, self.size)
        results = []

        if HAS_FAISS and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(q, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Exact NumPy Cosine Matrix Multiplication
            if self.embeddings is None:
                self.embeddings = np.load(str(config.EMBEDDINGS_PATH)).astype(np.float32)
            all_scores = (self.embeddings @ q.T).ravel()
            indices = np.argsort(all_scores)[::-1][:top_k]
            scores = all_scores[indices]

        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0:
                continue

            meta = self.metadata[idx] if self.metadata and idx < len(self.metadata) else {}
            results.append({
                "rank": rank,
                "index_position": int(idx),
                "chunk_id": str(meta.get("chunk_id", f"chunk_{idx}")),
                "article_id": int(meta.get("ArticleId", -1)),
                "category": str(meta.get("Category", "unknown")),
                "score": round(float(score), 4),
                "passage": str(meta.get("chunk_text", "")),
                "retrieval_method": "Semantic Embeddings (all-MiniLM-L6-v2 + FAISS)",
            })

        return results

    @property
    def size(self) -> int:
        """Total number of indexed vectors."""
        if HAS_FAISS and self.faiss_index is not None:
            return self.faiss_index.ntotal
        if self.embeddings is not None:
            return len(self.embeddings)
        return len(self.metadata)
