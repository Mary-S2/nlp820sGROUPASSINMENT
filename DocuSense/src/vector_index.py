
"""
DocuSense Vector Index Module
Encapsulates FAISS indexing (IndexFlatIP) with persistent disk serialization,
metadata alignment, and an exact NumPy cosine similarity fallback.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

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
    logger.warning(
        "FAISS not installed. Using NumPy cosine similarity fallback."
    )


class VectorIndex:
    """Vector search index using FAISS or NumPy cosine similarity."""

    def __init__(
        self,
        dimension: int = config.EMBEDDING_DIM,
        index_path: Optional[Path] = None,
    ):
        self.dimension = dimension
        self.index_path = Path(index_path or config.FAISS_INDEX_PATH)
        self.faiss_index = None
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
        self._is_indexed = False

    def build_or_load(
        self,
        embeddings: Optional[np.ndarray] = None,
        chunks_df: Optional[pd.DataFrame] = None,
        force_rebuild: bool = False,
    ) -> "VectorIndex":
        """Load a compatible index or build a new one."""

        if self._is_indexed and not force_rebuild:
            return self

        # Avoid reusing an index built with incompatible vectors or
        # metadata from a different number of chunks.
        if HAS_FAISS and self.index_path.exists() and not force_rebuild:
            try:
                logger.info("Checking saved FAISS index: %s", self.index_path)
                saved_index = faiss.read_index(str(self.index_path))

                if saved_index.d != self.dimension:
                    raise ValueError(
                        f"Saved FAISS dimension is {saved_index.d}, "
                        f"but expected {self.dimension}."
                    )

                if (
                    chunks_df is not None
                    and saved_index.ntotal != len(chunks_df)
                ):
                    raise ValueError(
                        f"Saved FAISS index contains {saved_index.ntotal} "
                        f"vectors, but the current dataset contains "
                        f"{len(chunks_df)} chunks."
                    )

                self.faiss_index = saved_index
                self.embeddings = None

                if chunks_df is not None:
                    self._bind_metadata(chunks_df)

                self._is_indexed = True
                logger.info(
                    "Loaded compatible FAISS index: %s vectors, %s dimensions",
                    saved_index.ntotal,
                    saved_index.d,
                )
                return self

            except Exception as exc:
                logger.warning(
                    "Saved FAISS index is incompatible or unreadable: %s. "
                    "Attempting to rebuild it.",
                    exc,
                )
                self.faiss_index = None
                self._is_indexed = False

        # Load the source embedding matrix when one wasn't supplied.
        if embeddings is None:
            if not config.EMBEDDINGS_PATH.exists():
                raise FileNotFoundError(
                    f"Embeddings file not found: {config.EMBEDDINGS_PATH}"
                )

            logger.info(
                "Loading source embeddings from %s",
                config.EMBEDDINGS_PATH,
            )
            embeddings = np.load(str(config.EMBEDDINGS_PATH))

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be a 2D array; got shape {embeddings.shape}."
            )

        # Do not silently change the configured model dimension.
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: the source embeddings have "
                f"{embeddings.shape[1]} dimensions, but this index expects "
                f"{self.dimension}. Regenerate the embeddings using "
                f"{config.EMBEDDING_MODEL_NAME} before rebuilding the index."
            )

        # Each vector must correspond to the matching chunk's metadata.
        if chunks_df is not None and len(embeddings) != len(chunks_df):
            raise ValueError(
                f"Embedding/metadata count mismatch: found "
                f"{len(embeddings)} embeddings and {len(chunks_df)} chunks. "
                f"Regenerate embeddings from the current chunk dataset."
            )

        # Normalize vectors for cosine similarity using inner product.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = np.ascontiguousarray(
            embeddings / norms,
            dtype=np.float32,
        )

        self.embeddings = embeddings

        if chunks_df is not None:
            self._bind_metadata(chunks_df)
        else:
            self.metadata = []

        if HAS_FAISS:
            logger.info(
                "Building FAISS index: %s vectors, %s dimensions",
                len(embeddings),
                self.dimension,
            )

            self.faiss_index = faiss.IndexFlatIP(self.dimension)
            self.faiss_index.add(self.embeddings)

            try:
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.faiss_index, str(self.index_path))
                logger.info("Saved rebuilt FAISS index to %s", self.index_path)
            except Exception as exc:
                logger.warning("Could not save FAISS index: %s", exc)
        else:
            self.faiss_index = None
            logger.info("Using NumPy fallback for %s vectors", len(embeddings))

        self._is_indexed = True
        return self

    def _bind_metadata(self, chunks_df: pd.DataFrame) -> None:
        """Map vector row positions to their corresponding chunk metadata."""
        self.metadata = chunks_df.to_dict(orient="records")

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for the most similar chunks."""

        if not self._is_indexed:
            self.build_or_load()

        if k <= 0 or self.size == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32)

        if q.ndim == 1:
            q = q.reshape(1, -1)

        if q.ndim != 2 or q.shape[0] != 1:
            raise ValueError(
                f"Expected one query vector with shape (1, {self.dimension}); "
                f"received {q.shape}."
            )

        actual_dimension = (
            self.faiss_index.d
            if HAS_FAISS and self.faiss_index is not None
            else self.dimension
        )

        if q.shape[1] != actual_dimension:
            raise ValueError(
                f"Query/index dimension mismatch: query has {q.shape[1]} "
                f"dimensions, but the index expects {actual_dimension}. "
                f"Check the embedding model and rebuild stale index files."
            )

        if not np.all(np.isfinite(q)):
            raise ValueError("Query vector contains NaN or infinite values.")

        # Normalize the query for cosine similarity.
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        q = np.ascontiguousarray(q / norm, dtype=np.float32)

        top_k = min(k, self.size)

        if HAS_FAISS and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(q, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            if self.embeddings is None:
                if not config.EMBEDDINGS_PATH.exists():
                    raise FileNotFoundError(
                        f"Embeddings file not found: {config.EMBEDDINGS_PATH}"
                    )
                self.embeddings = np.load(
                    str(config.EMBEDDINGS_PATH)
                ).astype(np.float32)

            if self.embeddings.ndim != 2:
                raise ValueError("Stored embeddings must be a 2D array.")

            if self.embeddings.shape[1] != q.shape[1]:
                raise ValueError(
                    f"Query dimension {q.shape[1]} does not match stored "
                    f"embedding dimension {self.embeddings.shape[1]}."
                )

            all_scores = (self.embeddings @ q.T).ravel()
            indices = np.argsort(all_scores)[::-1][:top_k]
            scores = all_scores[indices]

        results = []

        for rank, (idx, score) in enumerate(
            zip(indices, scores), start=1
        ):
            if idx < 0:
                continue

            meta = (
                self.metadata[idx]
                if self.metadata and idx < len(self.metadata)
                else {}
            )

            results.append({
                "rank": rank,
                "index_position": int(idx),
                "chunk_id": str(meta.get("chunk_id", f"chunk_{idx}")),
                "article_id": int(meta.get("ArticleId", -1)),
                "category": str(meta.get("Category", "unknown")),
                "score": round(float(score), 4),
                "passage": str(meta.get("chunk_text", "")),
                "retrieval_method": (
                    "Semantic Embeddings (all-MiniLM-L6-v2 + FAISS)"
                    if HAS_FAISS and self.faiss_index is not None
                    else "Semantic Embeddings (NumPy cosine similarity)"
                ),
            })

        return results

    @property
    def size(self) -> int:
        """Return the number of vectors available for searching."""
        if HAS_FAISS and self.faiss_index is not None:
            return self.faiss_index.ntotal

        if self.embeddings is not None:
            return len(self.embeddings)

        return len(self.metadata)
