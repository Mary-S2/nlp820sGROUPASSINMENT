"""
DocuSense Semantic Search Module
Implements dense semantic retrieval using sentence-transformers (all-MiniLM-L6-v2),
FAISS vector indexing, and fine-grained sentence-level evidence extraction.
"""

import logging
from typing import List, Dict, Any, Optional,Tuple
import numpy as np
import pandas as pd

try:
    import config
    from src.data_loader import get_data_loader
    from src.preprocessing import semantic_preprocess
    from src.vector_index import VectorIndex
except ImportError:
    from .. import config
    from .data_loader import get_data_loader
    from .preprocessing import semantic_preprocess
    from .vector_index import VectorIndex

logger = logging.getLogger("DocuSense.SemanticSearch")

# Global model cache to avoid repeated loads
_model_instance = None


def get_embedding_model(model_name: str = config.EMBEDDING_MODEL_NAME):
    """
    Lazily load SentenceTransformer model.
    """
    global _model_instance
    if _model_instance is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {model_name}...")
            _model_instance = SentenceTransformer(model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {_model_instance.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer ({e}).")
            raise
    return _model_instance


class SemanticSearchEngine:
    """
    Semantic retrieval engine powered by dense transformer embeddings.
    Maps natural language queries into 384-dimensional latent semantic space
    and queries the FAISS vector index.
    """

    def __init__(
        self,
        vector_index: Optional[VectorIndex] = None,
        model_name: str = config.EMBEDDING_MODEL_NAME,
    ):
        self.model_name = model_name
        self.vector_index = vector_index or VectorIndex()
        self.model = None
        self._is_ready = False

    def initialize(self, force_reload: bool = False) -> "SemanticSearchEngine":
        """
        Load embedding model, ensure dataset and FAISS index are initialized.
        """
        if self._is_ready and not force_reload:
            return self

        # 1. Load data
        loader = get_data_loader()
        chunks_df = loader.load_chunks()

        # 2. Build or load FAISS vector index
        self.vector_index.build_or_load(chunks_df=chunks_df, force_rebuild=force_reload)

        # 3. Model is loaded lazily on first query or now
        try:
            self.model = get_embedding_model(self.model_name)
        except Exception as e:
            logger.warning(f"SentenceTransformer not yet loaded or offline: {e}")

        self._is_ready = True
        return self

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into a normalized 384-dimensional vector.
        """
        if self.model is None:
            self.model = get_embedding_model(self.model_name)

        clean_q = semantic_preprocess(query)
        embedding = self.model.encode(
            [clean_q],
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
        )
        return np.asarray(embedding, dtype=np.float32)

    def search(
        self,
        query: str,
        k: int = 5,
        explain_sentences: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k semantically relevant chunks for a natural language query.
        Returns hits with ranking score and most relevant sentence excerpt.
        """
        if not self._is_ready:
            self.initialize()

        if not query or not query.strip():
            return []

        # Vectorize query
        q_vec = self.encode_query(query)

        # Query vector index
        raw_hits = self.vector_index.search(q_vec, k=k)

        # Enhance with fine-grained sentence-level evidence
        results = []
        for hit in raw_hits:
            chunk_text = hit.get("passage", "")
            if explain_sentences and chunk_text and self.model is not None:
                best_sentence, sent_score = self.find_most_similar_sentence(query, chunk_text)
                hit["most_similar_sentence"] = best_sentence
                hit["sentence_score"] = round(sent_score, 4)
                hit["explanation"] = f'Key sentence: "{best_sentence}"'
            else:
                hit["most_similar_sentence"] = ""
                hit["sentence_score"] = 0.0
                hit["explanation"] = "Dense semantic vector match"

            results.append(hit)

        return results

    def find_most_similar_sentence(
        self,
        query: str,
        chunk_text: str,
    ) -> Tuple[str, float]:
        """
        Identify the specific sentence inside a passage that has the highest
        cosine similarity with the query.
        """
        import re
        # Split passage on sentence boundaries (. ! ?)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text) if len(s.strip()) > 15]
        if not sentences:
            sentences = [chunk_text[:120]]

        try:
            sent_embs = self.model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
            q_emb = self.encode_query(query)
            scores = (sent_embs @ q_emb.T).ravel()
            best_idx = int(np.argmax(scores))
            return sentences[best_idx], float(scores[best_idx])
        except Exception:
            return sentences[0], 0.0


_semantic_instance: Optional[SemanticSearchEngine] = None


def get_semantic_search() -> SemanticSearchEngine:
    """Singleton getter for SemanticSearchEngine."""
    global _semantic_instance
    if _semantic_instance is None:
        _semantic_instance = SemanticSearchEngine().initialize()
    return _semantic_instance
