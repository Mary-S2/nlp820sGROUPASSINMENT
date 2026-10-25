"""
DocuSense TF-IDF Classical Retrieval Module
Implements classical lexical retrieval using scikit-learn's TfidfVectorizer,
cosine similarity ranking, top-k retrieval, and lexical term overlap explanations.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import config
    from src.data_loader import get_data_loader
    from src.preprocessing import lexical_preprocess
except ImportError:
    from .. import config
    from .data_loader import get_data_loader
    from .preprocessing import lexical_preprocess

logger = logging.getLogger("DocuSense.TfidfSearch")


class TfidfSearchEngine:
    """
    Classical TF-IDF retrieval engine for document and passage collections.
    Computes sparse vector representations and cosine similarity rankings.
    """

    def __init__(
        self,
        chunks_df: Optional[pd.DataFrame] = None,
        model_cache_path: Optional[Path] = None,
    ):
        self.chunks_df = chunks_df
        self.model_cache_path = model_cache_path or (config.OUTPUTS_DIR / "tfidf_model.pkl")
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.feature_names: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit_or_load(self, force_refit: bool = False) -> "TfidfSearchEngine":
        """
        Fit TF-IDF on chunks or load a cached serialized model.
        """
        if self._is_fitted and not force_refit:
            return self

        if self.chunks_df is None:
            loader = get_data_loader()
            self.chunks_df = loader.load_chunks()

        # Check if saved model exists on disk
        if self.model_cache_path.exists() and not force_refit:
            try:
                logger.info(f"Loading cached TF-IDF model from: {self.model_cache_path}")
                with open(self.model_cache_path, "rb") as f:
                    saved = pickle.load(f)
                    self.vectorizer = saved["vectorizer"]
                    self.tfidf_matrix = saved["tfidf_matrix"]
                    self.feature_names = self.vectorizer.get_feature_names_out()
                    self._is_fitted = True
                    return self
            except Exception as e:
                logger.warning(f"Could not load cached TF-IDF model ({e}). Re-fitting from scratch.")

        logger.info(
            f"Fitting TfidfVectorizer on {len(self.chunks_df)} chunks "
            f"(ngram_range={config.TFIDF_NGRAM_RANGE}, max_df={config.TFIDF_MAX_DF}, min_df={config.TFIDF_MIN_DF})..."
        )
        self.vectorizer = TfidfVectorizer(
            stop_words=config.TFIDF_STOP_WORDS,
            max_df=config.TFIDF_MAX_DF,
            min_df=config.TFIDF_MIN_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks_df["chunk_text"])
        self.feature_names = self.vectorizer.get_feature_names_out()
        self._is_fitted = True

        # Cache model for fast reload
        try:
            with open(self.model_cache_path, "wb") as f:
                pickle.dump({
                    "vectorizer": self.vectorizer,
                    "tfidf_matrix": self.tfidf_matrix,
                }, f)
            logger.info(f"Cached TF-IDF model to {self.model_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache TF-IDF model: {e}")

        logger.info(f"TF-IDF Index created: shape {self.tfidf_matrix.shape} (passages x vocabulary)")
        return self

    def search(
        self,
        query: str,
        k: int = 5,
        preprocess: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks matching the query via TF-IDF cosine similarity.
        Returns a list of structured hit dictionaries with term overlap explanations.
        """
        if not self._is_fitted:
            self.fit_or_load()

        clean_q = lexical_preprocess(query) if preprocess else query
        if not clean_q.strip():
            return []

        # Vectorize query
        q_vec = self.vectorizer.transform([clean_q])
        
        # Guard against zero query vector (all words out of vocabulary)
        if q_vec.nnz == 0:
            logger.info(f"Query terms in '{query}' were completely out of vocabulary.")
            return []

        # Compute cosine similarity
        scores = cosine_similarity(q_vec, self.tfidf_matrix).ravel()

        # Top-k indices
        top_k = min(k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0.0:
                continue  # Filter out non-matching documents

            row = self.chunks_df.iloc[idx]
            chunk_text = str(row["chunk_text"])
            matched = self.get_matched_terms(clean_q, chunk_text)

            results.append({
                "rank": rank,
                "chunk_id": str(row["chunk_id"]),
                "article_id": int(row["ArticleId"]),
                "category": str(row["Category"]),
                "score": round(score, 4),
                "retrieval_method": "TF-IDF (Lexical)",
                "matched_terms": matched,
                "passage": chunk_text,
                "explanation": (
                    f"Matched terms: {', '.join(matched)}"
                    if matched else "Lexical frequency overlap"
                ),
            })

        return results

    def get_matched_terms(self, query: str, chunk_text: str) -> List[str]:
        """
        Identify exact vocabulary terms shared between the query and the chunk text.
        """
        if not self._is_fitted:
            return []

        q_vec = self.vectorizer.transform([query])
        q_term_indices = q_vec.nonzero()[1]
        q_terms = [self.feature_names[i] for i in q_term_indices]

        # Case-insensitive term presence
        text_lower = chunk_text.lower()
        matched = [term for term in q_terms if term in text_lower]
        return sorted(list(set(matched)))

    def get_vocab_size(self) -> int:
        """Return the size of the fitted vocabulary."""
        return len(self.feature_names) if self.feature_names is not None else 0


_tfidf_instance: Optional[TfidfSearchEngine] = None


def get_tfidf_search() -> TfidfSearchEngine:
    """Singleton getter for TfidfSearchEngine."""
    global _tfidf_instance
    if _tfidf_instance is None:
        _tfidf_instance = TfidfSearchEngine().fit_or_load()
    return _tfidf_instance
