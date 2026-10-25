"""
DocuSense Text Preprocessing & Chunking Module
Provides differentiated preprocessing pipelines:
  - Lexical preprocessing for classical TF-IDF matching
  - Semantic preprocessing for Transformer sentence embeddings
  - Passage chunking with sliding window and metadata preservation
"""

import re
import logging
from typing import List, Dict, Any, Optional
import pandas as pd

try:
    import config
except ImportError:
    from .. import config

logger = logging.getLogger("DocuSense.Preprocessing")

# Optional NLTK lemmatizer for lexical pipeline
_lemmatizer = None


def get_lemmatizer():
    """Lazily load NLTK WordNet lemmatizer with quiet download fallback."""
    global _lemmatizer
    if _lemmatizer is None:
        try:
            import nltk
            try:
                from nltk.stem import WordNetLemmatizer
                _lemmatizer = WordNetLemmatizer()
                # Test call to trigger any missing corpus error safely
                _lemmatizer.lemmatize("testing")
            except (LookupError, AttributeError):
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                from nltk.stem import WordNetLemmatizer
                _lemmatizer = WordNetLemmatizer()
        except ImportError:
            logger.warning("NLTK not installed. Lemmatization will fall back to identity.")
            _lemmatizer = False
    return _lemmatizer


def clean_whitespace(text: str) -> str:
    """Collapse consecutive spaces, newlines, and tabs into a single space."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def lexical_preprocess(
    text: str,
    lemmatize: bool = False,
    remove_punctuation: bool = True,
    lower: bool = True,
) -> str:
    """
    Preprocessing tailored for lexical retrieval (TF-IDF):
    - Lowercasing
    - Optional punctuation stripping
    - Optional lemmatization
    """
    clean = clean_whitespace(text)
    if lower:
        clean = clean.lower()

    if remove_punctuation:
        # Keep alphanumeric and single spaces
        clean = re.sub(r"[^\w\s]", " ", clean)
        clean = clean_whitespace(clean)

    if lemmatize:
        lem = get_lemmatizer()
        if lem:
            words = clean.split()
            clean = " ".join(lem.lemmatize(w) for w in words)

    return clean


def semantic_preprocess(text: str) -> str:
    """
    Preprocessing tailored for Transformer Sentence Embeddings (all-MiniLM-L6-v2):
    Preserves casing, punctuation, dates, numbers, and negations,
    only normalizing irregular whitespaces.
    Transformers require full sentence syntax to construct rich contextual representations.
    """
    return clean_whitespace(text)


class DocumentChunker:
    """
    Splits long documents into overlapping passages for fine-grained retrieval
    and Retrieval-Augmented Generation (RAG).
    Preserves ArticleId, Category, and creates unique chunk identifiers.
    """

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        overlap: int = config.CHUNK_OVERLAP,
    ):
        if overlap >= chunk_size:
            raise ValueError(f"Overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Split a single text string into sliding-window word chunks.
        """
        words = clean_whitespace(text).split()
        if not words:
            return []

        if len(words) <= self.chunk_size:
            return [" ".join(words)]

        chunks = []
        for start in range(0, len(words), self.stride):
            end = start + self.chunk_size
            piece = words[start:end]
            chunks.append(" ".join(piece))
            if end >= len(words):
                break

        return chunks

    def chunk_dataframe(
        self,
        articles_df: pd.DataFrame,
        text_col: str = "Text",
        id_col: str = "ArticleId",
        category_col: str = "Category",
    ) -> pd.DataFrame:
        """
        Transform an article DataFrame into a passage-level chunk DataFrame.
        Each row contains:
          - chunk_id: f"{ArticleId}_{chunk_index}" (e.g., '1833_0')
          - ArticleId: source document identifier
          - Category: official BBC category
          - chunk_index: 0-indexed position within article
          - chunk_text: passage text
          - word_count: length in words
        """
        rows = []
        for _, r in articles_df.iterrows():
            article_id = r[id_col]
            category = r[category_col]
            text = r[text_col]

            chunks = self.chunk_text(text)
            for i, chunk in enumerate(chunks):
                rows.append({
                    "chunk_id": f"{article_id}_{i}",
                    "ArticleId": int(article_id),
                    "Category": str(category),
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "word_count": len(chunk.split()),
                })

        result_df = pd.DataFrame(rows)
        logger.info(
            f"Chunked {len(articles_df)} articles into {len(result_df)} passages "
            f"(chunk_size={self.chunk_size}, overlap={self.overlap})."
        )
        return result_df
