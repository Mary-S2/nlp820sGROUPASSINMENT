"""
DocuSense Data Ingestion & Validation Module
Handles loading, schema validation, deduplication, and metadata preservation
for the BBC News collection.
"""

import os
import re
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import config
except ImportError:
    from .. import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DocuSense.DataLoader")


class BBCDataLoader:
    """
    Robust loader for the BBC News document collection.
    Supports loading from raw Kaggle train files, cached article CSVs,
    or reconstructing articles from chunked collections.
    """

    def __init__(
        self,
        raw_path: Optional[str] = None,
        articles_path: Optional[str] = None,
        chunks_path: Optional[str] = None,
    ):
        self.raw_path = str(raw_path or config.RAW_DATA_PATH)
        self.articles_path = str(articles_path or config.ARTICLES_DATA_PATH)
        self.chunks_path = str(chunks_path or config.CHUNKS_DATA_PATH)
        self.articles_df: Optional[pd.DataFrame] = None
        self.chunks_df: Optional[pd.DataFrame] = None
        self.stats: Dict[str, Any] = {}

    def load_articles(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load and validate the article-level dataset.
        Priority:
          1. Pre-generated articles CSV (data/bbc_articles.csv)
          2. Raw Kaggle CSV/ZIP (data/BBC News Train.csv)
          3. Reconstructed from passage chunks (data/bbc_chunks.csv)
        """
        if self.articles_df is not None and not force_reload:
            return self.articles_df

        # Case 1: Load pre-generated articles CSV
        if os.path.exists(self.articles_path):
            logger.info(f"Loading articles from pre-generated CSV: {self.articles_path}")
            df = pd.read_csv(self.articles_path)
            self.articles_df = self._validate_and_normalize_articles(df)
            self._compute_stats()
            return self.articles_df

        # Case 2: Load raw Kaggle CSV/ZIP if present
        raw_zip = self.raw_path + ".zip"
        target_raw = raw_zip if os.path.exists(raw_zip) else self.raw_path

        if os.path.exists(target_raw):
            logger.info(f"Loading raw articles from: {target_raw}")
            df = pd.read_csv(target_raw)
            self.articles_df = self._process_raw_kaggle_df(df)
            self._compute_stats()
            # Cache for fast portable access
            self.articles_df.to_csv(self.articles_path, index=False)
            return self.articles_df

        # Case 3: Reconstruct from chunks CSV
        if os.path.exists(self.chunks_path):
            logger.info(f"Reconstructing articles from chunk collection: {self.chunks_path}")
            self.articles_df = self.reconstruct_articles_from_chunks()
            self._compute_stats()
            self.articles_df.to_csv(self.articles_path, index=False)
            return self.articles_df

        raise FileNotFoundError(
            f"No valid BBC News data source found. Checked:\n"
            f" - {self.articles_path}\n"
            f" - {self.raw_path}\n"
            f" - {self.chunks_path}"
        )

    def load_chunks(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load and validate the chunk-level passage dataset.
        Priority:
          1. Existing chunks CSV (data/bbc_chunks.csv)
          2. Generated on-the-fly from loaded articles
        """
        if self.chunks_df is not None and not force_reload:
            return self.chunks_df

        if os.path.exists(self.chunks_path):
            logger.info(f"Loading passages from chunk CSV: {self.chunks_path}")
            df = pd.read_csv(self.chunks_path)
            required_cols = {"chunk_id", "ArticleId", "Category", "chunk_text"}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"Chunks CSV missing required columns: {required_cols - set(df.columns)}")
            
            df["chunk_id"] = df["chunk_id"].astype(str)
            df["ArticleId"] = df["ArticleId"].astype(int)
            df["Category"] = df["Category"].astype(str).str.lower().str.strip()
            df["chunk_text"] = df["chunk_text"].astype(str)
            self.chunks_df = df
            return self.chunks_df

        logger.info("Chunk file not found. Generating chunks from articles...")
        from .preprocessing import DocumentChunker
        articles_df = self.load_articles()
        chunker = DocumentChunker(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
        self.chunks_df = chunker.chunk_dataframe(articles_df)
        self.chunks_df.to_csv(self.chunks_path, index=False)
        return self.chunks_df

    def _validate_and_normalize_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate columns, drop duplicates, and ensure clean structure."""
        required = {"ArticleId", "Category", "Text"}
        if not required.issubset(df.columns):
            raise ValueError(f"Article dataset missing required columns: {required - set(df.columns)}")

        initial_count = len(df)
        df["ArticleId"] = df["ArticleId"].astype(int)
        df["Category"] = df["Category"].astype(str).str.lower().str.strip()
        df["Text"] = df["Text"].astype(str)

        # Check for empty or whitespace-only texts
        df = df[df["Text"].str.strip().str.len() > 0].copy()

        # Deduplicate on Text
        dup_count = df["Text"].duplicated().sum()
        if dup_count > 0:
            logger.info(f"Detected {dup_count} duplicate articles out of {initial_count}. Retaining first occurrence.")
            df = df.drop_duplicates(subset=["Text"]).reset_index(drop=True)

        # Ensure title exists (extract from first sentence if missing)
        if "Title" not in df.columns:
            df["Title"] = df["Text"].apply(self._extract_headline)

        if "WordCount" not in df.columns:
            df["WordCount"] = df["Text"].apply(lambda t: len(t.split()))

        return df.sort_values("ArticleId").reset_index(drop=True)

    def _process_raw_kaggle_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw BBC News Kaggle format (ArticleId, Text, Category)."""
        logger.info(f"Raw Kaggle records loaded: {len(df)}")
        df = self._validate_and_normalize_articles(df)
        return df

    def reconstruct_articles_from_chunks(self) -> pd.DataFrame:
        """
        Reconstruct full articles from overlapping sliding-window passage chunks.
        Undoes the 30-word overlap cleanly to restore 100% of the original text.
        """
        chunks_df = pd.read_csv(self.chunks_path)
        logger.info(f"Reconstructing articles from {len(chunks_df)} chunks...")

        records = []
        # Group by ArticleId
        for article_id, group in chunks_df.groupby("ArticleId"):
            category = group["Category"].iloc[0]
            # Sort chunks by chunk index suffix (e.g., '1833_0', '1833_1')
            sorted_chunks = group.sort_values(
                by="chunk_id",
                key=lambda s: s.str.split("_").str[-1].astype(int)
            )

            full_words = []
            for i, (_, row) in enumerate(sorted_chunks.iterrows()):
                words = str(row["chunk_text"]).split()
                if i == 0:
                    full_words.extend(words)
                else:
                    # Skip the 30-word overlap from preceding chunk
                    full_words.extend(words[config.CHUNK_OVERLAP:])

            full_text = " ".join(full_words)
            headline = self._extract_headline(full_text)

            records.append({
                "ArticleId": int(article_id),
                "Category": category,
                "Title": headline,
                "Text": full_text,
                "WordCount": len(full_words),
                "ChunkCount": len(sorted_chunks),
            })

        reconstructed_df = pd.DataFrame(records).sort_values("ArticleId").reset_index(drop=True)
        logger.info(f"Successfully reconstructed {len(reconstructed_df)} unique articles.")
        return reconstructed_df

    @staticmethod
    def _extract_headline(text: str) -> str:
        """Extract headline (first sentence or first 12 words) and format title case."""
        clean = re.sub(r"\s+", " ", text).strip()
        dot_idx = clean.find(".")
        if 0 < dot_idx <= 120:
            headline = clean[:dot_idx].strip()
        else:
            words = clean.split()
            headline = " ".join(words[:10])
        return headline.capitalize()

    def _compute_stats(self) -> None:
        """Compute key dataset statistics for UI overview and reporting."""
        if self.articles_df is None:
            return

        cat_counts = self.articles_df["Category"].value_counts().to_dict()
        word_counts = self.articles_df["WordCount"].describe().to_dict()

        self.stats = {
            "total_articles": len(self.articles_df),
            "unique_articles": self.articles_df["ArticleId"].nunique(),
            "categories": cat_counts,
            "min_words": int(word_counts.get("min", 0)),
            "mean_words": round(float(word_counts.get("mean", 0)), 1),
            "median_words": int(word_counts.get("50%", 0)),
            "max_words": int(word_counts.get("max", 0)),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return dataset summary statistics."""
        if not self.stats:
            self._compute_stats()
        return self.stats


def get_data_loader() -> BBCDataLoader:
    """Convenience factory function."""
    return BBCDataLoader()
