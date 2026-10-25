"""
DocuSense Document Clustering & Topic Discovery Module
Implements unsupervised document discovery using dense embeddings, K-Means clustering,
TF-IDF cluster keyword extraction, and representative document (medoid) selection.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin_min

try:
    import config
    from src.data_loader import get_data_loader
except ImportError:
    from .. import config
    from .data_loader import get_data_loader

logger = logging.getLogger("DocuSense.Clustering")


class DocumentClusterer:
    """
    Discovers latent topics across the document collection using K-Means clustering
    over dense article embeddings.
    """

    def __init__(
        self,
        n_clusters: int = config.DEFAULT_N_CLUSTERS,
        random_state: int = config.KMEANS_RANDOM_STATE,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.articles_df: Optional[pd.DataFrame] = None
        self.article_embeddings: Optional[np.ndarray] = None
        self.kmeans: Optional[KMeans] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_summaries: List[Dict[str, Any]] = []
        self._is_clustered = False

    def fit(
        self,
        n_clusters: Optional[int] = None,
        force_recompute: bool = False,
    ) -> "DocumentClusterer":
        """
        Fit K-Means clustering on article embeddings.
        """
        if n_clusters is not None:
            self.n_clusters = n_clusters

        if self._is_clustered and not force_recompute and self.kmeans and self.kmeans.n_clusters == self.n_clusters:
            return self

        # 1. Load articles
        loader = get_data_loader()
        self.articles_df = loader.load_articles().copy()

        # 2. Obtain article embeddings
        self.article_embeddings = self._get_article_embeddings(loader)

        # 3. Fit K-Means
        logger.info(f"Clustering {len(self.articles_df)} articles into k={self.n_clusters} clusters (seed={self.random_state})...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=config.KMEANS_N_INIT,
        )
        self.cluster_labels = self.kmeans.fit_predict(self.article_embeddings)
        self.articles_df["cluster"] = self.cluster_labels

        # 4. Generate cluster summaries (keywords + representative articles)
        self._build_cluster_summaries()
        self._is_clustered = True
        return self

    def _get_article_embeddings(self, loader) -> np.ndarray:
        """
        Load cached article embeddings or compute them from passage embeddings.
        Computes pooled mean of chunk embeddings per article if direct file is not found.
        """
        # Try loading saved article embeddings
        if config.ARTICLE_EMBEDDINGS_PATH.exists():
            try:
                emb = np.load(str(config.ARTICLE_EMBEDDINGS_PATH))
                if len(emb) == len(self.articles_df):
                    return emb.astype(np.float32)
            except Exception as e:
                logger.warning(f"Failed to load cached article embeddings ({e}).")

        # Derive article embeddings by mean-pooling the precomputed chunk embeddings
        chunks_df = loader.load_chunks()
        if config.EMBEDDINGS_PATH.exists():
            chunk_embs = np.load(str(config.EMBEDDINGS_PATH))
            logger.info("Computing article-level embeddings via mean-pooling of passage vectors...")
            
            # Map chunk indices to article ids
            article_ids = self.articles_df["ArticleId"].tolist()
            art_id_to_idx = {aid: i for i, aid in enumerate(article_ids)}
            
            pooled = np.zeros((len(article_ids), chunk_embs.shape[1]), dtype=np.float32)
            counts = np.zeros(len(article_ids), dtype=np.int32)
            
            for chunk_row_idx, row in chunks_df.iterrows():
                aid = row["ArticleId"]
                if aid in art_id_to_idx:
                    pos = art_id_to_idx[aid]
                    pooled[pos] += chunk_embs[chunk_row_idx]
                    counts[pos] += 1
                    
            counts[counts == 0] = 1
            pooled /= counts[:, np.newaxis]
            
            # L2-normalize pooled vectors
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            pooled /= norms
            
            # Cache for future runs
            try:
                np.save(str(config.ARTICLE_EMBEDDINGS_PATH), pooled)
            except Exception:
                pass
                
            return pooled

        raise FileNotFoundError("Chunk embeddings missing. Cannot compute article embeddings.")

    def _build_cluster_summaries(self) -> None:
        """
        Extract top TF-IDF keywords, size, and representative articles for each cluster.
        """
        # Fit article-level TF-IDF for keyword extraction
        tfidf_vec = TfidfVectorizer(
            stop_words="english",
            max_df=0.8,
            min_df=2,
            max_features=5000,
        )
        tfidf_matrix = tfidf_vec.fit_transform(self.articles_df["Text"])
        feature_names = tfidf_vec.get_feature_names_out()

        # Find closest article to each cluster centroid (medoid)
        closest_indices, _ = pairwise_distances_argmin_min(
            self.kmeans.cluster_centers_,
            self.article_embeddings,
        )

        self.cluster_summaries = []
        for cluster_id in range(self.n_clusters):
            mask = (self.cluster_labels == cluster_id)
            size = int(np.sum(mask))

            if size == 0:
                continue

            # Top TF-IDF keywords for this cluster
            cluster_tfidf_mean = tfidf_matrix[mask].mean(axis=0).A1
            top_term_indices = np.argsort(cluster_tfidf_mean)[::-1][:config.TOP_KEYWORDS_PER_CLUSTER]
            keywords = [feature_names[i] for i in top_term_indices]

            # Representative article
            medoid_idx = int(closest_indices[cluster_id])
            medoid_row = self.articles_df.iloc[medoid_idx]

            # Dominant official BBC category
            cat_breakdown = self.articles_df.loc[mask, "Category"].value_counts().to_dict()
            dominant_category = max(cat_breakdown, key=cat_breakdown.get)
            category_purity = round((cat_breakdown[dominant_category] / size) * 100, 1)

            # Auto-generated readable title
            generated_label = f"Topic {cluster_id + 1}: {', '.join(keywords[:3]).title()}"

            self.cluster_summaries.append({
                "cluster_id": cluster_id,
                "label": generated_label,
                "size": size,
                "keywords": keywords,
                "dominant_category": dominant_category,
                "category_purity_pct": category_purity,
                "category_breakdown": cat_breakdown,
                "representative_article": {
                    "article_id": int(medoid_row["ArticleId"]),
                    "title": str(medoid_row.get("Title", "Untitled")),
                    "category": str(medoid_row["Category"]),
                    "snippet": str(medoid_row["Text"])[:250] + "...",
                },
            })

    def get_cluster_summary(self) -> List[Dict[str, Any]]:
        """Return the list of cluster descriptions."""
        if not self._is_clustered:
            self.fit()
        return self.cluster_summaries

    def get_articles_by_cluster(self, cluster_id: int) -> pd.DataFrame:
        """Filter articles belonging to a specific cluster."""
        if not self._is_clustered:
            self.fit()
        return self.articles_df[self.articles_df["cluster"] == cluster_id].copy()

    def find_similar_articles(
        self,
        article_id: int,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find articles most similar to a chosen article using dense embedding cosine similarity.
        """
        if not self._is_clustered:
            self.fit()

        idx_matches = self.articles_df.index[self.articles_df["ArticleId"] == article_id].tolist()
        if not idx_matches:
            raise ValueError(f"Article ID {article_id} not found in collection.")

        target_idx = idx_matches[0]
        target_vec = self.article_embeddings[target_idx]

        # Cosine similarity against all articles
        scores = (self.article_embeddings @ target_vec).ravel()

        # Exclude the target article itself
        scores[target_idx] = -1.0
        top_indices = np.argsort(scores)[::-1][:k]

        similar = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.articles_df.iloc[idx]
            similar.append({
                "rank": rank,
                "article_id": int(row["ArticleId"]),
                "title": str(row.get("Title", "Untitled")),
                "category": str(row["Category"]),
                "score": round(float(scores[idx]), 4),
                "snippet": str(row["Text"])[:200] + "...",
            })

        return similar


_clusterer_instance: Optional[DocumentClusterer] = None


def get_clusterer(n_clusters: int = config.DEFAULT_N_CLUSTERS) -> DocumentClusterer:
    """Singleton getter for DocumentClusterer."""
    global _clusterer_instance
    if _clusterer_instance is None or _clusterer_instance.n_clusters != n_clusters:
        _clusterer_instance = DocumentClusterer(n_clusters=n_clusters).fit()
    return _clusterer_instance
