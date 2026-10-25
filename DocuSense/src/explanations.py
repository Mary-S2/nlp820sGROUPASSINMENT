"""
DocuSense Explainable Ranking Module
Produces transparent, academically rigorous explanations for search rankings:
  - Lexical term overlap analysis for TF-IDF
  - Semantic sentence alignment for dense embeddings
  - Calibration warnings for score comparability
  - Comparative ranking explanations (lexical vs. semantic differences)
"""

import re
import logging
from typing import List, Dict, Any, Optional

try:
    import config
except ImportError:
    from .. import config

logger = logging.getLogger("DocuSense.Explanations")


class SearchExplainer:
    """
    Generates transparent, evidence-based explanations for search results.
    Distinguishes lexical keyword frequency from latent semantic similarity.
    """

    @staticmethod
    def explain_tfidf_hit(
        query: str,
        chunk_text: str,
        score: float,
        matched_terms: List[str],
    ) -> Dict[str, Any]:
        """
        Explain a TF-IDF match based on exact vocabulary overlap and term importance.
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        chunk_words = re.findall(r"\w+", chunk_text.lower())
        chunk_word_set = set(chunk_words)

        overlap = [w for w in matched_terms if w in chunk_word_set]
        coverage_pct = round((len(overlap) / max(len(query_words), 1)) * 100, 1)

        summary = (
            f"Lexical match (score: {score:.3f}): Shares {len(overlap)} query term(s) "
            f"({', '.join(overlap)}) covering {coverage_pct}% of query vocabulary."
            if overlap else
            f"Lexical match (score: {score:.3f}): Matched sublinear n-gram frequency terms."
        )

        return {
            "method": "TF-IDF (Lexical)",
            "score": score,
            "matched_terms": overlap,
            "query_coverage_pct": coverage_pct,
            "summary": summary,
            "score_interpretation": (
                "TF-IDF cosine score reflects sparse bag-of-words overlap weighted by "
                "inverse document frequency. Scores typically range between 0.05 and 0.50."
            ),
        }

    @staticmethod
    def explain_semantic_hit(
        query: str,
        chunk_text: str,
        score: float,
        best_sentence: str,
        sentence_score: float,
    ) -> Dict[str, Any]:
        """
        Explain a dense transformer match based on vector cosine similarity
        and identifying the most informative sentence inside the passage.
        """
        summary = (
            f"Semantic similarity (score: {score:.3f}): Latent vector alignment in 384-dimensional "
            f"embedding space. Most relevant sentence (similarity: {sentence_score:.3f}): \"{best_sentence[:140]}...\""
            if best_sentence else
            f"Semantic similarity (score: {score:.3f}): Dense embedding vector cosine proximity."
        )

        return {
            "method": "Semantic Embeddings (all-MiniLM-L6-v2 + FAISS)",
            "score": score,
            "key_sentence": best_sentence,
            "sentence_similarity": sentence_score,
            "summary": summary,
            "score_interpretation": (
                "Dense cosine similarity measures conceptual and semantic proximity "
                "regardless of whether the exact query words appear. Normalized scores "
                "typically range between 0.35 and 0.85."
            ),
        }

    @staticmethod
    def compare_hits(
        tfidf_hits: List[Dict[str, Any]],
        semantic_hits: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Produce a comparative breakdown of how lexical and semantic retrieval performed
        on the exact same query.
        """
        tfidf_ids = [h["chunk_id"] for h in tfidf_hits]
        semantic_ids = [h["chunk_id"] for h in semantic_hits]

        common_ids = list(set(tfidf_ids).intersection(set(semantic_ids)))
        tfidf_only = [cid for cid in tfidf_ids if cid not in semantic_ids]
        semantic_only = [cid for cid in semantic_ids if cid not in tfidf_ids]

        # Category distribution comparison
        tfidf_cats = [h.get("category", "") for h in tfidf_hits]
        semantic_cats = [h.get("category", "") for h in semantic_hits]

        explanation = []
        if common_ids:
            explanation.append(
                f"{len(common_ids)} passage(s) were retrieved by BOTH methods, "
                f"indicating both strong keyword overlap and high conceptual relevance."
            )
        if semantic_only:
            explanation.append(
                f"Semantic search discovered {len(semantic_only)} passage(s) missed by TF-IDF, "
                f"likely due to synonyms, paraphrases, or absence of exact keyword matches."
            )
        if tfidf_only:
            explanation.append(
                f"TF-IDF prioritized {len(tfidf_only)} passage(s) with high exact term frequency "
                f"that had lower global conceptual similarity in the dense embedding space."
            )

        return {
            "common_chunk_ids": common_ids,
            "tfidf_only_chunk_ids": tfidf_only,
            "semantic_only_chunk_ids": semantic_only,
            "tfidf_categories": tfidf_cats,
            "semantic_categories": semantic_cats,
            "jaccard_similarity": round(
                len(common_ids) / max(len(set(tfidf_ids).union(set(semantic_ids))), 1), 3
            ),
            "comparative_analysis": " ".join(explanation),
            "score_calibration_note": (
                "CRITICAL: TF-IDF cosine scores and Transformer embedding cosine scores are NOT "
                "on a common calibrated scale. A TF-IDF score of 0.35 often indicates a very strong "
                "lexical match, whereas an embedding score of 0.35 represents a weak semantic match. "
                "Do not compare numerical scores directly across methods."
            ),
        }
