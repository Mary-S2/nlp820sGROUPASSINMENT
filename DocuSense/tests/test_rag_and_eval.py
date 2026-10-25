"""
DocuSense RAG & Evaluation Unit Tests
Validates CitationValidator, RAG grounding, out-of-domain abstention,
and the mathematical precision of Information Retrieval metrics.
"""

import math
import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import CitationValidator, RAGAssistant, get_rag_assistant
from src.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    dcg_at_k,
    ndcg_at_k,
    get_evaluation_engine,
)
from src.explanations import SearchExplainer


# --- Test 1: Citation Validation & Hallucination Guard ---

def test_citation_validator_valid_sources():
    """Verify CitationValidator recognizes valid source citations."""
    validator = CitationValidator()
    sources = [
        {"source_index": 1, "article_id": 1588, "category": "sport", "full_passage": "Arsenal manager Arsene Wenger..."},
        {"source_index": 2, "article_id": 1439, "category": "tech", "full_passage": "Microsoft anti-spyware tool..."},
    ]

    answer_valid = "Arsene Wenger signed a contract extension [Source 1]. Microsoft also released tools [Source 2]."
    audit = validator.validate_citations(answer_valid, sources)

    assert not audit["hallucination_detected"]
    assert audit["citation_count"] == 2
    assert "Source 1" in audit["valid_citations"]
    assert "Source 2" in audit["valid_citations"]
    assert not audit["is_abstention"]


def test_citation_validator_detects_hallucination():
    """Verify CitationValidator flags hallucinated source citations out of range."""
    validator = CitationValidator()
    sources = [
        {"source_index": 1, "article_id": 1588, "category": "sport", "full_passage": "Wenger details..."},
    ]

    answer_hallucinated = "Wenger signed a deal [Source 1], but according to [Source 5] another coach left."
    audit = validator.validate_citations(answer_hallucinated, sources)

    assert audit["hallucination_detected"]
    assert len(audit["invalid_citations"]) > 0
    assert "Source 5 (Out of range 1..1)" in audit["invalid_citations"]


def test_citation_validator_abstention():
    """Verify CitationValidator detects when model abstains due to insufficient evidence."""
    validator = CitationValidator()
    sources = [{"source_index": 1, "article_id": 100, "category": "tech", "full_passage": "Random tech passage"}]

    answer_abstained = "The provided BBC News collection does not contain sufficient evidence to answer this question."
    audit = validator.validate_citations(answer_abstained, sources)

    assert audit["is_abstention"]


# --- Test 2: RAG Extractive Fallback ---

def test_rag_extractive_grounding_synthesizes_citations():
    """Verify extractive fallback connects overlapping sentences and appends [Source X]."""
    rag = get_rag_assistant()
    sources = [
        {
            "source_index": 1,
            "article_id": 1588,
            "chunk_id": "1588_0",
            "category": "sport",
            "score": 0.85,
            "full_passage": "Arsenal manager Arsene Wenger has signed a new contract to stay at the club until may 2008. He announced his excitement on Wednesday.",
        }
    ]

    ans = rag._extractive_grounded_fallback(
        question="What did Arsene Wenger sign regarding his Arsenal contract?",
        sources=sources,
    )

    assert "[Source 1]" in ans
    assert "Arsene wenger has signed a new contract" in ans.lower() or "arsenal" in ans.lower()


def test_rag_extractive_fallback_abstains_when_zero_overlap():
    """Verify extractive fallback abstains when question keywords have no overlap with retrieved text."""
    rag = get_rag_assistant()
    sources = [
        {
            "source_index": 1,
            "article_id": 700,
            "chunk_id": "700_0",
            "category": "entertainment",
            "score": 0.20,
            "full_passage": "Pop superstar Madonna performed a musical concert in Paris with loud acoustics.",
        }
    ]

    ans = rag._extractive_grounded_fallback(
        question="Who was elected chancellor of Germany?",
        sources=sources,
    )

    assert "does not contain sufficient evidence" in ans.lower()


# --- Test 3: Mathematical Verification of Evaluation Metrics ---

def test_precision_at_k():
    """Verify Precision@k on known cases."""
    retrieved = [10, 20, 30, 40, 50]
    relevant = {20, 40, 99}

    assert precision_at_k(retrieved, relevant, k=1) == 0.0  # 10 is not relevant
    assert precision_at_k(retrieved, relevant, k=2) == 0.5  # 20 is relevant (1/2)
    assert precision_at_k(retrieved, relevant, k=4) == 0.5  # 20, 40 relevant (2/4)
    assert precision_at_k(retrieved, relevant, k=5) == 0.4  # 2/5


def test_recall_at_k():
    """Verify Recall@k on known cases."""
    retrieved = [10, 20, 30, 40, 50]
    relevant = {20, 40}  # total 2 relevant

    assert recall_at_k(retrieved, relevant, k=1) == 0.0  # 0/2
    assert recall_at_k(retrieved, relevant, k=2) == 0.5  # 1/2
    assert recall_at_k(retrieved, relevant, k=5) == 1.0  # 2/2


def test_reciprocal_rank():
    """Verify Reciprocal Rank calculation."""
    relevant = {30, 40}

    # First relevant at rank 1
    assert reciprocal_rank([30, 10, 20], relevant) == 1.0

    # First relevant at rank 3
    assert reciprocal_rank([10, 20, 30], relevant) == pytest.approx(1.0 / 3.0)

    # No relevant retrieved
    assert reciprocal_rank([1, 2, 3], relevant) == 0.0


def test_ndcg_at_k():
    """Verify nDCG@k logarithmic discounting and normalization."""
    relevant = {10, 20}

    # Case A: Perfect ranking (relevant at pos 1 and 2)
    perfect = [10, 20, 99, 98]
    assert pytest.approx(ndcg_at_k(perfect, relevant, k=2)) == 1.0

    # Case B: Sub-optimal ranking (relevant at pos 2 and 3)
    suboptimal = [99, 10, 20]
    # DCG = 0/log2(2) + 1/log2(3) + 1/log2(4) = 0 + 0.6309 + 0.5 = 1.1309
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    # nDCG = 1.1309 / 1.6309 ~= 0.6934
    score = ndcg_at_k(suboptimal, relevant, k=3)
    assert 0.65 <= score <= 0.75

    # Case C: Zero relevant retrieved
    assert ndcg_at_k([1, 2, 3], relevant, k=3) == 0.0


# --- Test 4: Benchmark Dataset Verification ---

def test_benchmark_dataset_integrity():
    """Verify labeled_benchmark_queries.json contains 30 queries across 5 categories and 4 types."""
    engine = get_evaluation_engine()
    queries = engine.load_benchmark()

    assert len(queries) == 30, f"Expected 30 benchmark queries, got {len(queries)}"

    categories = {q["category"] for q in queries}
    assert categories == {"business", "entertainment", "politics", "sport", "tech"}

    query_types = {q["query_type"] for q in queries}
    assert query_types == {"exact_keyword", "synonym_paraphrase", "conceptual_semantic", "entity_typo"}

    # Ensure every query has an ID and at least 1 relevant article ID
    for q in queries:
        assert len(q["relevant_article_ids"]) >= 1
        assert len(q["query"].strip()) > 3


# --- Test 5: Explanations & Calibration Notice ---

def test_search_explainer_calibration_warning():
    """Verify SearchExplainer includes score calibration warning."""
    tfidf_hits = [{"chunk_id": "c1", "category": "tech"}]
    semantic_hits = [{"chunk_id": "c2", "category": "business"}]

    comp = SearchExplainer.compare_hits(tfidf_hits, semantic_hits)
    assert "score_calibration_note" in comp
    assert "not on a common calibrated scale" in comp["score_calibration_note"].lower()
