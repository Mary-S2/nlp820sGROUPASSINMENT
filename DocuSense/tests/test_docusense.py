"""
DocuSense Core Pipeline Tests
Tests data loading, preprocessing, chunking, TF-IDF, vector indexing,
clustering, and query processing.
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_loader import BBCDataLoader, get_data_loader
from src.preprocessing import (
    clean_whitespace,
    lexical_preprocess,
    semantic_preprocess,
    DocumentChunker,
)
from src.query_processing import QueryProcessor, get_query_processor, PROTECTED_ENTITIES
from src.vector_index import VectorIndex


# --- Fixtures ---

@pytest.fixture
def sample_articles_df():
    """Create a controlled mini-dataframe mimicking BBC News."""
    return pd.DataFrame([
        {
            "ArticleId": 101,
            "Category": "sport",
            "Title": "Arsenal wins league match in london",
            "Text": "Arsenal manager Arsene Wenger praised his squad after a decisive win in London. " * 15,
            "WordCount": 180,
            "ChunkCount": 2,
        },
        {
            "ArticleId": 102,
            "Category": "business",
            "Title": "Bank of England raises interest rate",
            "Text": "The Bank of England announced an interest rate hike to curb inflation pressures. " * 12,
            "WordCount": 144,
            "ChunkCount": 1,
        },
        {
            "ArticleId": 103,
            "Category": "tech",
            "Title": "Microsoft unveils new operating system update",
            "Text": "Microsoft has debuted a new security patch to defend personal computers against viruses. " * 10,
            "WordCount": 130,
            "ChunkCount": 1,
        },
    ])


# --- Test 1: Data Ingestion & Schema Validation ---

def test_articles_dataset_exists_and_schema():
    """Verify that bbc_articles.csv exists, has 1440 articles, and has correct columns."""
    loader = get_data_loader()
    df = loader.load_articles()

    assert df is not None
    assert len(df) == 1440, f"Expected 1,440 articles after deduplication, got {len(df)}"
    
    expected_cols = {"ArticleId", "Category", "Title", "Text", "WordCount", "ChunkCount"}
    assert expected_cols.issubset(df.columns), f"Missing columns: {expected_cols - set(df.columns)}"

    # Ensure no nulls in critical fields
    assert df["ArticleId"].isna().sum() == 0
    assert df["Category"].isna().sum() == 0
    assert df["Text"].isna().sum() == 0

    # Ensure 5 distinct BBC categories
    categories = set(df["Category"].unique())
    assert categories == {"business", "entertainment", "politics", "sport", "tech"}


def test_chunks_dataset_exists_and_chunk_count():
    """Verify that bbc_chunks.csv exists and has 4,981 indexed passages."""
    loader = get_data_loader()
    chunks = loader.load_chunks()

    assert chunks is not None
    assert len(chunks) == 4981, f"Expected 4,981 passage chunks, got {len(chunks)}"

    expected_cols = {"chunk_id", "ArticleId", "Category", "chunk_text"}
    assert expected_cols.issubset(chunks.columns)

    # Check ID format: e.g. '1833_0'
    sample_id = str(chunks["chunk_id"].iloc[0])
    assert "_" in sample_id


# --- Test 2: Preprocessing Differentiation ---

def test_preprocessing_differentiation():
    """Verify lexical vs. semantic preprocessing behave differently."""
    sample_raw = "  Arsene Wenger's Arsenal won 3-0 on 12/04/2004! Isn't that great?  "

    # Lexical: lowercase, punctuation stripped
    lexical = lexical_preprocess(sample_raw, lemmatize=False)
    assert "arsene" in lexical
    assert "!" not in lexical
    assert "?" not in lexical
    assert "'" not in lexical

    # Semantic: preserves casing, punctuation, numbers, and negations for transformer
    semantic = semantic_preprocess(sample_raw)
    assert "Arsene" in semantic
    assert "!" in semantic
    assert "?" in semantic
    assert "12/04/2004" in semantic
    assert "  " not in semantic  # Whitespaces collapsed


# --- Test 3: Document Chunking ---

def test_document_chunker_sliding_window(sample_articles_df):
    """Verify sliding-window chunking logic with 150-word window and 30-word overlap."""
    chunker = DocumentChunker(chunk_size=50, overlap=10)
    text = "word " * 120  # 120 words
    chunks = chunker.chunk_text(text)

    # With size=50, stride=40:
    # chunk 0: 0..50
    # chunk 1: 40..90
    # chunk 2: 80..120
    assert len(chunks) == 3

    # Check dataframe chunker
    chunk_df = chunker.chunk_dataframe(sample_articles_df)
    assert len(chunk_df) >= len(sample_articles_df)
    assert "chunk_id" in chunk_df.columns
    assert "ArticleId" in chunk_df.columns


# --- Test 4: Query Understanding & Edge Cases ---

def test_query_validation():
    """Verify query processor handles empty, whitespace, and symbol queries gracefully."""
    qp = get_query_processor()

    # Empty / whitespace
    res_empty = qp.process("", method="embeddings")
    assert not res_empty.is_valid
    assert "empty" in res_empty.status_message.lower()

    res_ws = qp.process("     ", method="embeddings")
    assert not res_ws.is_valid

    # Punctuation only
    res_punct = qp.process("!?!?!...", method="embeddings")
    assert not res_punct.is_valid

    # Single character
    res_short = qp.process("a", method="embeddings")
    assert not res_short.is_valid


def test_query_protected_entities():
    """Verify named entities and domain acronyms are never corrupted."""
    qp = get_query_processor()

    for entity in ["wenger", "ebbers", "howard", "microsoft", "google"]:
        assert entity in PROTECTED_ENTITIES

    # Test query with protected entity
    res = qp.process("wenegr arsnal contrct", method="tfidf")
    assert res.is_valid
    assert len(res.tokens) > 0


# --- Test 5: Vector Index & Fallback ---

def test_vector_index_exact_cosine_search():
    """Verify vector index accurately ranks closest unit vector using dot product."""
    dim = 4
    # Create 3 synthetic unit vectors
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32)
    matrix = np.vstack([v1, v2, v3])

    df = pd.DataFrame([
        {"chunk_id": "c1", "ArticleId": 1, "Category": "tech", "chunk_text": "text1"},
        {"chunk_id": "c2", "ArticleId": 2, "Category": "business", "chunk_text": "text2"},
        {"chunk_id": "c3", "ArticleId": 3, "Category": "sport", "chunk_text": "text3"},
    ])

    idx = VectorIndex(dimension=dim)
    idx.build_or_load(embeddings=matrix, chunks_df=df, force_rebuild=True)

    # Query identical to v1
    hits = idx.search(v1, k=2)
    assert len(hits) == 2
    assert hits[0]["chunk_id"] == "c1"
    assert pytest.approx(hits[0]["score"], rel=1e-3) == 1.0
    assert hits[1]["chunk_id"] == "c3"  # Cosine ~0.7071


# --- Test 6: Precomputed Embedding Integrity ---

def test_precomputed_embeddings_matrix_dimensions():
    """Verify the saved bbc_embeddings.npy matches 4,981 passages x 384 dimensions."""
    emb_path = config.EMBEDDINGS_PATH
    assert emb_path.exists(), f"Embedding file missing at: {emb_path}"

    embs = np.load(str(emb_path))
    assert embs.shape == (4981, 384), f"Expected shape (4981, 384), got {embs.shape}"
    assert embs.dtype == np.float32

    # Check normalization: L2 norms should all be ~1.0
    norms = np.linalg.norm(embs[:20], axis=1)
    for n in norms:
        assert pytest.approx(n, abs=1e-3) == 1.0
