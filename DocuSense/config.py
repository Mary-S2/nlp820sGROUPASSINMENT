"""
DocuSense Configuration Module
Centralized, portable configuration for dataset paths, NLP parameters,
vector indexing, clustering, RAG providers, and evaluation settings.
Works seamlessly across Windows, Linux, and Google Colab environments.
"""

import os
from pathlib import Path

# Base Paths (dynamic resolution relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset File Paths
RAW_DATA_FILENAME = "BBC News Train.csv"
RAW_DATA_PATH = DATA_DIR / RAW_DATA_FILENAME
ARTICLES_DATA_PATH = DATA_DIR / "bbc_articles.csv"
CHUNKS_DATA_PATH = DATA_DIR / "bbc_chunks.csv"
EMBEDDINGS_PATH = DATA_DIR / "bbc_embeddings.npy"
ARTICLE_EMBEDDINGS_PATH = DATA_DIR / "bbc_article_embeddings.npy"
FAISS_INDEX_PATH = DATA_DIR / "bbc_faiss.index"
BENCHMARK_QUERIES_PATH = DATA_DIR / "labeled_benchmark_queries.json"

# Preprocessing & Chunking Hyperparameters
CHUNK_SIZE = 150         # words per passage
CHUNK_OVERLAP = 30       # words overlapping between adjacent chunks
STRIDE = CHUNK_SIZE - CHUNK_OVERLAP  # 120 words

# Categories in BBC News Collection
BBC_CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]

# TF-IDF Configuration
TFIDF_STOP_WORDS = "english"
TFIDF_MAX_DF = 0.8
TFIDF_MIN_DF = 2
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams for rich lexical phrase matching

# Semantic Embedding Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 64
NORMALIZE_EMBEDDINGS = True

# Vector Index Configuration
FAISS_METRIC = "INNER_PRODUCT"  # Equivalent to Cosine Similarity when embeddings are L2 normalized

# Clustering Configuration
DEFAULT_N_CLUSTERS = 5
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10
TOP_KEYWORDS_PER_CLUSTER = 8

# RAG (Retrieval-Augmented Generation) Configuration
# Supported providers: "gemini", "openai", "groq", "huggingface", "extractive_fallback"
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "extractive_fallback")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# UI Server Settings
DEFAULT_SERVER_PORT = 7860
DEFAULT_SHARE = False
