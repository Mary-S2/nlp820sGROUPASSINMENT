# DocuSense: Technical Architecture & Algorithmic Guide

**Project:** DocuSense — Semantic Search, Similarity Index & RAG Assistant for a Document Collection  
**Course:** NAT820S Natural Language Processing and Applications  
**Institution:** Namibia University of Science and Technology (NUST)  

---

## 1. System Pipeline Overview

DocuSense implements a modular, production-ready NLP architecture decoupled into five foundational stages:
1. **Corpus Ingestion & Preprocessing Pipeline** (`data_loader.py`, `preprocessing.py`)
2. **Dual Indexing Engine** (`tfidf_search.py`, `semantic_search.py`, `vector_index.py`)
3. **Unsupervised Topic Clustering Engine** (`clustering.py`)
4. **Query Understanding & Normalization** (`query_processing.py`)
5. **Grounded RAG Assistant & Verification** (`rag.py`, `explanations.py`)

---

## 2. Ingestion & Preprocessing Mathematics

### 2.1 Sliding-Window Passage Chunking
Long news documents contain multiple thematic shifts. To preserve fine-grained retrieval precision and stay within the 256-token optimal window of `all-MiniLM-L6-v2`, documents are segmented via a fixed word sliding window:

$$\text{Stride } S = W - O = 150 - 30 = 120 \text{ words}$$

For a document with $N$ words, the total number of chunks $C$ produced is:
$$C = \max\left(1, \left\lceil \frac{N - W}{S} \right\rceil + 1\right)$$

- **Window Size ($W$)**: 150 words.
- **Overlap ($O$)**: 30 words.
- **Outcome**: 1,440 unique documents yield **4,981 indexed passages**.
- **Chunk ID Convention**: `{ArticleId}_{chunk_index}` (e.g. `1833_0`, `1833_1`).

---

## 3. Retrieval Representations & Algorithms

### 3.1 Classical TF-IDF Retrieval
The lexical engine uses `scikit-learn` `TfidfVectorizer` configured with:
- Sublinear term frequency scaling:
  $$\text{tf}_{\text{sublinear}}(t, d) = 1 + \log(\text{tf}(t, d)) \quad \text{for } \text{tf}(t, d) > 0$$
- Inverse document frequency with smooth weighting:
  $$\text{idf}(t) = \log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
- Cosine similarity ranking:
  $$\text{sim}_{\text{tfidf}}(\mathbf{q}, \mathbf{d}) = \frac{\sum_{i=1}^{V} q_i d_i}{\sqrt{\sum_{i=1}^{V} q_i^2} \sqrt{\sum_{i=1}^{V} d_i^2}}$$

### 3.2 Dense Semantic Embeddings (`all-MiniLM-L6-v2`)
- **Backbone**: 6 transformer layers, 12 attention heads, 384 hidden dimension.
- **Pooling**: Mean-pooling across token embeddings:
  $$\mathbf{u} = \frac{1}{\sum_{i=1}^{L} m_i} \sum_{i=1}^{L} m_i \mathbf{h}_i$$
  where $m_i \in \{0, 1\}$ is the attention mask and $\mathbf{h}_i$ is the output vector of token $i$.
- **L2 Normalization**:
  $$\mathbf{v} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2} = \frac{\mathbf{u}}{\sqrt{\sum_{j=1}^{384} u_j^2}}$$

### 3.3 FAISS Vector Indexing (`IndexFlatIP`)
Because all vectors are unit-normalized ($\|\mathbf{v}\| = 1$), the inner product computed by FAISS `IndexFlatIP` is mathematically equivalent to cosine similarity:
$$\langle \mathbf{q}, \mathbf{d} \rangle = \sum_{j=1}^{384} q_j d_j = \|\mathbf{q}\| \|\mathbf{d}\| \cos(\theta) = 1 \cdot 1 \cdot \cos(\theta) = \cos(\theta)$$

- **NumPy Fallback**: If FAISS is not compiled or available on a target operating system, the system falls back to:
  $$\mathbf{s} = \mathbf{E} \mathbf{q}^T \in \mathbb{R}^{4981}$$
  where $\mathbf{E} \in \mathbb{R}^{4981 \times 384}$ is the precomputed embeddings matrix.

---

## 4. Unsupervised Document Clustering

### 4.1 Article Vector Pooling
Article-level embeddings are computed by mean-pooling the passage vectors corresponding to each article:
$$\mathbf{e}_{\text{article}} = \frac{1}{C} \sum_{c=1}^{C} \mathbf{v}_c, \quad \mathbf{e}_{\text{norm}} = \frac{\mathbf{e}_{\text{article}}}{\|\mathbf{e}_{\text{article}}\|_2}$$

### 4.2 K-Means Objective Function
Partitions the 1,440 article vectors into $k$ disjoint clusters $\mathcal{S} = \{S_1, S_2, \dots, S_k\}$ by minimizing within-cluster sum of squares:
$$\arg\min_{\mathcal{S}} \sum_{i=1}^k \sum_{\mathbf{x} \in S_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$
where $\boldsymbol{\mu}_i$ is the mean centroid of cluster $S_i$.

### 4.3 Medoid & Keyword Extraction
- **Medoid Selection**:
  $$\text{medoid}_i = \arg\min_{\mathbf{x} \in S_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|_2$$
- **Cluster Keywords**: Derived by taking the mean TF-IDF feature weight across all articles in cluster $S_i$ and extracting the top 8 terms with highest average weight.

---

## 5. Grounded Retrieval-Augmented Generation (RAG)

### 5.1 Architecture & Flow
```
User Question
     │
     ▼
Query Normalization
     │
     ▼
Passage Retrieval (Top-k: 3..5)
     │
     ▼
Prompt Construction (Strict Evidence Rules)
     │
     ▼
LLM Generation / Deterministic Extractive Fallback
     │
     ▼
CitationValidator (Checks [Source X] against real candidates)
     │
     ▼
User Interface Presentation
```

### 5.2 Deterministic Local Extractive Fallback Algorithm
When external API keys are unavailable, DocuSense employs a high-precision extractive grounding fallback:
1. Extract question keywords $\mathcal{K} = \text{tokens}(\text{question}) \setminus \text{stopwords}$.
2. Segment retrieved passages into individual sentences $\{s_{i, j}\}$ where $i$ is the source index.
3. Compute keyword overlap score:
   $$\text{score}(s_{i, j}) = \frac{|\text{tokens}(s_{i, j}) \cap \mathcal{K}|}{|\mathcal{K}| + 0.1}$$
4. If $\max(\text{score}) = 0$, return an explicit **abstention notice**.
5. Otherwise, select the top 2–3 non-redundant sentences and append verified citations: `f"{sentence} [Source {i}]"`.

---

## 6. Information Retrieval Evaluation Formulations

1. **Precision@k**:
   $$\text{P@}k = \frac{\sum_{i=1}^k \mathbb{I}(\text{doc}_i \in \text{Relevant})}{k}$$
2. **Recall@k**:
   $$\text{Recall@}k = \frac{\sum_{i=1}^k \mathbb{I}(\text{doc}_i \in \text{Relevant})}{|\text{Relevant}|}$$
3. **Mean Reciprocal Rank (MRR)**:
   $$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\min \{i : \text{doc}_i \in \text{Relevant}_q\} \cup \{\infty\}}$$
4. **Normalized Discounted Cumulative Gain (nDCG@k)**:
   $$\text{DCG@}k = \sum_{i=1}^k \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$
