# DocuSense: User & Application Operation Guide

**Project:** DocuSense — Semantic Search, Similarity Index & RAG Assistant for a Document Collection  
**Course:** NAT820S Natural Language Processing and Applications  
**Institution:** Namibia University of Science and Technology (NUST)  

---

## 1. Introduction

DocuSense is an interactive research and information retrieval application built on the BBC News collection (1,440 articles, 4,981 passages). This guide explains how to launch, navigate, and utilize each tab of the DocuSense interface.

---

## 2. Quickstart Launch Instructions

### 2.1 Local Execution
```bash
# Ensure your virtual environment is active
python app.py
```
Open your web browser and navigate to `http://localhost:7860`.

### 2.2 Public Sharing Mode (for Colab / Remote Demos)
```bash
python app.py --share
```
This generates a temporary public link (e.g. `https://xxxx.gradio.live`) valid for 72 hours.

---

## 3. Navigating the 7 Application Tabs

### Tab 1: 📊 Collection Overview
- **Purpose**: Displays live ground-truth metrics calculated directly from the indexed dataset.
- **Key Metrics Displayed**:
  - **1,440 Loaded Articles**: Deduplicated BBC news collection.
  - **4,981 Indexed Passages**: 150-word sliding-window chunks with 30-word overlap.
  - **5 Categories**: Exact distribution (`sport`: 342, `business`: 335, `politics`: 274, `entertainment`: 271, `tech`: 260).
  - **TF-IDF Vocabulary Size**: Fitted n-gram feature count (~14,000+ unigrams and bigrams).
  - **Vector Dimension**: 384 dimensions (`all-MiniLM-L6-v2`).

---

### Tab 2: 🔍 Search & Comparison
- **Purpose**: Query the collection using classical lexical search (TF-IDF), dense semantic search, or side-by-side comparative mode.
- **How to Use**:
  1. Enter your search query in the text box (e.g., *"cost of borrowing money and inflation impact"* or *"worldcom fraud trial bernie ebbers"*).
  2. Select your **Retrieval Paradigm**:
     - `Semantic Search (Dense Embeddings)`: Uses `all-MiniLM-L6-v2` and FAISS vector inner product.
     - `TF-IDF (Lexical)`: Uses classical n-gram term frequency and cosine similarity.
     - `Side-by-Side Comparison`: Runs both paradigms simultaneously on the exact same query.
  3. Adjust the **Top-k Passages** slider ($1 \dots 10$).
  4. Click **Search** or press `Enter`.
- **Interpreting Results**:
  - **Query Understanding Audit**: Shows the normalized query and explains whether casing/punctuation was preserved or stripped.
  - **Results Tables**: Displays Rank, ArticleId, Category, Cosine Score, and Excerpt.
  - **Why It Ranked**: For TF-IDF, shows exact matched query terms; for Semantic Search, extracts the single most informative sentence inside the passage.
  - **Comparative Analysis**: Calculates Jaccard similarity between the top-k result sets and provides a score calibration notice.

---

### Tab 3: 🤖 Ask DocuSense (RAG Assistant)
- **Purpose**: Ask natural-language questions and receive grounded answers backed by genuine source citations.
- **How to Use**:
  1. Type a factual question into the text box (e.g., *"What did Arsene Wenger agree to regarding his contract at Arsenal?"*).
  2. Choose your **Passage Retriever** (`Semantic Embeddings (Recommended)` or `TF-IDF`).
  3. Select the number of evidence passages to retrieve ($2 \dots 6$, default 3).
  4. Click **Generate Grounded Answer**.
- **Understanding RAG Output**:
  - **Grounded Answer**: Concise factual answer with bracketed citations like `[Source 1]`.
  - **Citation Audit Badge**:
    - `✅ Status: VERIFIED`: All citations match genuine candidate passages in the retrieved set.
    - `⚠️ Status: ABSTAINED`: Question is out-of-domain or the corpus lacks evidence; the model refused to fabricate facts.
  - **Accordion (Supporting Evidence)**: Click *"View Supporting Evidence Passages & Source Citations"* to read the full verbatim passages and inspect the source ArticleIds and similarity scores.

---

### Tab 4: 🧭 Explore Topics & Clusters
- **Purpose**: Unsupervised thematic exploration using K-Means clustering over dense document embeddings.
- **Controls**:
  - **Number of Discovered Topics Slider ($k=2 \dots 15$)**: Set any arbitrary cluster count and click **Recompute Clusters**.
  - **Preset Buttons**:
    - **Macro View ($k=5$)**: Matches the 5 official BBC categories and displays category purity percentages.
    - **Granular View ($k=12$)**: Breaks general topics into specialized news sub-themes.
- **Cluster Browsing**:
  - Select any topic from the dropdown menu and click **Browse Cluster Articles** to view the top 25 articles assigned to that cluster, including headlines and text previews.

---

### Tab 5: 🔗 Article Similarity Explorer
- **Purpose**: Document-to-document recommendation based on global 384-dimensional cosine similarity.
- **How to Use**:
  1. Select an article from the dropdown list.
  2. Choose the number of similar articles to retrieve ($1 \dots 10$).
  3. Click **Find Similar Articles**.
  4. The table displays ranked similar documents with similarity scores, categories, headlines, and text snippets.

---

### Tab 6: 📈 Evaluation & Benchmark
- **Purpose**: Review quantitative performance measured across the 30-query benchmark.
- **Features**:
  - Metric comparison table displaying **Precision@1, Precision@3, Precision@5, Recall@5, MRR, and nDCG@5**.
  - Mathematical formulations and explanations for each metric.
  - Category-by-category and query-type breakdown analysis.

---

### Tab 7: 📖 Methodology & Responsible AI
- **Purpose**: Complete technical reference covering pipeline design, chunking hyperparameters, responsible AI considerations, temporal limits (2004–2005), and ethical guidelines.
