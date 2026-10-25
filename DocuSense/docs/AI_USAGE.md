# NUST Official Student AI Declaration & Usage Reflection

**Course:** NAT820S Natural Language Processing and Applications (2026 Semester 2)  
**Lecturer:** Mr. Naftali N. Indongo (`nindongo@nust.na`)  
**Department:** Computer Science, Faculty of Computing and Informatics  
**Institution:** Namibia University of Science and Technology (NUST), Windhoek, Namibia  

**Project Title:** DocuSense: Semantic Search, Similarity Index & RAG Assistant for a Document Collection  
**Group Members:**
1. **Gift Mubita** (Student Number: `222009632`)
2. **Romario Arnat** (Student Number: `221011056`)
3. **Maria Shikwambi** (Student Number: `222064374`)
4. **Adam Noa** (Student Number: `214018687`)

---

## 1. Official Student AI Declaration Form

In accordance with NUST's **Artificial Intelligence Policy** and the **Guidelines for the Artificial Intelligence Policy**, our group declares the following use of AI tools during the conception, development, testing, and documentation of the DocuSense prototype:

### Tools Consulted:
- **Google Antigravity IDE / Gemini 3.8 Flash**: Code scaffolding, interactive session debugging, and metric calculation scripts.
- **Anthropic Claude**: Architecture documentation drafting, test case design, and README structuring.

### Categories of AI Assistance (Ticked / Applied):
- [x] **Idea Generation & Brainstorming**: Exploring UI layout paradigms and parameter options for sliding-window chunking.
- [x] **Methods & Technical Design**: Designing the multi-provider RAG architecture and FAISS fallback mechanisms.
- [x] **Code Development & Scaffolding**: Drafting baseline functions, Gradio Blocks layout, and pytest test structures.
- [x] **Data Analysis & Evaluation Scripting**: Automating metric calculations (Precision@k, Recall@5, MRR, nDCG@5) across benchmark queries.
- [x] **Documentation Editing & Formatting**: Markdown formatting, Mermaid diagram generation, and LaTeX equation typesetting.
- [ ] **Data Fabrication**: **NONE**. All 1,440 documents, 4,981 passages, and 30 benchmark relevance judgements are authentic and strictly based on the real BBC News dataset.
- [ ] **Citation Fabrication**: **NONE**. Every source citation was mathematically audited against candidate passages.

---

## 2. Required Reflection: "How We Used AI Tools"

### 2.1 Tools Utilized and Scope of Use
Throughout the semester-long build of DocuSense, our team utilized AI coding and writing assistants in a deliberate, disciplined manner:
1. **Scaffolding and Interface Layout**: We utilized AI to generate initial boilerplate code for our Gradio Blocks user interface (styling, CSS classes, tab arrangements) and to assist in writing repetitive data manipulation routines in pandas.
2. **Evaluation Scripting**: We used AI assistance to implement the mathematical formulas for standard Information Retrieval metrics (DCG@k, nDCG@k, MRR) according to standard academic definitions (Manning et al., *Introduction to Information Retrieval*).
3. **Drafting Documentation**: AI tools assisted in organizing technical Markdown files and formatting Mermaid architecture flowcharts.

### 2.2 Instance of Overriding and Rejecting an AI Recommendation
A concrete instance where our team explicitly **rejected and overrode** an AI tool's recommendation occurred during the initial design of our text preprocessing pipeline:
- **AI Suggestion**: The assistant initially proposed applying an aggressive preprocessing routine across *all* models: lowercasing, stripping all punctuation, removing all stopwords, and applying Porter stemming to both queries and passages prior to feeding them into `sentence-transformers/all-MiniLM-L6-v2`.
- **Why We Rejected It**:
  1. Transformer models rely fundamentally on **bidirectional self-attention mechanisms** across word pieces. Capitalization is essential for identifying Named Entities (e.g. *"Michael Howard"* vs. *"michael toward"*, or *"Arsenal"* vs. *"arsenal"*).
  2. Punctuation (commas, periods, question marks) demarcates syntactic clause boundaries, which transformers use to resolve semantic dependencies and negations.
  3. Stemming truncates words (e.g. *"banking"* $\to$ *"bank"*, *"inflationary"* $\to$ *"inflat"*), which corrupts the subword vocabulary of MiniLM.
- **Our Resolution**: We rejected the unified aggressive preprocessing proposal and designed **decoupled, differentiated pipelines**:
  - *Lexical Pipeline (TF-IDF)*: Applies lowercasing, punctuation stripping, and optional WordNet lemmatization.
  - *Semantic Pipeline (MiniLM)*: Preserves full casing, punctuation, numbers, and grammatical sentence syntax.
  - When tested on Query Q19 (*"wenegr arsnal contrct"*), this decision directly preserved entity recognition and improved semantic ranking accuracy.

### 2.3 Verification Against Ground Truth and Empirical Test Sets
To ensure compliance with Rule 3 (*"Verify, don't trust blindly"*):
1. **Dataset Integrity**: We independently verified that the 1,440 articles in `data/bbc_articles.csv` matched the raw BBC news texts, and manually confirmed the removal of 50 exact duplicate articles.
2. **Benchmark Relevance Judgements**: For our 30-query benchmark (`data/labeled_benchmark_queries.json`), every relevant document ID was manually read and verified by group members to ensure that the assigned binary relevance ($rel=1$) was factually accurate.
3. **Metric Formulas**: We verified the calculations of Precision@k, Recall@k, MRR, and nDCG@k using synthetic, hand-calculated test vectors (implemented in `tests/test_rag_and_eval.py`) before trusting the automated evaluation script.
4. **Citation Auditing**: For every RAG generation test, our `CitationValidator` automatically checked whether generated `[Source X]` tags mapped to real candidate passages in the top-$k$ retrieved list.

---

## 3. Team Member Sign-off & Defense Readiness

Each member of our group has reviewed the entire codebase, understands every module, and is prepared to defend any line of code, modelling choice, or evaluation metric during the oral demonstration:

- **Gift Mubita** (222009632) — *Data Preprocessing, Chunking & Deduplication*
- **Romario Arnat** (221011056) — *Classical TF-IDF Retrieval & Lexical Overlap Explanations*
- **Maria Shikwambi** (222064374) — *Semantic Embeddings, FAISS Vector Indexing & Clustering*
- **Adam Noa** (214018687) — *Grounded RAG Assistant, Citation Validation & Gradio Interface*
