# DocuSense

Semantic search, topic discovery and question answering with source passages over a BBC News collection.

NAT820S: Natural Language Processing and Applications — Group Project 2, Namibia University of Science and Technology.

- **Hosted interface:** [Open DocuSense](https://huggingface.co/spaces/MaryJS/Docusense)
- **Final project notebook:** [DocusenseFinal1,.ipynb](DocusenseFinal1%2C.ipynb)
- **Open the notebook:** [Google Colab](https://colab.research.google.com/github/Mary-S2/nlp820sGROUPASSINMENT/blob/main/DocusenseFinal1%2C.ipynb)

## Which version to use

This README documents the final BERTopic notebook and its adapted Hugging Face interface.

| Location | Purpose |
| --- | --- |
| `DocusenseFinal1,.ipynb` | Latest uploaded Colab notebook: final pipeline, BERTopic analysis, saved development evaluations and review tools enabled. The comma is part of the uploaded filename. |
| `DocusenseFinal1.ipynb` | The same saved analysis with the optional review tab hidden by default. |
| [Hugging Face Space](https://huggingface.co/spaces/MaryJS/Docusense) | Standalone deployment of Search, Ask a question and Review retrieval. Its deployment files are in the Space's [Files tab](https://huggingface.co/spaces/MaryJS/Docusense/tree/main). |
| [`DocuSense/`](DocuSense/) | A separate modular app implementation with its own [README](DocuSense/README.md), requirements and tests. Its K-means clustering, interface and benchmark differ from the final notebook documented here. |
| `notebooks/Docusensework.ipynb`, `NLP.ipynb` | Additional project notebooks. |
| `data/` | Saved passage and embedding files. Embeddings must match the passage text and row order used to create them. |

## What the final system does

- Removes duplicate articles, normalizes whitespace and splits text into overlapping passages while retaining source identifiers.
- Offers **keyword search** using TF-IDF and **semantic search** using MiniLM embeddings with a FAISS index.
- Applies corpus-aware spelling correction and noun lemmatization to search queries.
- Explains keyword results with matched terms and semantic results with a similar sentence from the passage.
- Uses **BERTopic** to discover article groups, display topic keywords and browse sample articles in the notebook.
- Answers questions using retrieved and reranked passages, attaches source identifiers and can abstain when relevance and answerability checks fail.
- Collects article relevance judgments and reasons, then calculates precision and recall from completed judgments.

Topic analysis is performed in the notebook. The hosted interface has three tabs: **Search**, **Ask a question** and **Review retrieval**.

## Dataset and preprocessing

The prototype uses the [BBC News Classification training dataset on Kaggle](https://www.kaggle.com/competitions/learn-ai-bbc/data), supplied as `BBC News Train.csv`.

| Item | Saved notebook run |
| --- | ---: |
| Input articles | 1,490 |
| Duplicate texts removed | 50 |
| Retained articles | 1,440 |
| Overlapping passages | 4,981 |
| Maximum passage length | 150 words |
| Passage overlap | 30 words |
| Embedding dimensions | 384 |

Required columns are `ArticleId`, `Text` and `Category`. The five source categories are business, entertainment, politics, sport and tech. Missing or empty text is removed before deduplication. Passage records retain `ArticleId`, `Category`, `chunk_id` and `chunk_text`.

Category agreement is a diagnostic experiment; an article's category alone does not establish that it answers a particular query.

## Models and component roles

| Component | Role in the final pipeline |
| --- | --- |
| TF-IDF | Keyword baseline with cosine similarity; English stop words, `min_df=2`, `max_df=0.8`. |
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Encodes passages and queries into normalized 384-dimensional vectors. |
| FAISS `IndexFlatIP` | Searches the normalized passage vectors using inner products, equivalent to cosine similarity for unit vectors. |
| [BERTopic](https://maartengr.github.io/BERTopic/) | Discovers topics from whole article texts with MiniLM embeddings and `min_topic_size=15`; the topic count is not fixed in advance. |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Reranks question–passage candidates retrieved by both search methods. |
| `deepset/minilm-uncased-squad2` | Extracts an answer span and compares its score with a no-answer score. |
| `google/flan-t5-base` | Attempts a short answer using the selected evidence span. |

For question answering, both retrievers contribute candidates. The cross-encoder selects the top passages, and the reader checks for an answer span. FLAN-T5's phrase is accepted only if its normalized text occurs in that span; otherwise the extracted span is returned. Source labels are added in Python using the article and chunk identifiers. If the evidence checks fail, the system returns:

> I could not find a supported answer in the retrieved passages.

Question answering keeps the original question by default. It reads retrieved **passages**, rather than automatically supplying every full parent article. Literal phrase containment checks provenance; it does not guarantee that an answer is semantically correct.

## Run the notebook in Google Colab

1. Open [the final notebook in Colab](https://colab.research.google.com/github/Mary-S2/nlp820sGROUPASSINMENT/blob/main/DocusenseFinal1%2C.ipynb) and save a working copy.
2. Use a fresh Python runtime and run the cells in order. The notebook contains its package installation cells; internet access is needed for packages and pretrained models.
3. At the upload prompt, select exactly one CSV, or one ZIP containing a single CSV, with the required dataset columns.
4. Continue through data preparation, retrieval, topic modelling and evaluation. The final cell starts the Gradio interface.
5. The latest `DocusenseFinal1,.ipynb` enables **Review retrieval**, as does the hosted Space. If using the other copy, `DocusenseFinal1.ipynb`, set `SHOW_REVIEW_TOOLS = True` in its final interface cell and rerun that cell to display the optional tab.

The notebook and the separate `DocuSense/` app have different setup paths. `DocuSense/requirements.txt` is for that app; it is not a complete environment specification for the final BERTopic notebook. The notebook's installation cells are not version-pinned, so dependency changes can affect a fresh run.

## Use the hosted interface

Open the [DocuSense Space](https://huggingface.co/spaces/MaryJS/Docusense) in a browser. Once the Space is running, visitors do not need Colab or a local Python installation. Hugging Face runs the server independently of the notebook. ZeroGPU requests can queue and are subject to usage limits.

- **Search:** enter a query, select keyword or semantic search, choose the number of results and inspect the source passage and explanation. An existing example is `wenegr arsnal contrct`.
- **Ask a question:** try `What did Arsene Wenger agree to?`, then inspect the cited article/chunk and supporting passage. A saved example returns `a long-term contract [1]` from ArticleId 1588.
- **Review retrieval:** read each query and candidate article, choose Relevant or Not relevant, add a reason and save the judgment. Calculate scores after every candidate has a binary label. Successful calculation enables downloads of per-query results and the summary.

The deployment uses the same four pretrained models and retrieval/RAG functions, with model preloading and GPU callback registration for ZeroGPU. No external generation API key is required. Review judgments are shared server data; download the current CSV before restarting or rebuilding because ordinary runtime storage is temporary.

## Saved development results

These figures describe the saved final-notebook run. They are development checks, not an independent final benchmark.

### Distinct-article retrieval

Fifteen inspected queries each have one seed article marked relevant. Results are deduplicated by `ArticleId` before evaluation at five articles.

| Method | Macro precision@5 | Macro recall@5 | Queries retrieving the seed article |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.200 | 1.000 | 15/15 |
| MiniLM + FAISS | 0.173 | 0.867 | 13/15 |

With only one labelled positive per query, precision@5 is capped at 0.200 and recall@5 acts as a seed-article hit rate. Other returned articles may also be relevant. These scores do not establish that one method is generally better.

A separate pooled relevance file has 110 candidate query/article pairs. The last supplied file had 2 completed labels; it does not support final reviewed precision/recall yet. Labels entered later in the app must be downloaded and evaluated separately.

### RAG checks

The saved automatic checks show **6/7 expected phrase matches** on answerable questions and **3/3 expected refusals** on unanswerable questions. Phrase matching is not answer accuracy or proof of citation support. Seven answer/citation judgments remain pending in the saved manual-review table. One observed failure gives Luxembourg for a question about the UK's EU rebate, illustrating that a copied source phrase can still answer the question incorrectly.

### Topic discovery

The saved BERTopic run contains **9 topics and 20 outlier articles**, with topic IDs `0`–`8` and outlier ID `-1`. Topic counts are 298, 273, 243, 230, 230, 59, 44, 22 and 21; these sum to 1,420 assigned articles. The model prints keyword-based names and sample articles. A rerun can produce different clusters because the notebook does not fix a dimensionality-reduction seed. The discovered topics are not the five supplied BBC category labels.

## Limitations and further validation

- The collection is historical English news from one outlet; current events, other languages and Namibian document collections have not been validated.
- Spelling correction can alter intended meaning: the saved example changes `contrct` to `contact`.
- Retrieval labels and human answer/citation judgments are incomplete, and the small query sets were inspected during development.
- Reader margins and reranker cutoffs are provisional scores, not calibrated confidence probabilities. Citations and refusal checks reduce some failures but do not guarantee correctness.
- Topic keywords need human interpretation. Human-readable topic labels still need to be documented for the final submission, and the number of discovered topics can change between runs.
- Another dataset requires schema checks, rebuilt chunks/embeddings/indexes and new source-specific evaluation questions and judgments. Saved BBC vectors and article IDs cannot be reused for unrelated text.

## Team and submission documentation

- Gift Mubita
- Romario Arnat
- Maria Shikwambi
- Adam Noa

AI assistance was used during development and debugging. The final report includes the project's details.
