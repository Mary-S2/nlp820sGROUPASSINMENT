"""
DocuSense Information Retrieval & RAG Grounding Evaluation Module
Calculates standard information retrieval evaluation metrics:
  - Precision@1, Precision@3, Precision@5
  - Recall@5
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (nDCG@5)
Evaluates and compares TF-IDF and Semantic Search over the 30-query benchmark
and executes a qualitative RAG grounding review.
"""

import json
import logging
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
import pandas as pd

try:
    import config
    from src.data_loader import get_data_loader
    from src.tfidf_search import get_tfidf_search
    from src.semantic_search import get_semantic_search
    from src.rag import get_rag_assistant
except ImportError:
    from .. import config
    from .data_loader import get_data_loader
    from .tfidf_search import get_tfidf_search
    from .semantic_search import get_semantic_search
    from .rag import get_rag_assistant

logger = logging.getLogger("DocuSense.Evaluation")


def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate Precision@k: proportion of retrieved items in top-k that are relevant.
    """
    if k <= 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    num_relevant = sum(1 for doc_id in top_k if doc_id in relevant)
    return num_relevant / float(k)


def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate Recall@k: proportion of total relevant items retrieved in top-k.
    """
    if not relevant or not retrieved or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    num_relevant = sum(1 for doc_id in top_k if doc_id in relevant)
    return num_relevant / float(len(relevant))


def reciprocal_rank(retrieved: List[int], relevant: Set[int]) -> float:
    """
    Calculate Reciprocal Rank (RR): 1 / rank of the first relevant retrieved document.
    Returns 0.0 if no relevant document is retrieved.
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / float(rank)
    return 0.0


def dcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at rank k (binary relevance).
    DCG@k = sum_{i=1}^k (rel_i / log2(i + 1))
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at rank k (nDCG@k).
    nDCG@k = DCG@k / IDCG@k
    """
    if not relevant:
        return 0.0
    actual_dcg = dcg_at_k(retrieved, relevant, k)
    # Ideal DCG: all relevant items ranked at the top positions
    ideal_retrieved = list(relevant)
    ideal_dcg = dcg_at_k(ideal_retrieved, relevant, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


class EvaluationEngine:
    """
    Reproducible benchmark evaluator for DocuSense retrieval and RAG systems.
    Executes comparative retrieval evaluation and RAG grounding verification.
    """

    def __init__(self, benchmark_path: Optional[Path] = None):
        self.benchmark_path = benchmark_path or config.BENCHMARK_QUERIES_PATH
        self.benchmark_data: Dict[str, Any] = {}
        self.results_cache: Optional[Dict[str, Any]] = None

    def load_benchmark(self) -> List[Dict[str, Any]]:
        """Load the 30-query labeled benchmark set."""
        if not self.benchmark_data:
            if not self.benchmark_path.exists():
                raise FileNotFoundError(f"Benchmark file not found: {self.benchmark_path}")
            with open(self.benchmark_path, "r", encoding="utf-8") as f:
                self.benchmark_data = json.load(f)
        return self.benchmark_data.get("queries", [])

    def evaluate_search_engine(
        self,
        search_fn,
        method_name: str,
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluate a single retrieval method over all benchmark queries.
        """
        queries = self.load_benchmark()
        per_query_results = []

        p1_list, p3_list, p5_list = [], [], []
        r5_list, mrr_list, ndcg5_list = [], [], []

        cat_breakdown: Dict[str, Dict[str, List[float]]] = {}
        type_breakdown: Dict[str, Dict[str, List[float]]] = {}

        for q_item in queries:
            qid = q_item["query_id"]
            query_str = q_item["query"]
            cat = q_item["category"]
            qtype = q_item["query_type"]
            relevant_set = set(q_item["relevant_article_ids"])

            # Execute search
            raw_hits = search_fn(query_str, k=k)
            # Map chunk hits to article IDs (preserve rank order, remove consecutive duplicate articles)
            retrieved_article_ids = []
            for h in raw_hits:
                aid = h.get("article_id")
                if aid is not None and (not retrieved_article_ids or retrieved_article_ids[-1] != aid):
                    retrieved_article_ids.append(aid)

            # Compute metrics
            p1 = precision_at_k(retrieved_article_ids, relevant_set, 1)
            p3 = precision_at_k(retrieved_article_ids, relevant_set, 3)
            p5 = precision_at_k(retrieved_article_ids, relevant_set, 5)
            r5 = recall_at_k(retrieved_article_ids, relevant_set, 5)
            rr = reciprocal_rank(retrieved_article_ids, relevant_set)
            ndcg = ndcg_at_k(retrieved_article_ids, relevant_set, 5)

            p1_list.append(p1)
            p3_list.append(p3)
            p5_list.append(p5)
            r5_list.append(r5)
            mrr_list.append(rr)
            ndcg5_list.append(ndcg)

            # Record per-query details
            per_query_results.append({
                "query_id": qid,
                "query": query_str,
                "category": cat,
                "query_type": qtype,
                "relevant_ids": list(relevant_set),
                "retrieved_article_ids": retrieved_article_ids[:5],
                "p@1": round(p1, 3),
                "p@3": round(p3, 3),
                "p@5": round(p5, 3),
                "recall@5": round(r5, 3),
                "rr": round(rr, 3),
                "ndcg@5": round(ndcg, 3),
                "hit_at_5": (r5 > 0.0),
            })

            # Accumulate category breakdown
            if cat not in cat_breakdown:
                cat_breakdown[cat] = {"p@5": [], "recall@5": [], "mrr": []}
            cat_breakdown[cat]["p@5"].append(p5)
            cat_breakdown[cat]["recall@5"].append(r5)
            cat_breakdown[cat]["mrr"].append(rr)

            # Accumulate query type breakdown
            if qtype not in type_breakdown:
                type_breakdown[qtype] = {"p@5": [], "recall@5": [], "mrr": []}
            type_breakdown[qtype]["p@5"].append(p5)
            type_breakdown[qtype]["recall@5"].append(r5)
            type_breakdown[qtype]["mrr"].append(rr)

        # Macro-averages
        summary = {
            "method": method_name,
            "total_queries": len(queries),
            "mean_precision@1": round(float(np.mean(p1_list)), 4),
            "mean_precision@3": round(float(np.mean(p3_list)), 4),
            "mean_precision@5": round(float(np.mean(p5_list)), 4),
            "mean_recall@5": round(float(np.mean(r5_list)), 4),
            "mrr": round(float(np.mean(mrr_list)), 4),
            "mean_ndcg@5": round(float(np.mean(ndcg5_list)), 4),
            "category_metrics": {
                c: {
                    "mean_p@5": round(float(np.mean(v["p@5"])), 3),
                    "mean_recall@5": round(float(np.mean(v["recall@5"])), 3),
                    "mean_mrr": round(float(np.mean(v["mrr"])), 3),
                }
                for c, v in cat_breakdown.items()
            },
            "type_metrics": {
                t: {
                    "mean_p@5": round(float(np.mean(v["p@5"])), 3),
                    "mean_recall@5": round(float(np.mean(v["recall@5"])), 3),
                    "mean_mrr": round(float(np.mean(v["mrr"])), 3),
                }
                for t, v in type_breakdown.items()
            },
            "per_query_details": per_query_results,
        }

        return summary

    def run_comparative_evaluation(self, k: int = 5) -> Dict[str, Any]:
        """
        Run both TF-IDF and Semantic Search on the exact same 30 benchmark queries.
        Returns combined metrics and side-by-side comparison.
        """
        logger.info("Running TF-IDF evaluation on benchmark...")
        tfidf_searcher = get_tfidf_search()
        tfidf_summary = self.evaluate_search_engine(
            search_fn=lambda q, k: tfidf_searcher.search(q, k=k),
            method_name="TF-IDF (Lexical Baseline)",
            k=k,
        )

        logger.info("Running Semantic Search evaluation on benchmark...")
        semantic_searcher = get_semantic_search()
        semantic_summary = self.evaluate_search_engine(
            search_fn=lambda q, k: semantic_searcher.search(q, k=k, explain_sentences=False),
            method_name="Semantic Embeddings (all-MiniLM-L6-v2 + FAISS)",
            k=k,
        )

        combined = {
            "benchmark_info": {
                "name": self.benchmark_data.get("description", "DocuSense Benchmark"),
                "num_queries": len(self.load_benchmark()),
                "k": k,
            },
            "tfidf_metrics": tfidf_summary,
            "semantic_metrics": semantic_summary,
            "metric_comparison_table": [
                {
                    "Metric": "Precision@1",
                    "TF-IDF": tfidf_summary["mean_precision@1"],
                    "Semantic": semantic_summary["mean_precision@1"],
                    "Delta": round(semantic_summary["mean_precision@1"] - tfidf_summary["mean_precision@1"], 4),
                },
                {
                    "Metric": "Precision@3",
                    "TF-IDF": tfidf_summary["mean_precision@3"],
                    "Semantic": semantic_summary["mean_precision@3"],
                    "Delta": round(semantic_summary["mean_precision@3"] - tfidf_summary["mean_precision@3"], 4),
                },
                {
                    "Metric": "Precision@5",
                    "TF-IDF": tfidf_summary["mean_precision@5"],
                    "Semantic": semantic_summary["mean_precision@5"],
                    "Delta": round(semantic_summary["mean_precision@5"] - tfidf_summary["mean_precision@5"], 4),
                },
                {
                    "Metric": "Recall@5",
                    "TF-IDF": tfidf_summary["mean_recall@5"],
                    "Semantic": semantic_summary["mean_recall@5"],
                    "Delta": round(semantic_summary["mean_recall@5"] - tfidf_summary["mean_recall@5"], 4),
                },
                {
                    "Metric": "MRR (Mean Reciprocal Rank)",
                    "TF-IDF": tfidf_summary["mrr"],
                    "Semantic": semantic_summary["mrr"],
                    "Delta": round(semantic_summary["mrr"] - tfidf_summary["mrr"], 4),
                },
                {
                    "Metric": "nDCG@5",
                    "TF-IDF": tfidf_summary["mean_ndcg@5"],
                    "Semantic": semantic_summary["mean_ndcg@5"],
                    "Delta": round(semantic_summary["mean_ndcg@5"] - tfidf_summary["mean_ndcg@5"], 4),
                },
            ],
        }

        # Cache results to disk
        out_file = config.OUTPUTS_DIR / "evaluation_results.json"
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2)
            logger.info(f"Saved evaluation benchmark results to {out_file}")
        except Exception as e:
            logger.warning(f"Could not save evaluation results: {e}")

        self.results_cache = combined
        return combined

    def run_rag_grounding_review(self) -> List[Dict[str, Any]]:
        """
        Execute qualitative grounding review on representative test questions:
          - Factual support check
          - Citation validity verification
          - Unsupported claims check
          - Abstention behavior verification on out-of-domain question
        """
        rag_assistant = get_rag_assistant()
        review_questions = [
            {
                "id": "RAG-Q1",
                "question": "What did Arsene Wenger agree to regarding his contract at Arsenal?",
                "expected_topic": "Arsene Wenger signing a new deal until May 2008",
                "is_answerable": True,
            },
            {
                "id": "RAG-Q2",
                "question": "What tools did Microsoft release to combat computer viruses and spyware?",
                "expected_topic": "Tools cleaning up PCs harbouring malicious spyware and viruses",
                "is_answerable": True,
            },
            {
                "id": "RAG-Q3",
                "question": "Why did Conservative leader Michael Howard admit to playing truant at school?",
                "expected_topic": "Playing truant to go to a snooker hall",
                "is_answerable": True,
            },
            {
                "id": "RAG-Q4",
                "question": "Who won the FIFA World Cup 2022 in Qatar?",
                "expected_topic": "Unanswerable (Dataset ends in 2005 - must abstain)",
                "is_answerable": False,
            },
        ]

        review_records = []
        for item in review_questions:
            q = item["question"]
            res = rag_assistant.answer_question(q, retrieval_method="embeddings", k=3)
            
            # Grounding assessment
            answer_text = res["answer"]
            citations = res["citations"]
            is_abstention = res["is_abstention"]
            
            if not item["is_answerable"]:
                grounding_verdict = "PASSED (Appropriate Abstention)" if is_abstention else "FAILED (Hallucinated out-of-domain knowledge)"
            else:
                has_citations = citations["citation_count"] > 0
                no_hallucinations = not citations["hallucination_detected"]
                grounding_verdict = "PASSED (Grounded with verified citations)" if (has_citations and no_hallucinations) else "NEEDS REVIEW"

            review_records.append({
                "test_id": item["id"],
                "question": q,
                "expected_topic": item["expected_topic"],
                "is_answerable": item["is_answerable"],
                "provider_used": res["provider"],
                "generated_answer": answer_text,
                "citations_verified": citations["valid_citations"],
                "citations_invalid": citations["invalid_citations"],
                "abstained": is_abstention,
                "verdict": grounding_verdict,
            })

        return review_records


_eval_engine: Optional[EvaluationEngine] = None


def get_evaluation_engine() -> EvaluationEngine:
    """Singleton getter for EvaluationEngine."""
    global _eval_engine
    if _eval_engine is None:
        _eval_engine = EvaluationEngine()
    return _eval_engine
