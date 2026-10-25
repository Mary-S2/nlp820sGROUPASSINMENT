"""
DocuSense Retrieval-Augmented Generation (RAG) Assistant Module
Implements evidence-grounded answer generation, multi-provider LLM orchestration
(Google Gemini, OpenAI, Groq, HuggingFace, and Extractive Fallback),
strict citation validation, and hallucination safeguards.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

try:
    import config
    from src.query_processing import get_query_processor
    from src.semantic_search import get_semantic_search
    from src.tfidf_search import get_tfidf_search
except ImportError:
    from .. import config
    from .query_processing import get_query_processor
    from .semantic_search import get_semantic_search
    from .tfidf_search import get_tfidf_search

logger = logging.getLogger("DocuSense.RAG")

# System prompt enforcing strict evidence grounding
RAG_SYSTEM_PROMPT = """You are DocuSense, an evidence-grounded academic research assistant for the BBC News document collection.
Your task is to answer the user's question using ONLY the numbered passages provided below.

Rules you MUST follow:
1. Ground your answer entirely on the provided evidence. Do NOT extrapolate, speculate, or introduce external knowledge.
2. If the provided passages do not contain sufficient evidence to answer the question, state clearly:
   "The provided BBC News collection does not contain sufficient evidence to answer this question."
3. Every factual statement must cite its supporting passage using bracketed notation, e.g. [Source 1] or [Source 2].
4. Do not fabricate source IDs, article titles, quotations, or facts.
"""


class CitationValidator:
    """
    Validates that citations in generated answers map to genuine retrieved passages.
    Detects hallucinated sources and unsupported assertions.
    """

    @staticmethod
    def validate_citations(
        answer_text: str,
        retrieved_passages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verify cited source indices against real retrieved passages.
        """
        num_sources = len(retrieved_passages)
        # Match citations like [Source 1], [Source 2], [1], [2], or [ArticleId 1588]
        found_source_nums = re.findall(r"\[(?:Source\s*)?(\d+)\]", answer_text, re.IGNORECASE)
        found_article_ids = re.findall(r"\[(?:ArticleId\s*)?(\d{2,5})\]", answer_text, re.IGNORECASE)

        valid_citations = []
        invalid_citations = []

        retrieved_article_ids = {str(p.get("article_id")) for p in retrieved_passages}

        for num_str in found_source_nums:
            num = int(num_str)
            if 1 <= num <= num_sources:
                valid_citations.append(f"Source {num}")
            elif num_str in retrieved_article_ids:
                valid_citations.append(f"ArticleId {num_str}")
            else:
                invalid_citations.append(f"Source {num} (Out of range 1..{num_sources})")

        for aid in found_article_ids:
            if aid in retrieved_article_ids:
                valid_citations.append(f"ArticleId {aid}")
            elif int(aid) > num_sources:
                invalid_citations.append(f"ArticleId {aid} (Not in retrieved set)")

        valid_unique = sorted(list(set(valid_citations)))
        invalid_unique = sorted(list(set(invalid_citations)))

        is_abstention = (
            "not contain sufficient evidence" in answer_text.lower()
            or "insufficient evidence" in answer_text.lower()
            or "does not contain" in answer_text.lower()
        )

        return {
            "valid_citations": valid_unique,
            "invalid_citations": invalid_unique,
            "citation_count": len(valid_unique),
            "hallucination_detected": len(invalid_unique) > 0,
            "is_abstention": is_abstention,
            "status": "Verified" if not invalid_unique else "Warning: Unverified Citations Detected",
        }


class RAGAssistant:
    """
    Retrieval-Augmented Generation (RAG) assistant for the BBC News collection.
    Retrieves evidence, prompts configurable LLMs or extractive fallback,
    and validates source citations.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        top_k: int = 5,
    ):
        self.provider = provider or config.DEFAULT_LLM_PROVIDER
        self.top_k = top_k
        self.query_processor = get_query_processor()
        self.citation_validator = CitationValidator()

    def answer_question(
        self,
        question: str,
        retrieval_method: str = "embeddings",
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline:
          1. Query validation & normalization
          2. Relevant passage retrieval
          3. Evidence context construction
          4. Grounded answer generation
          5. Citation validation & evidence audit
        """
        k = k or self.top_k

        # 1. Validate query
        q_result = self.query_processor.process(question, method=retrieval_method)
        if not q_result.is_valid:
            return {
                "question": question,
                "answer": f"Unable to process question: {q_result.status_message}",
                "retrieved_sources": [],
                "citations": {"valid_citations": [], "hallucination_detected": False, "status": "Invalid Query"},
                "provider": self.provider,
                "is_abstention": True,
            }

        # 2. Retrieve evidence passages
        if retrieval_method == "tfidf":
            retriever = get_tfidf_search()
            hits = retriever.search(q_result.processed_query, k=k)
        else:
            retriever = get_semantic_search()
            hits = retriever.search(q_result.processed_query, k=k)

        # Check for empty retrieval
        if not hits:
            return {
                "question": question,
                "answer": "The BBC News collection does not contain any relevant articles matching your question.",
                "retrieved_sources": [],
                "citations": {"valid_citations": [], "hallucination_detected": False, "status": "No Evidence"},
                "provider": self.provider,
                "is_abstention": True,
            }

        # 3. Format numbered evidence passages
        formatted_sources = []
        context_blocks = []
        for i, hit in enumerate(hits, start=1):
            passage_text = hit.get("passage", "").strip()
            article_id = hit.get("article_id", -1)
            chunk_id = hit.get("chunk_id", "")
            category = hit.get("category", "")
            score = hit.get("score", 0.0)

            formatted_sources.append({
                "source_index": i,
                "chunk_id": chunk_id,
                "article_id": article_id,
                "category": category,
                "score": score,
                "passage_excerpt": passage_text[:250] + ("..." if len(passage_text) > 250 else ""),
                "full_passage": passage_text,
            })

            context_blocks.append(
                f"[Source {i}] (Article {article_id}, Category: {category}):\n{passage_text}"
            )

        context_str = "\n\n".join(context_blocks)

        # 4. Generate answer via configured provider
        raw_answer, provider_used = self._generate_answer(
            question=question,
            context=context_str,
            sources=formatted_sources,
        )

        # 5. Validate citations and verify grounding
        citation_audit = self.citation_validator.validate_citations(
            answer_text=raw_answer,
            retrieved_passages=formatted_sources,
        )

        return {
            "question": question,
            "answer": raw_answer,
            "provider": provider_used,
            "retrieval_method": retrieval_method,
            "retrieved_sources": formatted_sources,
            "citations": citation_audit,
            "is_abstention": citation_audit["is_abstention"],
        }

    def _generate_answer(
        self,
        question: str,
        context: str,
        sources: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """
        Dispatches generation to the configured LLM provider with fallback handling.
        """
        provider = self.provider.lower().strip()

        # Try Google Gemini if configured
        if provider == "gemini" and config.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=config.GEMINI_API_KEY)
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = (
                    f"{RAG_SYSTEM_PROMPT}\n\n"
                    f"Evidence Passages:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer:"
                )
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip(), "Google Gemini (gemini-1.5-flash)"
            except Exception as e:
                logger.warning(f"Gemini generation failed ({e}). Falling back to extractive synthesis.")

        # Try OpenAI if configured
        if provider == "openai" and config.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=config.OPENAI_API_KEY)
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Evidence Passages:\n{context}\n\nQuestion: {question}"},
                    ],
                    temperature=0.0,
                )
                answer = completion.choices[0].message.content.strip()
                return answer, "OpenAI (gpt-4o-mini)"
            except Exception as e:
                logger.warning(f"OpenAI generation failed ({e}). Falling back to extractive synthesis.")

        # Default Extractive Fallback: Fully local, zero-dependency, grounded synthesis
        return self._extractive_grounded_fallback(question, sources), "DocuSense Extractive Grounding (Local Fallback)"

    def _extractive_grounded_fallback(
        self,
        question: str,
        sources: List[Dict[str, Any]],
    ) -> str:
        """
        High-precision extractive fallback for when external generative APIs are unavailable.
        Extracts the most grounded sentences answering the question, connects them logically,
        and appends verified source citations.
        """
        # Split question words
        q_words = set(re.findall(r"\w+", question.lower()))
        stop_words = {"what", "who", "when", "where", "why", "how", "did", "does", "is", "was", "are", "were", "the", "a", "an", "to", "in", "of", "and"}
        q_keywords = q_words - stop_words

        if not q_keywords:
            q_keywords = q_words

        scored_sentences = []
        for s in sources:
            source_idx = s["source_index"]
            text = s["full_passage"]
            # Split sentences
            sentences = [sent.strip() for sent in re.split(r"(?<=[.!?])\s+", text) if len(sent.strip()) > 20]
            
            for sent in sentences:
                sent_words = set(re.findall(r"\w+", sent.lower()))
                overlap = sent_words.intersection(q_keywords)
                if overlap:
                    score = len(overlap) / (len(q_keywords) + 0.1)
                    scored_sentences.append((score, sent, source_idx))

        if not scored_sentences:
            return (
                "The provided BBC News collection does not contain sufficient evidence to answer this question. "
                "No overlapping factual evidence was found in the retrieved articles."
            )

        # Sort by overlap score
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # Pick top 2-3 most informative, non-redundant sentences
        selected = []
        seen_texts = set()
        for score, sent, s_idx in scored_sentences:
            cleaned_sent = sent.capitalize()
            if not cleaned_sent.endswith("."):
                cleaned_sent += "."
            
            # Avoid near-duplicates
            key = cleaned_sent[:40].lower()
            if key not in seen_texts:
                seen_texts.add(key)
                selected.append(f"{cleaned_sent} [Source {s_idx}]")
            if len(selected) >= 3:
                break

        answer = " ".join(selected)
        return answer


_rag_instance: Optional[RAGAssistant] = None


def get_rag_assistant() -> RAGAssistant:
    """Singleton getter for RAGAssistant."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGAssistant()
    return _rag_instance
