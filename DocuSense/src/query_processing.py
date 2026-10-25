"""
DocuSense Query Processing & Understanding Module
Implements differentiated query processing, validation, entity-aware normalization,
and transparent query explanation for lexical vs. semantic retrieval.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional,Tuple

try:
    import config
    from src.preprocessing import clean_whitespace, lexical_preprocess, semantic_preprocess, get_lemmatizer
except ImportError:
    from .. import config
    from .preprocessing import clean_whitespace, lexical_preprocess, semantic_preprocess, get_lemmatizer

logger = logging.getLogger("DocuSense.QueryProcessing")

# Known BBC News named entities to protect against spellchecker corruption
PROTECTED_ENTITIES = {
    "wenger": "Wenger",
    "arsene": "Arsene",
    "arsenal": "Arsenal",
    "ebbers": "Ebbers",
    "worldcom": "WorldCom",
    "cynthia": "Cynthia",
    "cooper": "Cooper",
    "ronaldo": "Ronaldo",
    "gerrard": "Gerrard",
    "ferguson": "Ferguson",
    "blair": "Blair",
    "howard": "Howard",
    "goldsmith": "Goldsmith",
    "chirac": "Chirac",
    "snoop": "Snoop",
    "dogg": "Dogg",
    "playstation": "PlayStation",
    "microsoft": "Microsoft",
    "google": "Google",
    "sony": "Sony",
    "chelsea": "Chelsea",
    "liverpool": "Liverpool",
}

# Domain acronyms & abbreviations
DOMAIN_ACRONYMS = {
    "uk": "UK",
    "eu": "EU",
    "us": "US",
    "usa": "USA",
    "mpc": "MPC",
    "boe": "BOE",
    "ecb": "ECB",
    "un": "UN",
    "pc": "PC",
    "pcs": "PCs",
}


@dataclass
class QueryResult:
    """Holds processed query details and audit information."""
    raw_query: str
    processed_query: str
    retrieval_method: str
    tokens: List[str] = field(default_factory=list)
    suggested_correction: Optional[str] = None
    is_valid: bool = True
    status_message: str = "Query is valid."


class QueryProcessor:
    """
    Intelligent query processing pipeline.
    Applies differentiated normalization tailored to the retrieval method:
      - Lexical (TF-IDF): tokenization, case-folding, optional lemmatization, stop-word alignment.
      - Semantic (Transformers): preserves casing, numbers, negations, punctuation for rich contextual encoding.
    """

    def __init__(self, enable_spellcheck: bool = True):
        self.enable_spellcheck = enable_spellcheck
        self.spell = None
        self._init_spellchecker()

    def _init_spellchecker(self):
        """Optionally load pyspellchecker with domain-specific terms."""
        if not self.enable_spellcheck:
            return
        try:
            from spellchecker import SpellChecker
            self.spell = SpellChecker()
            # Register protected entities in dictionary
            self.spell.word_frequency.load_words(list(PROTECTED_ENTITIES.keys()))
        except ImportError:
            logger.info("pyspellchecker not installed. Spelling correction disabled.")
            self.spell = None

    def validate_query(self, raw_query: str) -> Tuple[bool, str]:
        """
        Check for empty, whitespace-only, or pure punctuation queries.
        """
        if not raw_query or not isinstance(raw_query, str):
            return False, "Query cannot be empty."

        clean = clean_whitespace(raw_query)
        if len(clean) == 0:
            return False, "Query contains only whitespace."

        # Check if query contains any alphanumeric character
        if not re.search(r"\w", clean):
            return False, "Query contains only punctuation or special symbols."

        if len(clean) < 2:
            return False, "Query is too short (minimum 2 characters required)."

        return True, "Query is valid."

    def process(
        self,
        raw_query: str,
        method: str = "embeddings",
        apply_lemmatization: bool = False,
    ) -> QueryResult:
        """
        Process user query according to the targeted retrieval method.
        method: 'tfidf' or 'embeddings' (or 'comparison')
        """
        is_valid, msg = self.validate_query(raw_query)
        if not is_valid:
            return QueryResult(
                raw_query=raw_query,
                processed_query="",
                retrieval_method=method,
                is_valid=False,
                status_message=msg,
            )

        clean = clean_whitespace(raw_query)

        # Entity-protected spelling correction
        corrected_query, had_corrections = self.correct_spelling_safe(clean)

        if method == "tfidf":
            # Lexical pipeline
            query_to_use = corrected_query if had_corrections else clean
            processed = lexical_preprocess(
                query_to_use,
                lemmatize=apply_lemmatization,
                remove_punctuation=True,
                lower=True,
            )
            tokens = processed.split()
            explanation = (
                f"Lexical normalisation: lowercased, punctuation removed"
                f"{', lemmatized' if apply_lemmatization else ''}."
            )
        else:
            # Semantic transformer pipeline (DO NOT over-clean or aggressively strip!)
            processed = semantic_preprocess(clean)
            tokens = processed.split()
            explanation = "Semantic normalisation: preserved sentence structure, punctuation, and casing for transformer attention."

        return QueryResult(
            raw_query=raw_query,
            processed_query=processed,
            retrieval_method=method,
            tokens=tokens,
            suggested_correction=corrected_query if had_corrections else None,
            is_valid=True,
            status_message=explanation,
        )

    def correct_spelling_safe(self, text: str) -> Tuple[str, bool]:
        """
        Spelling correction with named-entity guard to prevent corrupting
        proper nouns into common words.
        """
        if self.spell is None:
            return text, False

        words = text.split()
        corrected_words = []
        changed = False

        for word in words:
            word_lower = re.sub(r"[^\w]", "", word).lower()
            
            # Check protected entities
            if word_lower in PROTECTED_ENTITIES:
                corrected_words.append(word)
                continue

            # Check domain acronyms
            if word_lower in DOMAIN_ACRONYMS:
                corrected_words.append(DOMAIN_ACRONYMS[word_lower])
                continue

            # Skip numbers or single characters
            if word.isdigit() or len(word) <= 2:
                corrected_words.append(word)
                continue

            # Check if word is already recognized
            if word_lower in self.spell:
                corrected_words.append(word)
                continue

            # Attempt safe correction
            candidate = self.spell.correction(word_lower)
            if candidate and candidate != word_lower:
                # Retain capitalization if original was capitalized
                if word[0].isupper():
                    candidate = candidate.capitalize()
                corrected_words.append(candidate)
                changed = True
            else:
                corrected_words.append(word)

        return " ".join(corrected_words), changed


_processor_instance: Optional[QueryProcessor] = None


def get_query_processor() -> QueryProcessor:
    """Singleton getter for QueryProcessor."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = QueryProcessor()
    return _processor_instance
