"""
POS Tagger for code-switching between English and Mandarin Chinese.

Uses spaCy models:
- en_core_web_sm for English tokens
- zh_core_web_sm for Chinese tokens

Extracts Universal POS (UPOS) tags via token.pos for language-independent comparison.
"""

import spacy
from typing import List, Dict, Tuple, Optional


class CodeSwitchingPOSTagger:
    """
    POS tagger for code-switched utterances between English and Mandarin Chinese.

    Processes full utterances through both English and Chinese models, then merges
    results based on per-token language labels to get the best POS tags.
    """

    def __init__(
        self,
        en_model_name: str = "en_core_web_sm",
        zh_model_name: str = "zh_core_web_sm",
    ):
        """
        Initialize the code-switching POS tagger.

        Args:
            en_model_name: Name of the English spaCy model
            zh_model_name: Name of the Chinese spaCy model
        """
        try:
            self.en_model = spacy.load(en_model_name)
        except OSError:
            raise OSError(
                f"English model '{en_model_name}' not found. "
                f"Install with: python -m spacy download {en_model_name}"
            )

        try:
            self.zh_model = spacy.load(zh_model_name)
        except OSError:
            raise OSError(
                f"Chinese model '{zh_model_name}' not found. "
                f"Install with: python -m spacy download {zh_model_name}"
            )

    def tag_utterance(
        self,
        text: str,
        language_labels: Optional[List[str]] = None,
        tokens: Optional[List[str]] = None,
    ) -> List[Tuple[str, str]]:
        """
        Tag POS for a full code-switched utterance.

        Args:
            text: Full utterance as a string (e.g., "我刚刚开始record")
            language_labels: List of language labels per token (e.g., ["ZH", "ZH", "EN"]).
                           If provided, uses these to decide which model's output to keep.
                           If None, attempts to auto-detect based on character types.
            tokens: List of tokens from preprocessing. Used for alignment with preprocessed tokens.

        Returns:
            List of (token, upos_tag) tuples
        """
        # If preprocessed tokens are provided, use them directly for alignment
        if tokens is not None:
            return self._tag_with_preprocessed_tokens(text, tokens, language_labels)

        # Otherwise, use spaCy's tokenization
        # Process through both models
        en_doc = self.en_model(text)
        zh_doc = self.zh_model(text)

        # Get language labels if not provided
        if language_labels is None:
            language_labels = self._auto_detect_language(
                [token.text for token in en_doc]
            )

        # For each token, decide which model to use
        result = []
        for idx, (en_token, zh_token) in enumerate(zip(en_doc, zh_doc)):
            if idx < len(language_labels):
                lang = language_labels[idx]
            else:
                # Fallback: auto-detect
                lang = self._detect_token_language(en_token.text)

            # Use the appropriate model's UPOS tag
            if lang.upper() == "EN":
                upos = en_token.pos_
            else:  # ZH
                upos = zh_token.pos_

            result.append((en_token.text, upos))

        return result

    def _tag_with_preprocessed_tokens(
        self, text: str, tokens: List[str], language_labels: Optional[List[str]]
    ) -> List[Tuple[str, str]]:
        """
        Tag POS using preprocessed tokens for alignment.

        This method aligns preprocessed tokens with spaCy's token outputs
        by matching character positions in the full text.

        Args:
            text: Full utterance text
            tokens: Preprocessed tokens
            language_labels: Language label per token

        Returns:
            List of (token, upos_tag) tuples
        """
        # Get language labels if not provided
        if language_labels is None:
            language_labels = self._auto_detect_language(tokens)

        # Process through both models
        en_doc = self.en_model(text)
        zh_doc = self.zh_model(text)

        # Build character position -> token mapping for alignment
        char_pos = 0
        token_positions = {}  # maps token to its character positions

        for token in tokens:
            token_start = text.find(token, char_pos)
            if token_start != -1:
                token_positions[token] = (token_start, token_start + len(token))
                char_pos = token_start + len(token)

        # For each preprocessed token, find which spaCy tokens overlap with it
        result = []
        for token_idx, token in enumerate(tokens):
            if token_idx < len(language_labels):
                lang = language_labels[token_idx]
            else:
                lang = self._detect_token_language(token)

            # Get the character positions of this token
            if token in token_positions:
                token_start, token_end = token_positions[token]
            else:
                # Fallback: try to find it anyway
                token_start = text.find(token)
                token_end = token_start + len(token) if token_start != -1 else -1

            # Find overlapping spaCy token(s) and get their POS tag
            if token_start != -1:
                upos = self._get_upos_for_position(
                    text, token_start, token_end, lang, en_doc, zh_doc
                )
            else:
                upos = "UNKNOWN"

            result.append((token, upos))

        return result

    def _get_upos_for_position(
        self, text: str, start: int, end: int, lang: str, en_doc, zh_doc
    ) -> str:
        """
        Get the UPOS tag for a specific character position range.

        Args:
            text: Full text
            start: Character start position
            end: Character end position
            lang: Language label ("EN" or "ZH")
            en_doc: English spaCy doc
            zh_doc: Chinese spaCy doc

        Returns:
            The UPOS tag
        """
        mid_pos = start + (end - start) // 2

        # Find spaCy token that contains this position
        if lang.upper() == "EN":
            doc = en_doc
        else:
            doc = zh_doc

        for token in doc:
            if token.idx <= mid_pos < token.idx + len(token.text):
                return token.pos_

        # If no exact match, use character detection
        if lang.upper() == "EN":
            return en_doc[0].pos_ if len(en_doc) > 0 else "UNKNOWN"
        else:
            return zh_doc[0].pos_ if len(zh_doc) > 0 else "UNKNOWN"

    def tag_batch(self, data: List[Dict]) -> List[Dict]:
        """
        Tag POS for a batch of utterances.

        Expected data format:
        [
            {
                "id": 0,
                "text": "我刚刚开始record",
                "tokens": ["我", "刚", "刚", "开", "始", "record"],
                "language_labels": ["ZH", "ZH", "ZH", "ZH", "ZH", "EN"]  # optional
            },
            ...
        ]

        Args:
            data: List of dictionaries with utterance information

        Returns:
            List of dictionaries with added "pos_tags" field
        """
        results = []
        for item in data:
            text = item.get("text", "")
            tokens = item.get("tokens", [])
            language_labels = item.get("language_labels")

            pos_results = self.tag_utterance(text, language_labels, tokens)

            # Add results to item
            item_with_pos = item.copy()
            item_with_pos["pos_tags"] = pos_results
            # Also store aligned format for convenience
            item_with_pos["tokens_with_pos"] = [
                {"token": token, "pos": pos} for token, pos in pos_results
            ]
            results.append(item_with_pos)

        return results

    @staticmethod
    def _auto_detect_language(tokens: List[str]) -> List[str]:
        """
        Auto-detect language for each token (simple heuristic).

        Chinese characters (ideograms) are in the CJK Unicode ranges.
        English tokens are mostly ASCII letters and numbers.

        Args:
            tokens: List of token strings

        Returns:
            List of language labels ("EN" or "ZH")
        """
        labels = []
        for token in tokens:
            labels.append(CodeSwitchingPOSTagger._detect_token_language(token))
        return labels

    @staticmethod
    def _detect_token_language(token: str) -> str:
        """
        Detect the language of a single token.

        Args:
            token: A token string

        Returns:
            "EN" or "ZH"
        """
        # Check if token contains Chinese characters
        for char in token:
            # CJK Unicode ranges
            if "\u4e00" <= char <= "\u9fff":
                return "ZH"

        # Default to English if no Chinese characters found
        return "EN"

    def get_model_info(self) -> Dict[str, str]:
        """Get information about loaded models."""
        return {
            "en_model": self.en_model.meta.get("name", "unknown"),
            "zh_model": self.zh_model.meta.get("name", "unknown"),
            "en_version": self.en_model.meta.get("version", "unknown"),
            "zh_version": self.zh_model.meta.get("version", "unknown"),
        }


class NaiveBaselinePOSTagger:
    """
    Naive baseline POS tagger: processes all tokens through English spaCy model only.

    This baseline ignores language detection and processes everything as English.
    Used to demonstrate degradation on Chinese and code-switch tokens.
    """

    def __init__(self, en_model_name: str = "en_core_web_sm"):
        """
        Initialize the naive baseline tagger with English model only.

        Args:
            en_model_name: Name of the English spaCy model
        """
        try:
            self.en_model = spacy.load(en_model_name)
        except OSError:
            raise OSError(
                f"English model '{en_model_name}' not found. "
                f"Install with: python -m spacy download {en_model_name}"
            )

    def tag_utterance(
        self,
        text: str,
        tokens: Optional[List[str]] = None,
        language_labels: Optional[List[str]] = None,
    ) -> List[Tuple[str, str]]:
        """
        Tag POS using only the English model (ignoring language labels).

        Args:
            text: Full utterance as a string
            tokens: List of tokens from preprocessing (used for alignment)
            language_labels: Ignored; kept for interface compatibility

        Returns:
            List of (token, upos_tag) tuples
        """
        # If preprocessed tokens are provided, use them directly for alignment
        if tokens is not None:
            return self._tag_with_preprocessed_tokens(text, tokens)

        # Otherwise, use spaCy's tokenization
        en_doc = self.en_model(text)
        result = [(token.text, token.pos_) for token in en_doc]
        return result

    def _tag_with_preprocessed_tokens(
        self, text: str, tokens: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Tag POS using preprocessed tokens with English model only.

        Args:
            text: Full utterance text
            tokens: Preprocessed tokens

        Returns:
            List of (token, upos_tag) tuples
        """
        # Process text through English model
        en_doc = self.en_model(text)

        # Build character position -> token mapping for alignment
        char_pos = 0
        token_positions = {}

        for token in tokens:
            token_start = text.find(token, char_pos)
            if token_start != -1:
                token_positions[token] = (token_start, token_start + len(token))
                char_pos = token_start + len(token)

        # For each preprocessed token, find overlapping spaCy token
        result = []
        for token in tokens:
            if token in token_positions:
                token_start, token_end = token_positions[token]
            else:
                token_start = text.find(token)
                token_end = token_start + len(token) if token_start != -1 else -1

            # Find overlapping spaCy token and get its POS tag
            if token_start != -1:
                upos = self._get_upos_for_position(text, token_start, token_end, en_doc)
            else:
                upos = "UNKNOWN"

            result.append((token, upos))

        return result

    def _get_upos_for_position(self, text: str, start: int, end: int, en_doc) -> str:
        """
        Get the UPOS tag for a specific character position range.

        Args:
            text: Full text
            start: Character start position
            end: Character end position
            en_doc: English spaCy doc

        Returns:
            The UPOS tag
        """
        mid_pos = start + (end - start) // 2

        for token in en_doc:
            if token.idx <= mid_pos < token.idx + len(token.text):
                return token.pos_

        # Fallback to first token's POS if no match
        return en_doc[0].pos_ if len(en_doc) > 0 else "UNKNOWN"

    def tag_batch(self, data: List[Dict]) -> List[Dict]:
        """
        Tag POS for a batch of utterances using English model only.

        Args:
            data: List of dictionaries with utterance information

        Returns:
            List of dictionaries with added "pos_tags" field
        """
        results = []
        for item in data:
            text = item.get("text", "")
            tokens = item.get("tokens", [])

            pos_results = self.tag_utterance(text, tokens)

            # Add results to item
            item_with_pos = item.copy()
            item_with_pos["pos_tags"] = pos_results
            item_with_pos["tokens_with_pos"] = [
                {"token": token, "pos": pos} for token, pos in pos_results
            ]
            results.append(item_with_pos)

        return results

    def get_model_info(self) -> Dict[str, str]:
        """Get information about loaded model."""
        return {
            "en_model": self.en_model.meta.get("name", "unknown"),
            "en_version": self.en_model.meta.get("version", "unknown"),
            "tagger_type": "naive_baseline",
        }


def print_results(results: List[Dict], num_examples: int = 5):
    """
    Pretty-print POS tagging results.

    Args:
        results: List of results from tag_batch
        num_examples: Number of examples to print
    """
    for item in results[:num_examples]:
        print(f"\n{'=' * 60}")
        print(f"ID: {item.get('id')}")
        print(f"Text: {item.get('text')}")
        print(f"Tokens & POS Tags:")
        for token_pos in item.get("tokens_with_pos", []):
            token = token_pos["token"]
            pos = token_pos["pos"]
            print(f"  {token:15} -> {pos}")
        if "language_labels" in item:
            print(f"Language Labels: {item['language_labels']}")
