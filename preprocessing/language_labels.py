"""
Language label generator for code-switched utterances.

Creates EN/ZH language labels for preprocessed tokens.
"""

from typing import List, Tuple


def detect_token_language(token: str) -> str:
    """
    Detect the language of a single token.

    Args:
        token: A token string

    Returns:
        "EN" for English, "ZH" for Chinese
    """
    # Check if token contains Chinese characters
    for char in token:
        # CJK Unicode ranges
        if "\u4e00" <= char <= "\u9fff":
            return "ZH"

    # Default to English if no Chinese characters found
    return "EN"


def label_tokens(tokens: List[str]) -> List[str]:
    """
    Label a list of tokens with their language.

    Args:
        tokens: List of tokens

    Returns:
        List of language labels ("EN" or "ZH")
    """
    return [detect_token_language(token) for token in tokens]


def find_switch_points(language_labels: List[str]) -> List[int]:
    """
    Find switch point indices where language changes.

    A switch point is where the language label changes from one token to the next.

    Args:
        language_labels: List of language labels

    Returns:
        List of indices where switches occur
    """
    switch_points = []
    for i in range(len(language_labels) - 1):
        if language_labels[i] != language_labels[i + 1]:
            switch_points.append(i + 1)
    return switch_points


def add_language_labels_to_data(data: List[dict]) -> List[dict]:
    """
    Add language labels and switch points to preprocessed data.

    Args:
        data: List of dicts with "tokens" field

    Returns:
        List of dicts with added "language_labels" and "switch_points" fields
    """
    enhanced_data = []
    for item in data:
        item_copy = item.copy()
        tokens = item.get("tokens", [])

        # Generate labels
        language_labels = label_tokens(tokens)
        item_copy["language_labels"] = language_labels

        # Find switch points
        switch_points = find_switch_points(language_labels)
        item_copy["switch_points"] = switch_points

        enhanced_data.append(item_copy)

    return enhanced_data
