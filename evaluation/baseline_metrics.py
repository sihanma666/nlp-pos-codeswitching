"""
Evaluation metrics for POS tagging.

Provides accuracy metrics for the code-switching POS tagger.
"""

from typing import List, Dict, Tuple
from collections import defaultdict


def token_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Calculate token-level accuracy.

    Args:
        predictions: List of predicted POS tags
        references: List of reference/gold POS tags

    Returns:
        Accuracy as a float (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: predictions {len(predictions)}, references {len(references)}"
        )

    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(predictions)


def language_specific_accuracy(
    predictions: List[str], references: List[str], language_labels: List[str]
) -> Dict[str, float]:
    """
    Calculate accuracy broken down by language.

    Args:
        predictions: List of predicted POS tags
        references: List of reference/gold POS tags
        language_labels: List of language labels ("EN" or "ZH")

    Returns:
        Dict with keys "overall", "EN", "ZH" containing accuracies
    """
    if len(predictions) != len(references) or len(predictions) != len(language_labels):
        raise ValueError("Input lists must have the same length")

    # Overall accuracy
    overall = token_accuracy(predictions, references)

    # Per-language accuracy
    en_preds = [p for p, lang in zip(predictions, language_labels) if lang == "EN"]
    en_refs = [r for r, lang in zip(references, language_labels) if lang == "EN"]
    en_acc = token_accuracy(en_preds, en_refs) if en_preds else 0.0

    zh_preds = [p for p, lang in zip(predictions, language_labels) if lang == "ZH"]
    zh_refs = [r for r, lang in zip(references, language_labels) if lang == "ZH"]
    zh_acc = token_accuracy(zh_preds, zh_refs) if zh_preds else 0.0

    return {
        "overall": overall,
        "EN": en_acc,
        "ZH": zh_acc,
        "en_count": len(en_preds),
        "zh_count": len(zh_preds),
    }


def confusion_matrix(
    predictions: List[str], references: List[str]
) -> Dict[Tuple[str, str], int]:
    """
    Build a confusion matrix for POS tags.

    Args:
        predictions: List of predicted POS tags
        references: List of reference/gold POS tags

    Returns:
        Dict mapping (reference, prediction) to count
    """
    matrix = defaultdict(int)
    for ref, pred in zip(references, predictions):
        matrix[(ref, pred)] += 1
    return dict(matrix)


def evaluate_batch(results: List[Dict], gold_standard: List[Dict]) -> Dict:
    """
    Evaluate a batch of POS tagging results against gold standard.

    Args:
        results: List of tagged results (from CodeSwitchingPOSTagger.tag_batch)
        gold_standard: List of gold standard results with "gold_pos_tags" field

    Returns:
        Dict with evaluation metrics
    """
    all_predictions = []
    all_references = []
    all_languages = []

    for result, gold in zip(results, gold_standard):
        result_pos = [pos for _, pos in result.get("pos_tags", [])]
        gold_pos = gold.get("gold_pos_tags", [])
        languages = result.get("language_labels", [])

        if len(result_pos) == len(gold_pos):
            all_predictions.extend(result_pos)
            all_references.extend(gold_pos)
            all_languages.extend(languages)

    # Calculate metrics
    overall_acc = token_accuracy(all_predictions, all_references)
    lang_specific = language_specific_accuracy(
        all_predictions, all_references, all_languages
    )

    return {
        "overall_accuracy": overall_acc,
        "accuracy_by_language": lang_specific,
        "total_tokens": len(all_predictions),
        "confusion_matrix": confusion_matrix(all_predictions, all_references),
    }


def print_evaluation(metrics: Dict):
    """
    Pretty-print evaluation metrics.

    Args:
        metrics: Dict from evaluate_batch()
    """
    print("\n" + "=" * 60)
    print("POS TAGGING EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nTotal tokens evaluated: {metrics['total_tokens']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")

    lang_acc = metrics["accuracy_by_language"]
    print(f"\nAccuracy by language:")
    print(f"  English: {lang_acc['EN']:.4f} ({lang_acc['en_count']} tokens)")
    print(f"  Chinese: {lang_acc['ZH']:.4f} ({lang_acc['zh_count']} tokens)")
