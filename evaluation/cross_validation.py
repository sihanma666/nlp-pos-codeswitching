import random
from statistics import mean

from preprocessing.pos_tagger import CodeSwitchingPOSTagger


from evaluation.baseline_metrics import macro_f1, token_accuracy, confusion_matrix, language_specific_accuracy


def remove_gold_labels(data):
    return [
        {
            "id": item["id"],
            "text": item["text"],
            "tokens": item["tokens"],
            "language_labels": item["labels"]
        }
        for item in data
    ]
    
def remove_gold_labels_baseline(data):
    return [
        {
            "id": item["id"],
            "text": item["text"],
            "tokens": item["tokens"]
        }
        for item in data
    ]
    
def average(results, key):
    return mean([r[key] for r in results if r.get(key) is not None])
  
def k_fold_split(data, k = 5, seed = 42):
    # Split data into k folds and return [(train_fold, val_fold), ...].

    items = list(data)
    random.Random(seed).shuffle(items)

    fold_sizes = [len(items) // k] * k
    for i in range(len(items) % k):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for size in fold_sizes:
        end = start + size
        folds.append(items[start:end])
        start = end

    split_pairs = []
    for i in range(k):
        val_fold = folds[i]
        train_fold = []
        for j, fold in enumerate(folds):
            if j != i:
                train_fold.extend(fold)
        split_pairs.append((train_fold, val_fold))
    return split_pairs

def run_kfold_evaluation(gold_data, tagger, method, k=5, seed=42):
    
    splits = k_fold_split(gold_data, k=k, seed=seed)
    fold_results = []

    for fold_idx, (_, val_fold) in enumerate(splits, start=1):
        if method == "our":
            val_input = remove_gold_labels(val_fold)
        if method == "base":
            val_input = remove_gold_labels(val_fold)
            

        predictions = tagger.tag_batch(val_input)
        metrics = evaluate_batch(predictions, val_fold)

        metrics["fold"] = fold_idx
        metrics["val_size"] = len(val_fold)

        fold_results.append(metrics)

    summary = {
        "k": k,
        "cv_overall_accuracy": average(fold_results, "overall_accuracy"),
        "cv_overall_macro_f1": average(fold_results, "overall_macro_f1"),
    }

    return fold_results, summary

def evaluate_batch(results, gold_standard):
    all_predictions = []
    all_references = []
    all_languages = []

    for result, gold in zip(results, gold_standard):
        result_pos = [pos for _, pos in result.get("pos_tags", [])]
        gold_pos = [pos for _, pos in gold.get("gold_pos_tags", [])]
        languages = result.get("language_labels", [])

        if len(result_pos) == len(gold_pos):
            all_predictions.extend(result_pos)
            all_references.extend(gold_pos)
            all_languages.extend(languages)

    # overall metrics
    overall_acc = token_accuracy(all_predictions, all_references)
    overall_macro_f1 = macro_f1(all_predictions, all_references)

    # per-language accuracy
    lang_acc = language_specific_accuracy(
        all_predictions, all_references, all_languages
    )

    # per-language F1
    en_pred, en_ref = [], []
    zh_pred, zh_ref = [], []

    for p, r, l in zip(all_predictions, all_references, all_languages):
        if l == "EN":
            en_pred.append(p)
            en_ref.append(r)
        elif l == "ZH":
            zh_pred.append(p)
            zh_ref.append(r)

    lang_f1 = {
        "EN": macro_f1(en_pred, en_ref) if en_ref else None,
        "ZH": macro_f1(zh_pred, zh_ref) if zh_ref else None,
    }

    return {
        "overall_accuracy": overall_acc,
        "overall_macro_f1": overall_macro_f1,
        "accuracy_by_language": lang_acc,
        "f1_by_language": lang_f1,
        "total_tokens": len(all_predictions),
        "confusion_matrix": confusion_matrix(all_predictions, all_references),
    }
