import random
from typing import Dict, List, Sequence, Tuple


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

def run_kfold_evaluation(gold_data, tagger, k = 5,seed = 42):
    
    splits = k_fold_split(gold_data, k=k, seed=seed)
    fold_results = []

    for fold_idx, (train_fold, val_fold) in enumerate(splits, start=1):
        predictions = tagger.tag_batch(val_fold)
        metrics = evaluate_batch(predictions, list(val_fold))
        metrics["fold"] = fold_idx
        metrics["val_size"] = len(val_fold)
        fold_results.append(metrics)

    summary = {
        "k": k,
        "cv_overall_accuracy": average(fold_results, "overall_accuracy"),
        "cv_overall_macro_f1": average(fold_results, "overall_macro_f1"),
        "cv_switch_accuracy": average(fold_results, "switch_accuracy"),
        "cv_switch_macro_f1": average(fold_results, "switch_macro_f1"),
    }
    return fold_results, summary

