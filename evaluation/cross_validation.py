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

    folds: List[List[Dict]] = []
    start = 0
    for size in fold_sizes:
        end = start + size
        folds.append(items[start:end])
        start = end

    split_pairs: List[Tuple[List[Dict], List[Dict]]] = []
    for i in range(k):
        val_fold = folds[i]
        train_fold: List[Dict] = []
        for j, fold in enumerate(folds):
            if j != i:
                train_fold.extend(fold)
        split_pairs.append((train_fold, val_fold))
    return split_pairs


