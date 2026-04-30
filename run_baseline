import json
from preprocessing.pos_tagger import MonolingualPOSTagger
from evaluation.baseline_metrics import print_evaluation
from evaluation.cross_validation import evaluate_batch, run_kfold_evaluation


def remove_gold_labels_baseline(data):
    return [
        {
            "id": item["id"],
            "text": item["text"],
            "tokens": item["tokens"]
        }
        for item in data
    ]


with open("./data/data_stanza_gold.json", "r", encoding="utf-8") as f:
    gold_data = json.load(f)

# hold out ASCEND test split (IDs 10999-12313)
trainval_data = [item for item in gold_data if item["id"] < "10999"]
test_data = [item for item in gold_data if item["id"] >= "10999"]

print(f"Train+val: {len(trainval_data)} utterances | Test (held out): {len(test_data)} utterances")

tagger = MonolingualPOSTagger()

# cross-validation on train+val
fold_results, summary = run_kfold_evaluation(trainval_data, tagger, method="base")

print("\n" + "=" * 60)
print(f"{summary['k']}-fold cross validation results (train+val)")
print("=" * 60)

for result in fold_results:
    print(f"\nFold {result['fold']}:")
    print(f"  overall accuracy:       {result['overall_accuracy']:.4f}")
    print(f"  overall macro f1:       {result['overall_macro_f1']:.4f}")
    print(f"  switch point accuracy:  {result['switch_point_accuracy']:.4f}")
    print(f"  switch point macro f1:  {result['switch_point_f1']:.4f}")
    print(f"  total tokens: {result['total_tokens']}")

print("\nAverage across folds:")
print(f"  overall accuracy:       {summary['cv_overall_accuracy']:.4f}")
print(f"  overall macro f1:       {summary['cv_overall_macro_f1']:.4f}")
print(f"  switch point accuracy:  {summary['cv_switch_point_accuracy']:.4f}")
print(f"  switch point macro f1:  {summary['cv_switch_point_f1']:.4f}")

# final evaluation on held-out test set
print("\n" + "=" * 60)
print("Final evaluation on held-out test set")
print("=" * 60)

test_input = remove_gold_labels_baseline(test_data)
test_predictions = tagger.tag_batch(test_input)
test_metrics = evaluate_batch(test_predictions, test_data)

print(f"  overall accuracy:       {test_metrics['overall_accuracy']:.4f}")
print(f"  overall macro f1:       {test_metrics['overall_macro_f1']:.4f}")
print(f"  switch point accuracy:  {test_metrics['switch_point_accuracy']:.4f}")
print(f"  switch point macro f1:  {test_metrics['switch_point_f1']:.4f}")
print(f"  total tokens: {test_metrics['total_tokens']}")
