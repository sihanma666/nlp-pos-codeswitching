import json
from preprocessing.pos_tagger import CodeSwitchingPOSTagger
from evaluation.cross_validation import evaluate_batch
from evaluation.baseline_metrics import print_evaluation
from evaluation.cross_validation import run_kfold_evaluation


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


with open("./data/data_stanza_gold.json", "r", encoding="utf-8") as f:
    gold_data = json.load(f)

input_data = remove_gold_labels(gold_data)

tagger = CodeSwitchingPOSTagger()

predictions = tagger.tag_batch(input_data)
#metrics = evaluate_batch(predictions, gold_data)
#print_evaluation(metrics)

fold_results, summary = run_kfold_evaluation(gold_data, tagger, method ="our")


print("\n" + "=" * 60)
print(f"{summary['k']}-fold cross validation results")
print("=" * 60)

for result in fold_results:
    print(f"\nFold {result['fold']}:")
    print(f"  overall accuracy: {result['overall_accuracy']:.4f}")
    print(f"  overall macro f1: {result['overall_macro_f1']:.4f}")
    print(f"  total tokens: {result['total_tokens']}")

print("\nAverage across folds:")
print(f"  overall accuracy: {summary['cv_overall_accuracy']:.4f}")
print(f"  overall macro f1: {summary['cv_overall_macro_f1']:.4f}")
