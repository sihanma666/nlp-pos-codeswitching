import json
from preprocessing.pos_tagger import CodeSwitchingPOSTagger
from evaluation.baseline_metrics import evaluate_batch, print_evaluation
from evaluation.cross_validation import run_kfold_evaluation, print_cv_summary
#########gold data !!!
with open("your_gold_data.json", "r", encoding="utf-8") as f:
    gold_data = json.load(f)

tagger = CodeSwitchingPOSTagger()

predictions = tagger.tag_batch(gold_data)
metrics = evaluate_batch(predictions, gold_data)
print_evaluation(metrics)

fold_results, summary = run_kfold_evaluation(gold_data, tagger)


print("\n" + "=" * 60)
print(f"{summary['k']}-fold cross validation results")
print("=" * 60)

for result in fold_results:
    print(f"\nFold {result['fold']}:")

    print(f"  overall accuracy: {result['overall_accuracy']}")
    print(f"  overall macro f1: {result['overall_macro_f1']}")

    if result["switch_accuracy"] != None:
        print(f"  switch accuracy: {result['switch_accuracy']}")
        print(f"  switch macro f1: {result['switch_macro_f1']}")
    else:
        print("  switch metrics: N/A")

print("\nAverage across folds:")

print(f"  overall accuracy: {summary['cv_overall_accuracy']}")
print(f"  overall macro f1: {summary['cv_overall_macro_f1']}")

if summary["cv_switch_accuracy"] != None:
    print(f"  switch accuracy: {summary['cv_switch_accuracy']}")
    print(f"  switch macro f1: {summary['cv_switch_macro_f1']}")
else:
    print("  switch metrics: N/A")
