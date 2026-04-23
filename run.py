import json
from preprocessing.pos_tagger import CodeSwitchingPOSTagger
from evaluation.baseline_metrics import evaluate_batch, print_evaluation
from evaluation.cross_validation import run_kfold_evaluation, print_cv_summary

with open("your_gold_data.json", "r", encoding="utf-8") as f:
    gold_data = json.load(f)

tagger = CodeSwitchingPOSTagger()

predictions = tagger.tag_batch(gold_data)
metrics = evaluate_batch(predictions, gold_data)
print_evaluation(metrics)

fold_results, summary = run_kfold_evaluation(gold_data, tagger)
print_cv_summary(fold_results, summary)
