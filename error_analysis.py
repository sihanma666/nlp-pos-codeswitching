import json
import sys
from collections import Counter
from preprocessing.pos_tagger import CodeSwitchingPOSTagger

sys.stdout.reconfigure(encoding='utf-8')

with open("./data/data_stanza_gold.json", "r", encoding="utf-8") as f:
    gold_data = json.load(f)

test_data = [item for item in gold_data if item["id"] >= "10999"]

tagger = CodeSwitchingPOSTagger()

test_input = [
    {"id": item["id"], "text": item["text"], "tokens": item["tokens"],
     "language_labels": item["labels"]}
    for item in test_data
]
predictions = tagger.tag_batch(test_input)

# --- collect switch-point errors ---
sp_errors = []        # (gold, pred, token, lang, tokens, sp_idx, position_type)
at_sp_correct = 0
at_sp_total = 0
after_sp_correct = 0
after_sp_total = 0
en_sp_correct = 0; en_sp_total = 0
zh_sp_correct = 0; zh_sp_total = 0

for pred_item, gold_item in zip(predictions, test_data):
    tokens = gold_item["tokens"]
    labels = gold_item["labels"]
    switch_points = gold_item.get("switch_points", [])
    pred_pos = [pos for _, pos in pred_item.get("pos_tags", [])]
    gold_pos = [p for _, p in gold_item.get("gold_pos_tags", [])]

    if len(pred_pos) != len(gold_pos):
        continue

    sp_set = set(switch_points)
    after_sp_set = set(idx + 1 for idx in switch_points if idx + 1 < len(tokens))

    for idx in switch_points:
        if idx >= len(tokens):
            continue
        g, p, tok, lang = gold_pos[idx], pred_pos[idx], tokens[idx], labels[idx]
        sp_errors.append((g, p, tok, lang, tokens, idx, "at"))
        at_sp_total += 1
        if g == p:
            at_sp_correct += 1
        if lang == "EN":
            en_sp_total += 1
            if g == p: en_sp_correct += 1
        else:
            zh_sp_total += 1
            if g == p: zh_sp_correct += 1

    for idx in after_sp_set:
        if idx in sp_set:
            continue
        if idx >= len(tokens):
            continue
        g, p = gold_pos[idx], pred_pos[idx]
        sp_errors.append((g, p, tokens[idx], labels[idx], tokens, idx, "after"))
        after_sp_total += 1
        if g == p:
            after_sp_correct += 1

# --- confusion matrix ---
error_pairs = Counter(
    (g, p) for g, p, tok, lang, tokens, idx, pos_type in sp_errors
    if g != p and pos_type == "at"
)

# --- collect examples for top 5 error pairs ---
top5 = [pair for pair, _ in error_pairs.most_common(5)]
examples = {pair: [] for pair in top5}

for pred_item, gold_item in zip(predictions, test_data):
    tokens = gold_item["tokens"]
    labels = gold_item["labels"]
    switch_points = gold_item.get("switch_points", [])
    pred_pos = [pos for _, pos in pred_item.get("pos_tags", [])]
    gold_pos = [p for _, p in gold_item.get("gold_pos_tags", [])]
    if len(pred_pos) != len(gold_pos):
        continue
    for idx in switch_points:
        if idx >= len(tokens):
            continue
        pair = (gold_pos[idx], pred_pos[idx])
        if pair in examples and len(examples[pair]) < 3:
            window_start = max(0, idx - 2)
            window_end = min(len(tokens), idx + 3)
            examples[pair].append({
                "window": tokens[window_start:window_end],
                "switch_token": tokens[idx],
                "switch_idx_in_window": idx - window_start,
                "lang": labels[idx],
                "pred": pred_pos[idx],
                "gold": gold_pos[idx],
            })

# ============================================================
print("=" * 60)
print("ERROR ANALYSIS: SWITCH-POINT TOKENS (HELD-OUT TEST SET)")
print("=" * 60)

print(f"\nTotal switch-point token positions evaluated: {at_sp_total}")
print(f"Accuracy at switch points: {at_sp_correct/at_sp_total:.4f} ({at_sp_correct}/{at_sp_total})")

print("\n" + "-" * 60)
print("TOP 15 MISMATCH PAIRS AT SWITCH POINTS (gold → predicted)")
print("-" * 60)
print(f"{'Gold':12} {'Predicted':12} {'Count':>6}  {'% of SP errors':>14}")
total_sp_errors = sum(c for _, c in error_pairs.most_common())
for (g, p), count in error_pairs.most_common(15):
    print(f"{g:12} {p:12} {count:6}   {count/total_sp_errors*100:>12.1f}%")

print("\n" + "-" * 60)
print("TOP 5 ERROR PAIRS — EXAMPLE SENTENCES")
print("-" * 60)
for rank, pair in enumerate(top5, 1):
    g, p = pair
    count = error_pairs[pair]
    print(f"\n#{rank}. Gold={g} → Predicted={p}  (n={count})")
    for i, ex in enumerate(examples[pair], 1):
        window_str = " | ".join(
            f"[{tok}]" if j == ex["switch_idx_in_window"] else tok
            for j, tok in enumerate(ex["window"])
        )
        print(f"  Example {i}:")
        print(f"    Context:    {window_str}")
        print(f"    Token:      {ex['switch_token']}  (language: {ex['lang']})")
        print(f"    Predicted:  {ex['pred']}")
        print(f"    Gold:       {ex['gold']}")

print("\n" + "-" * 60)
print("AT SWITCH POINT vs. ONE POSITION AFTER")
print("-" * 60)
at_acc = at_sp_correct / at_sp_total if at_sp_total else 0
after_acc = after_sp_correct / after_sp_total if after_sp_total else 0
print(f"  Tokens AT switch point:         accuracy = {at_acc:.4f}  ({at_sp_correct}/{at_sp_total})")
print(f"  Tokens ONE AFTER switch point:  accuracy = {after_acc:.4f}  ({after_sp_correct}/{after_sp_total})")
print(f"  Difference: {abs(at_acc - after_acc):.4f} ({'AT harder' if at_acc < after_acc else 'AFTER harder'})")

print("\n" + "-" * 60)
print("ERROR BREAKDOWN BY LANGUAGE LABEL AT SWITCH POINTS")
print("-" * 60)
en_acc = en_sp_correct / en_sp_total if en_sp_total else 0
zh_acc = zh_sp_correct / zh_sp_total if zh_sp_total else 0
print(f"  EN-labeled switch-point tokens: accuracy = {en_acc:.4f}  ({en_sp_correct}/{en_sp_total})")
print(f"  ZH-labeled switch-point tokens: accuracy = {zh_acc:.4f}  ({zh_sp_correct}/{zh_sp_total})")
