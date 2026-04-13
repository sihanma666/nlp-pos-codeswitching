# Baseline POS Tagger - Quick Reference

## What Was Created

### Core Modules

1. **`preprocessing/pos_tagger.py`** - Main POS tagger
   - `CodeSwitchingPOSTagger` class
   - Load English & Chinese spaCy models
   - Tag utterances with UPOS tags

2. **`preprocessing/language_labels.py`** - Language detection  
   - Auto-detect EN/ZH per token
   - Find code-switching points
   - Label token batches

3. **`preprocessing/end_to_end_tagger.py`** - Complete pipeline
   - Takes raw preprocessed tokens
   - Automatically adds language labels
   - Outputs POS-tagged results
   - **Use this for new data!**

4. **`evaluation/baseline_metrics.py`** - Evaluation tools
   - Token accuracy
   - Per-language breakdown
   - Confusion matrices

### Demo & Documentation

5. **`preprocessing/pos_tagger_demo.py`** - Demo script
   - Shows full pipeline on sample data
   - Run with: `python -m preprocessing.pos_tagger_demo`

6. **`POS_TAGGER_README.md`** - Complete documentation
   - Setup instructions
   - API reference
   - POS tag explanations
   - Troubleshooting

7. **`data/sample_preprocessed_with_labels.json`** - Sample with labels
   - 5 example utterances
   - Pre-labeled with language tags
   - Ready for tagging demo

### Generated Outputs

- `data/final_tagged_output.json` - Fully tagged output
- `data/sample_preprocessed_with_pos_tags.json` - Demo output

## Quick Start Commands

```bash
# 1. Install (already done)
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm

# 2. Tag existing preprocessed data (RECOMMENDED)
python preprocessing/end_to_end_tagger.py \
  --input data/sample_preprocessed.json \
  --output data/my_tagged_output.json

# 3. Or run the demo
python -m preprocessing.pos_tagger_demo
```

## API Cheat Sheet

### Simple Usage

```python
from preprocessing.pos_tagger import CodeSwitchingPOSTagger

tagger = CodeSwitchingPOSTagger()
results = tagger.tag_batch(data)
```

### With Language Labels

```python
from preprocessing.language_labels import add_language_labels_to_data

# Auto-generate labels
data_with_labels = add_language_labels_to_data(data)
results = tagger.tag_batch(data_with_labels)
```

### Evaluate Results

```python
from evaluation.baseline_metrics import evaluate_batch

metrics = evaluate_batch(results, gold_standard)
# Returns: {"overall_accuracy": 0.85, "accuracy_by_language": {...}, ...}
```

## Output Format

Each item has:
- `text`: Original utterance
- `tokens`: Preprocessed tokens
- `language_labels`: EN/ZH per token (auto-generated if missing)
- `switch_points`: Indices where code-switching occurs
- `pos_tags`: List of [token, UPOS] pairs
- `tokens_with_pos`: List of {token, pos} dicts

Example:
```json
{
  "text": "我刚刚开始record",
  "tokens": ["我", "刚", "刚", "开", "始", "record"],
  "language_labels": ["ZH", "ZH", "ZH", "ZH", "ZH", "EN"],
  "switch_points": [5],
  "pos_tags": [
    ["我", "PRON"],
    ["刚", "ADV"],
    ["刚", "ADV"],
    ["开", "VERB"],
    ["始", "VERB"],
    ["record", "PROPN"]
  ]
}
```

## Important: Why UPOS?

Uses `token.pos` (Universal POS) not `token.tag_` (language-specific)
- Same tags for English and Chinese = directly comparable
- Standard across all languages
- Based on [Universal Dependencies](https://universaldependencies.org/)

Common UPOS tags: NOUN, VERB, ADJ, ADV, PRON, INTJ, PROPN, PART, ADP, DET

## Key Statistics From Sample

- 5 utterances, 52 tokens
- Most common: VERB (25%), NOUN (25%)
- 9 different POS tags used
- Works with mixed EN-ZH utterances

## Next Steps

1. ✓ Prepare your real data (tokens only is fine)
2. Run: `python preprocessing/end_to_end_tagger.py --input YOUR_DATA.json --output OUTPUT.json`
3. Get output with UPOS tags for each token
4. Ready for downstream tasks (parsing, NER, classification, etc.)

## Troubleshooting

**"Module not found" on import?**
- Make sure you're in the workspace root directory
- Python path includes preprocessing/

**Low accuracy on certain tags?**
- This is a baseline using only spaCy's pre-trained models
- Consider annotating data for fine-tuning
- Language labels help select correct model per token

**Tokenization misalignment?**
- Ensure preprocessed tokens match the exact text
- Use provided tokenizer.py for consistency

## Files Summary

```
preprocessing/
  ├── pos_tagger.py              (Main class)
  ├── language_labels.py         (Language detection)
  ├── end_to_end_tagger.py       (Complete pipeline)
  ├── pos_tagger_demo.py         (Demo)
  └── tokenizer.py               (Original tokenizer)

evaluation/
  └── baseline_metrics.py        (Evaluation tools)

data/
  ├── sample_preprocessed.json    (Input: tokens only)
  ├── sample_preprocessed_with_labels.json  (With labels)
  ├── final_tagged_output.json    (Fully tagged output)
  └── sample_preprocessed_with_pos_tags.json (Demo output)

POS_TAGGER_README.md             (Full documentation)
QUICK_REFERENCE.md               (This file)
```

## Contact & Next Stage

The tagger is ready for:
- ✓ Tagging your full ASCEND dataset
- ✓ Analysis of English vs Chinese POS patterns
- ✓ Code-switching behavior analysis
- ✓ Building features for downstream models

See `POS_TAGGER_README.md` for comprehensive documentation.
