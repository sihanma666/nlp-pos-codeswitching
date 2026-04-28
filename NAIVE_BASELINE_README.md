# Naive Baseline POS Tagger

## Overview

The naive baseline mode processes **all tokens through the English spaCy model only**, with **no language detection**. This provides a simple baseline to demonstrate POS tagging degradation on Chinese and code-switch tokens.

## Usage

### Command Line

Run the end-to-end pipeline with naive baseline mode:

```bash
# Naive baseline (English only)
python preprocessing/end_to_end_tagger.py --input data/sample_preprocessed.json --output data/baseline_output.json --mode naive_baseline

# Smart code-switching mode (default)
python preprocessing/end_to_end_tagger.py --input data/sample_preprocessed.json --output data/tagged_output.json --mode codeswitching
```

### Python API

```python
from preprocessing.pos_tagger import NaiveBaselinePOSTagger

# Initialize with English model only
tagger = NaiveBaselinePOSTagger(en_model_name="en_core_web_sm")

# Tag a single utterance
text = "我刚刚开始record"
tokens = ["我", "刚", "刚", "开", "始", "record"]
results = tagger.tag_utterance(text, tokens=tokens)
# Returns: [("我", POS), ("刚", POS), ..., ("record", POS)]

# Tag a batch of utterances
data = [
    {"id": 0, "text": "我开始record", "tokens": ["我", "开", "始", "record"]},
    {"id": 1, "text": "你在做什么", "tokens": ["你", "在", "做", "什", "么"]},
]
tagged_results = tagger.tag_batch(data)
```

## Key Differences from Smart Code-Switching Mode

| Feature | Naive Baseline | Code-Switching (Smart) |
|---------|---|---|
| Language Detection | ❌ No | ✅ Yes |
| Models Used | English only | English + Chinese |
| Chinese Token Handling | Tagged with English model | Tagged with Chinese model |
| Switch Point Aware | ❌ No | ✅ Yes |
| Expected Performance | Low on mixed-language | High on all language types |

## Use Case: Baseline Comparison

Use this to show degradation:
1. Run smart mode and save results
2. Run naive baseline mode and save results
3. Compare POS accuracy on:
   - Pure English tokens
   - Pure Chinese tokens
   - Code-switch points (tokens at language boundaries)

Example comparison script:

```python
from preprocessing.pos_tagger import CodeSwitchingPOSTagger, NaiveBaselinePOSTagger

data = load_test_data()

# Smart tagger
smart_tagger = CodeSwitchingPOSTagger()
smart_results = smart_tagger.tag_batch(data)

# Naive baseline
naive_tagger = NaiveBaselinePOSTagger()
naive_results = naive_tagger.tag_batch(data)

# Compare results by language type
# - Chinese tokens: naive baseline should perform worse
# - Switch points: naive baseline should perform worse
# - English tokens: similar performance
```

## Implementation Details

The `NaiveBaselinePOSTagger` class:
- Only loads the English spaCy model
- Ignores `language_labels` parameter for compatibility
- Uses the same token alignment strategy as the smart tagger
- Maintains the same output format for easy comparison

## File Structure

- `preprocessing/pos_tagger.py`: Contains both `CodeSwitchingPOSTagger` and `NaiveBaselinePOSTagger`
- `preprocessing/end_to_end_tagger.py`: Pipeline with `--mode` flag support
