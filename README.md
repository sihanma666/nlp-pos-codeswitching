# NLP POS Code-Switching project 

## Preprocessing

We use the ASCEND dataset and extract the `transcription` field.

Each sentence is tokenized into a sequence of tokens using a simple rule-based tokenizer:

- English words are kept as full tokens
- Chinese characters are split into individual tokens

Example:

Input:
"嗯hello我的名字叫徐妍"

Output:
["嗯", "hello", "我", "的", "名", "字", "叫", "徐", "妍"]

The output format for each example is:

{
  "id": int,
  "text": str,
  "tokens": list[str]
}

A sample output is provided in `data/sample_preprocessed.json`.

## POS Tagging - Baseline Pipeline

This project includes a **baseline POS tagger** for code-switched utterances between Mandarin Chinese and English.

### Quick Start

**1. Install dependencies:**
```bash
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

**2. Run the tagger on your data:**
```bash
# With original preprocessed data (auto-generates language labels)
python preprocessing/end_to_end_tagger.py --input data/sample_preprocessed.json --output data/output.json

# Or use the demo with sample data that has labels
python -m preprocessing.pos_tagger_demo
```

### Key Features

- ✓ Uses spaCy models with **Universal POS (UPOS)** tags (`token.pos`) for language-independent comparison
- ✓ Processes utterances through both EN and ZH models, merging results based on language labels
- ✓ Automatically detects language of tokens (Chinese vs English)
- ✓ Properly aligns preprocessed tokens with spaCy's tokenization

### Output Format

Each tagged utterance includes:
```json
{
  "id": 0,
  "text": "我刚刚开始record",
  "tokens": ["我", "刚", "刚", "开", "始", "record"],
  "language_labels": ["ZH", "ZH", "ZH", "ZH", "ZH", "EN"],
  "switch_points": [5],
  "pos_tags": [["我", "PRON"], ["刚", "ADV"], ...],
  "tokens_with_pos": [{"token": "我", "pos": "PRON"}, ...]
}
```

### Available Modules

- **`preprocessing/pos_tagger.py`**: Main `CodeSwitchingPOSTagger` class
- **`preprocessing/language_labels.py`**: Label tokens by language and detect switch points
- **`preprocessing/end_to_end_tagger.py`**: Complete pipeline script (recommended for new data)
- **`preprocessing/pos_tagger_demo.py`**: Demo on sample data
- **`evaluation/baseline_metrics.py`**: Evaluation utilities
- **`POS_TAGGER_README.md`**: Detailed documentation

### Example Usage

```python
from preprocessing.pos_tagger import CodeSwitchingPOSTagger

# Initialize
tagger = CodeSwitchingPOSTagger()

# Tag data
results = tagger.tag_batch(data)

# Results include pos_tags: [(token, UPOS), ...]
```

See `POS_TAGGER_README.md` for complete documentation.