# POS Tagging Baseline Pipeline for Code-Switching Detection

## Overview

This baseline POS tagger is designed for multi-lingual utterances that switch between **Mandarin Chinese** and **English**. The pipeline uses spaCy's pre-trained models with **Universal POS (UPOS)** tags, which are language-independent and directly comparable across languages.

## Key Features

✓ **Dual-Model Approach**: Processes utterances through both English and Chinese spaCy models  
✓ **Language-Aware Merging**: Uses EN/ZH language labels per token to select the appropriate model's output  
✓ **Universal POS Tags**: Uses `token.pos` (UPOS) instead of `token.tag_` for language-independent comparison  
✓ **Token Alignment**: Properly aligns preprocessed tokens with spaCy's tokenization outputs  

## Setup

### 1. Install Dependencies

```bash
pip install spacy
```

### 2. Download Pre-trained Models

```bash
# English model
python -m spacy download en_core_web_sm

# Chinese model
python -m spacy download zh_core_web_sm
```

## Data Format

### Input Format

Your data should have preprocessed tokens and language labels:

```json
{
  "id": 0,
  "text": "我刚刚开始record",
  "tokens": ["我", "刚", "刚", "开", "始", "record"],
  "language_labels": ["ZH", "ZH", "ZH", "ZH", "ZH", "EN"]
}
```

**Required Fields:**
- `text`: Full utterance as a string
- `tokens`: List of preprocessed tokens (from tokenizer)
- `language_labels`: Language label for each token ("EN" or "ZH")

**Optional Fields:**
- `switch_points`: Indices where code-switching occurs

### Output Format

Tagged results include UPOS tags for each token:

```json
{
  "id": 0,
  "text": "我刚刚开始record",
  "tokens": ["我", "刚", "刚", "开", "始", "record"],
  "language_labels": ["ZH", "ZH", "ZH", "ZH", "ZH", "EN"],
  "pos_tags": [
    ["我", "PRON"],
    ["刚", "ADV"],
    ["刚", "ADV"],
    ["开", "VERB"],
    ["始", "VERB"],
    ["record", "PROPN"]
  ],
  "tokens_with_pos": [
    {"token": "我", "pos": "PRON"},
    {"token": "刚", "pos": "ADV"},
    ...
  ]
}
```

## Usage

### Basic Usage

```python
from preprocessing.pos_tagger import CodeSwitchingPOSTagger
import json

# Initialize tagger
tagger = CodeSwitchingPOSTagger(
    en_model_name="en_core_web_sm",
    zh_model_name="zh_core_web_sm"
)

# Load your data
with open("data/sample_preprocessed_with_labels.json") as f:
    data = json.load(f)

# Tag utterances
results = tagger.tag_batch(data)

# Save results
with open("output.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

### Tagging Individual Utterances

```python
# Simple tagging with auto-detection
result = tagger.tag_utterance(
    text="我刚刚开始record",
    tokens=["我", "刚", "刚", "开", "始", "record"]
)
# Returns: [("我", "PRON"), ("刚", "ADV"), ...]

# With explicit language labels
result = tagger.tag_utterance(
    text="我刚刚开始record",
    tokens=["我", "刚", "刚", "开", "始", "record"],
    language_labels=["ZH", "ZH", "ZH", "ZH", "ZH", "EN"]
)
```

### Adding Language Labels to Existing Data

If you only have tokens but no language labels:

```python
from preprocessing.language_labels import add_language_labels_to_data

# Load data with just tokens
data = json.load(open("data/sample_preprocessed.json"))

# Automatically generate labels
data_with_labels = add_language_labels_to_data(data)

# Now ready to tag
results = tagger.tag_batch(data_with_labels)
```

### Run the Demo

```bash
# This runs a full example on sample data
python -m preprocessing.pos_tagger_demo
```

## Universal POS Tags (UPOS)

The tagger uses the [Universal Dependencies](https://universaldependencies.org/) POS tagset, which is language-independent. Common tags include:

| Tag | Part of Speech | Example |
|-----|---|---|
| NOUN | Noun | 名字 (name), dog |
| VERB | Verb | 开始 (start), run |
| ADJ | Adjective | 好 (good), nice |
| ADV | Adverb | 刚 (just), quickly |
| PRON | Pronoun | 我 (I), you |
| INTJ | Interjection | 嗯 (um), hello |
| PROPN | Proper noun | 徐 (surname), John |
| PART | Particle | 的, 了 | 
| ADP | Adposition | 在, through |
| DET | Determiner | 这 (this), the |

See [UD POS Tags](https://universaldependencies.org/u/pos/) for the complete list.

## Why UPOS?

- **Language-independent**: Same tags across English and Chinese
- **Comparable**: Direct POS comparison between different languages
- **Standard**: Based on Universal Dependencies framework
- **vs. `token.tag_`**: `token.tag_` gives language-specific tags (e.g., NN vs. n), not comparable across languages

## Pipeline Flow

```
Input Utterance (Code-switched text)
        ↓
    [Extract full text and preprocessed tokens]
        ↓
    [Process through EN and ZH spaCy models]
        ↓
    [For each token, use language label to select appropriate model's POS]
        ↓
    [Align spaCy tokens with preprocessed tokens]
        ↓
    [Output: Token + UPOS tag pairs]
```

## Implementation Details

### Key Classes

**`CodeSwitchingPOSTagger`**: Main class for POS tagging
- `__init__()`: Load models
- `tag_utterance()`: Tag a single utterance
- `tag_batch()`: Tag multiple utterances
- `_tag_with_preprocessed_tokens()`: Internal alignment logic

### Key Functions

**`language_labels.py`**:
- `detect_token_language()`: Classify token as EN or ZH
- `label_tokens()`: Label a full token list
- `find_switch_points()`: Find code-switching points
- `add_language_labels_to_data()`: Process batch

**`baseline_metrics.py`**:
- `token_accuracy()`: Overall accuracy
- `language_specific_accuracy()`: Accuracy by language
- `evaluate_batch()`: Full evaluation

## Example Output

```
ID: 2
Text: 嗯初次见面nice to meet you嗯
Tokens & POS Tags:
  嗯               -> INTJ (Chinese: Interjection)
  初               -> NOUN (Chinese: Noun)
  次               -> NOUN (Chinese: Noun)
  见               -> VERB (Chinese: Verb)
  面               -> VERB (Chinese: Verb)
  nice            -> NOUN (English: Adjective/Noun)
  to              -> PART (English: Particle/Preposition)
  meet            -> VERB (English: Verb)
  you             -> NOUN (English: Pronoun/Noun)
  嗯               -> INTJ (Chinese: Interjection)
Language Labels: ['ZH', 'ZH', 'ZH', 'ZH', 'ZH', 'EN', 'EN', 'EN', 'EN', 'ZH']
```

## Next Steps

1. **Prepare your data**: Ensure you have preprocessed tokens and language labels
2. **Run the tagger**: Tag your full dataset
3. **Evaluate**: Compare against gold standard annotations (if available)
4. **Analyze**: Look at confusions and per-language performance

## Troubleshooting

**"Model not found" error**: 
```bash
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

**Misaligned tokens**: 
Ensure your preprocessed tokens match the exact characters in the text. The aligner uses character position matching.

**Low accuracy for certain tags**:
This is expected for a baseline. Consider fine-tuning models or using a more advanced architecture for production use.

## References

- [spaCy Documentation](https://spacy.io/)
- [Universal Dependencies POS Tags](https://universaldependencies.org/u/pos/)
- [spaCy Chinese Model](https://github.com/explosion/spacy-pkuseg)
