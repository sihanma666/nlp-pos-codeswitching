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