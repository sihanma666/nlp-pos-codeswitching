import re
import json
from preprocessing.data_loader import get_transcriptions


def simple_tokenize(text):
    """
    A simple tokenizer for mixed Chinese-English text.

    Rules:
    1. Consecutive English letters/numbers stay together as one token.
       Example: "hello123" -> ["hello123"]
    2. Each Chinese character is treated as one token.
       Example: "我的名字" -> ["我", "的", "名", "字"]
    3. Punctuation is kept as separate tokens.
    4. Spaces are ignored.
    """
    if not isinstance(text, str):
        return []

    text = text.strip()
    if not text:
        return []

    pattern = r"[A-Za-z0-9]+|[\u4e00-\u9fff]|[^\w\s]"
    tokens = re.findall(pattern, text)
    return tokens


def preprocess_data(path, limit=None):
    texts = get_transcriptions(path, limit=limit)
    processed = []

    for i, text in enumerate(texts):
        tokens = simple_tokenize(text)
        processed.append({
            "id": i,
            "text": text,
            "tokens": tokens
        })

    return processed

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    path = "external_data/ASCEND/main/train-00000-of-00003.parquet"
    data = preprocess_data(path, limit=5)

    for item in data:
        print("ID:", item["id"])
        print("TEXT:", item["text"])
        print("TOKENS:", item["tokens"])
        print("-" * 40)
        
    save_json(data, "data/sample_preprocessed.json")
    print("Saved sample output to data/sample_preprocessed.json")
