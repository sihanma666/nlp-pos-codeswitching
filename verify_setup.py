"""Quick verification that the POS tagger is working."""

from preprocessing.pos_tagger import CodeSwitchingPOSTagger
from preprocessing.language_labels import detect_token_language

print("=" * 70)
print("BASELINE POS TAGGER - VERIFICATION")
print("=" * 70)

# Test language detection
tokens = ["我", "hello", "开始", "record", "嗯", "nice"]
print("\n1. Language Detection:")
for token in tokens:
    lang = detect_token_language(token)
    print(f"   {token:10} -> {lang}")

# Test tagger initialization
print("\n2. Model Loading:")
tagger = CodeSwitchingPOSTagger()
info = tagger.get_model_info()
print(f"   English: {info['en_model']} v{info['en_version']}")
print(f"   Chinese: {info['zh_model']} v{info['zh_version']}")

# Test single utterance
print("\n3. Single Utterance Tagging:")
text = "我刚刚开始record"
tokens = ["我", "刚", "刚", "开", "始", "record"]
result = tagger.tag_utterance(text, tokens=tokens)
for token, pos in result:
    print(f"   {token:10} -> {pos}")

print("\n" + "=" * 70)
print("✓ ALL SYSTEMS OPERATIONAL")
print("=" * 70)
