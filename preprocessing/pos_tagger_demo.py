"""
Baseline POS Tagger for Code-Switching Detection

Demonstrates how to use the CodeSwitchingPOSTagger on the ASCEND dataset
with code-switching between Mandarin Chinese and English.

Usage:
    python -m preprocessing.pos_tagger_demo
"""

import json
import sys
from pathlib import Path
from preprocessing.pos_tagger import CodeSwitchingPOSTagger, print_results


def load_json(file_path):
    """Load JSON data from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path):
    """Save JSON data to file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    """Main demonstration of the POS tagger."""
    print("=" * 70)
    print("CODE-SWITCHING POS TAGGER - BASELINE PIPELINE")
    print("=" * 70)

    # Initialize the tagger
    print("\nInitializing POS tagger...")
    try:
        tagger = CodeSwitchingPOSTagger(
            en_model_name="en_core_web_sm", zh_model_name="zh_core_web_sm"
        )
        print("✓ Tagger initialized successfully")

        # Print model info
        model_info = tagger.get_model_info()
        print(f"\nLoaded Models:")
        print(f"  English: {model_info['en_model']} (v{model_info['en_version']})")
        print(f"  Chinese: {model_info['zh_model']} (v{model_info['zh_version']})")

    except OSError as e:
        print(f"✗ Error loading models: {e}")
        print("\nTo fix, run:")
        print("  python -m spacy download en_core_web_sm")
        print("  python -m spacy download zh_core_web_sm")
        sys.exit(1)

    # Load sample data
    sample_data_path = Path("data/sample_preprocessed_with_labels.json")
    if not sample_data_path.exists():
        print(f"\n✗ Sample data file not found: {sample_data_path}")
        print("Creating it now...")
        # This would have been created in the earlier step
        sys.exit(1)

    print(f"\nLoading sample data from {sample_data_path}...")
    data = load_json(sample_data_path)
    print(f"✓ Loaded {len(data)} utterances")

    # Tag the batch
    print("\nProcessing utterances through POS tagger...")
    tagged_results = tagger.tag_batch(data)
    print(f"✓ Tagged all {len(tagged_results)} utterances")

    # Display results
    print("\n" + "=" * 70)
    print("SAMPLE RESULTS (first 3 utterances)")
    print("=" * 70)
    print_results(tagged_results, num_examples=3)

    # Save results
    output_path = Path("data/sample_preprocessed_with_pos_tags.json")
    save_json(tagged_results, output_path)
    print(f"\n✓ Results saved to {output_path}")

    # Print statistics
    print("\n" + "=" * 70)
    print("POS TAG STATISTICS")
    print("=" * 70)

    pos_counts = {}
    total_tokens = 0

    for item in tagged_results:
        for token_pos in item.get("tokens_with_pos", []):
            pos = token_pos["pos"]
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
            total_tokens += 1

    print(f"\nTotal tokens tagged: {total_tokens}")
    print(f"Unique POS tags: {len(pos_counts)}")
    print("\nPOS Tag Distribution:")

    # Sort by frequency
    sorted_tags = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
    for tag, count in sorted_tags:
        percentage = (count / total_tokens) * 100
        print(f"  {tag:10} : {count:3} ({percentage:5.1f}%)")

    print("\n" + "=" * 70)
    print("BASELINE POS TAGGER SETUP COMPLETE")
    print("=" * 70)
    print("\nNotes:")
    print("  - Uses spaCy Universal POS (UPOS) tags via token.pos")
    print("  - UPOS tags are language-independent and directly comparable")
    print("  - English tokens are tagged using en_core_web_sm")
    print("  - Chinese tokens are tagged using zh_core_web_sm")
    print("  - Language labels guide which model's output to use per token")


if __name__ == "__main__":
    main()
