"""
Complete end-to-end POS tagging pipeline.

Takes preprocessed data (tokens only) and produces POS-tagged output
by automatically adding language labels and switch points.

Usage:
    python preprocessing/end_to_end_tagger.py --input data/sample_preprocessed.json --output data/tagged_output.json
"""

import json
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.language_labels import add_language_labels_to_data
from preprocessing.pos_tagger import CodeSwitchingPOSTagger, print_results


def process_file(input_path: str, output_path: str, verbose: bool = True) -> dict:
    """
    Complete end-to-end processing pipeline.

    Args:
        input_path: Path to input JSON file with preprocessed tokens
        output_path: Path for output JSON file with POS tags
        verbose: Whether to print progress and results

    Returns:
        Dict with processing statistics
    """
    if verbose:
        print("=" * 70)
        print("END-TO-END POS TAGGING PIPELINE")
        print("=" * 70)

    # Step 1: Load input data
    if verbose:
        print(f"\n[1/4] Loading preprocessed data from {input_path}...")

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if verbose:
        print(f"✓ Loaded {len(data)} utterances")

    # Step 2: Add language labels and switch points
    if verbose:
        print(f"\n[2/4] Adding language labels and switch points...")

    # Check if labels already exist
    has_labels = all("language_labels" in item for item in data)
    if has_labels:
        if verbose:
            print("✓ Language labels already present, skipping...")
        enhanced_data = data
    else:
        enhanced_data = add_language_labels_to_data(data)
        if verbose:
            print(f"✓ Added language labels and switch points")

    # Step 3: Initialize and tag with POS tagger
    if verbose:
        print(f"\n[3/4] Initializing POS tagger and tagging utterances...")

    try:
        tagger = CodeSwitchingPOSTagger(
            en_model_name="en_core_web_sm", zh_model_name="zh_core_web_sm"
        )
        if verbose:
            info = tagger.get_model_info()
            print(f"✓ English model: {info['en_model']} v{info['en_version']}")
            print(f"✓ Chinese model: {info['zh_model']} v{info['zh_version']}")
    except OSError as e:
        print(f"✗ Error loading models: {e}")
        raise

    # Tag the batch
    tagged_results = tagger.tag_batch(enhanced_data)
    if verbose:
        print(f"✓ Tagged {len(tagged_results)} utterances")

    # Step 4: Save output
    if verbose:
        print(f"\n[4/4] Saving tagged results to {output_path}...")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tagged_results, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"✓ Results saved to {output_path}")

    # Compute statistics
    total_tokens = sum(len(item.get("tokens_with_pos", [])) for item in tagged_results)
    pos_counts = {}
    for item in tagged_results:
        for token_pos in item.get("tokens_with_pos", []):
            pos = token_pos["pos"]
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  Total utterances: {len(tagged_results)}")
        print(f"  Total tokens tagged: {total_tokens}")
        print(f"  Unique POS tags: {len(pos_counts)}")
        print(f"\nMost common POS tags:")
        sorted_tags = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
        for tag, count in sorted_tags[:10]:
            percentage = (count / total_tokens) * 100
            print(f"    {tag:10}: {count:4} ({percentage:5.1f}%)")

        print(f"\nSample output (first 2 utterances):")
        print_results(tagged_results, num_examples=2)

    return {
        "total_utterances": len(tagged_results),
        "total_tokens": total_tokens,
        "unique_pos_tags": len(pos_counts),
        "pos_distribution": pos_counts,
        "output_file": str(output_path),
    }


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="End-to-end POS tagging pipeline for code-switched utterances"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_preprocessed.json",
        help="Path to input JSON file with preprocessed tokens",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample_preprocessed_with_pos_tags.json",
        help="Path for output JSON file with POS tags",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    try:
        stats = process_file(args.input, args.output, verbose=not args.quiet)
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
