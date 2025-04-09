"""
Debug script for the Hierarchical Temporal Predictive Coding system.
This script helps diagnose issues with construction recognition.
"""

from htpc_main import HTPCSystem
import pprint


def print_construction_details(level, construction_dict):
    """Print detailed information about constructions at a level."""
    print(f"\nCONSTRUCTIONS AT LEVEL {level}:")
    if not construction_dict:
        print("  No constructions found")
        return

    for const_id, const_info in construction_dict.items():
        print(f"  Construction: {const_id}")
        print(f"    Pattern: {const_info.get('pattern', 'unknown')}")
        print(f"    Count: {const_info.get('count', 0)}")
        print(f"    Instances: {len(const_info.get('instances', []))}")
        print()


def main():
    # Initialize the system
    print("Initializing HTPC system...")
    system = HTPCSystem(num_hierarchical_levels=3)

    # Define some simple test sequences
    test_sequences = [
        ['DET', 'NOUN', 'VERB', 'DET', 'NOUN'],
        ['DET', 'ADJ', 'NOUN', 'VERB', 'DET', 'NOUN'],
        ['NOUN', 'VERB', 'DET', 'NOUN'],
        ['DET', 'NOUN', 'VERB', 'NOUN'],
        ['PRON', 'VERB', 'DET', 'NOUN'],
    ]

    # Process each sequence with verbose output
    for idx, sequence in enumerate(test_sequences):
        print(f"\n{'=' * 80}")
        print(f"PROCESSING SEQUENCE {idx + 1}: {sequence}")
        results = system.process_sequence(sequence)

        # Print level outputs
        if 'level_outputs' in results:
            for level, level_output in results['level_outputs'].items():
                if level == 0:  # POS level
                    patterns = level_output.get('patterns', [])
                    print(f"\nLevel {level} patterns: {len(patterns)}")
                    for p in patterns[:5]:  # First 5 patterns
                        print(f"  {p.get('pattern', 'unknown')}, Pos: {p.get('start', '?')}-{p.get('end', '?')}")
                else:
                    constructions = level_output.get('construction_instances', [])
                    print(f"\nLevel {level} constructions: {len(constructions)}")
                    for c in constructions[:5]:  # First 5 constructions
                        print(f"  ID: {c.get('id', 'unknown')}, Pos: {c.get('start', '?')}-{c.get('end', '?')}")
                        if 'items' in c:
                            print(f"    Items: {[str(item) for item in c['items']]}")

        print(f"\nPREDICTIONS:")
        if 'predictions' in results:
            for level, level_preds in results['predictions'].items():
                print(f"  Level {level}: {len(level_preds)} predictions")
                for pos, pred in list(level_preds.items())[:3]:
                    print(f"    Position {pos}: {pred.get('type', 'unknown')}")
                    if 'probabilities' in pred:
                        probs = pred['probabilities']
                        top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                        print(f"      Top predictions: {top_probs}")

    # Print final state of constructions at each level
    print("\n" + "=" * 80)
    print("FINAL SYSTEM STATE:")
    constructions = system.get_constructions()

    for level, level_constructions in constructions.items():
        print_construction_details(level, level_constructions)

    # Try a prediction
    print("\nTESTING PREDICTION:")
    test_partial = ['DET', 'NOUN']
    print(f"Partial sequence: {test_partial}")
    predictions = system.predict_next_pos(test_partial)
    print("Predicted next POS:")
    for pos, prob in predictions:
        print(f"  {pos}: {prob:.3f}")


if __name__ == "__main__":
    main()