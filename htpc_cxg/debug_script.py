"""
Debug script for the Predictive Coding Construction Grammar System.

This script helps diagnose issues with data flow and output structure.
"""

from main_module import MainModule
import json


def pretty_print_dict(d, indent=0):
    """
    Pretty print a dictionary with indentation.

    Args:
        d: Dictionary to print
        indent: Current indentation level
    """
    for key, value in d.items():
        print(' ' * indent + str(key) + ':', end=' ')
        if isinstance(value, dict):
            print()
            pretty_print_dict(value, indent + 4)
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                print(f"[List of {len(value)} items]")
                for i, item in enumerate(value[:2]):  # Show first two items
                    print(' ' * (indent + 4) + f"Item {i}:")
                    pretty_print_dict(item, indent + 8)
                if len(value) > 2:
                    print(' ' * (indent + 4) + f"... and {len(value) - 2} more items")
            else:
                print(f"{value[:5]} ...")
        else:
            print(str(value))


def main():
    """
    Main function to test and debug the system.
    """
    print("Initializing Construction Grammar Predictive Coding System...")

    # Define some predefined constructions
    predefined_constructions = [
        ('DET', 'NOUN'),
        ('DET', 'ADJ', 'NOUN'),
        ('VERB', 'DET', 'NOUN')
    ]

    # Initialize the system
    system = MainModule(predefined_constructions=predefined_constructions)

    # Sample sequence
    sequence = ['DET', 'ADJ', 'NOUN', 'VERB', 'DET', 'NOUN']
    print(f"\nProcessing sequence: {sequence}")

    # Process the sequence
    results = system.process_sequence(sequence, bidirectional=True)

    # Inspect results
    print("\n=== Full Results Structure ===")
    pretty_print_dict(results)

    # Check specific parts
    print("\n=== Checking for Constructions ===")
    if 'constructions' in results:
        print("Found directly in results")
        if results['constructions']:
            print(f"Contains: {list(results['constructions'].keys())}")
    elif 'combined' in results and 'constructions' in results['combined']:
        print("Found in results['combined']")
        print(f"Contains: {list(results['combined']['constructions'].keys())}")
    else:
        print("Not found!")
        # Look for any field that might contain construction info
        for key, value in results.items():
            if isinstance(value, dict):
                if any(k.startswith('const') or 'construct' in k.lower() for k in value.keys()):
                    print(f"Potential construction data in results['{key}']")
                    pretty_print_dict(value, 4)

    # Check for attention
    print("\n=== Checking for Attention ===")
    if 'attention' in results:
        print("Found directly in results")
        if results['attention']:
            print(f"Contains: {list(results['attention'].keys())}")
    elif 'combined' in results and 'attention' in results['combined']:
        print("Found in results['combined']")
        print(f"Contains: {list(results['combined']['attention'].keys())}")
    else:
        print("Not found!")

    # Check for predictions
    print("\n=== Checking for Predictions ===")
    if 'predictions' in results:
        print("Found directly in results")
        if results['predictions']:
            print(f"Contains: {list(results['predictions'].keys())}")
    elif 'combined' in results and 'predictions' in results['combined']:
        print("Found in results['combined']")
        print(f"Contains: {list(results['combined']['predictions'].keys())}")
    else:
        print("Not found!")


if __name__ == "__main__":
    main()