#!/usr/bin/env python3
"""
SEQUITUR Grammar Generator with Independent Sequence Processing

This script reads sequences from an input file, processes each one independently
with scikit-sequitur, and combines the results into a single grammar while
tracking rule usage statistics for weighting connections in a relational network.

Usage:
    python sequitur_independent_grammar.py input_sequences.txt grammar.txt
"""

import sys
import os
import re
from sksequitur import parse
from collections import Counter, defaultdict


def read_sequences(filename):
    """Read sequences from a text file, one sequence per line."""
    sequences = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                sequences.append(line)
    return sequences


def process_sequences_independently(sequences):
    """
    Process each sequence independently and merge the resulting grammars.

    Returns a merged grammar with usage statistics.
    """
    all_rules = {}
    rule_id_counter = 1  # Start from 1, reserve 0 for the root rule

    # Keep track of rule usage
    rule_usage = Counter()

    # Process each sequence separately
    sequence_grammars = []
    for i, sequence in enumerate(sequences):
        # Parse with scikit-sequitur
        grammar = parse(sequence)
        grammar_str = str(grammar)

        # Convert to a dictionary representation
        rules = {}
        rule_pattern = re.compile(r'(\d+) -> (.*)')

        for line in grammar_str.strip().split('\n'):
            match = rule_pattern.match(line)
            if match:
                rule_id, expansion = match.groups()
                rules[rule_id] = expansion.split()

        sequence_grammars.append(rules)

    # First, process each grammar independently to find unique rules
    for grammar_rules in sequence_grammars:
        # Process rules in depth-first order to ensure dependencies are resolved
        process_grammar_rules(grammar_rules, all_rules, rule_id_counter, rule_usage)

    # Create a new root rule that connects to the top-level rules of each sequence
    root_rule_parts = []
    for i, grammar_rules in enumerate(sequence_grammars):
        if "0" in grammar_rules:  # The original top-level rule
            # Find the corresponding rule in all_rules
            expansion = grammar_rules["0"]
            sub_rule_key = make_rule_key(expansion)

            if sub_rule_key in all_rules:
                for rule_id, rule_data in all_rules.items():
                    if rule_data["key"] == sub_rule_key:
                        root_rule_parts.append(rule_id)
                        break

    # Create the final root rule
    all_rules["0"] = {
        "expansion": root_rule_parts,
        "key": make_rule_key(root_rule_parts),
        "original": True
    }

    # Count and normalize rule usage
    for rule_id in all_rules.keys():
        if rule_id != "0":  # Skip the root rule
            rule_usage[rule_id] += sum(1 for r in all_rules.values()
                                       if rule_id in r["expansion"])

    # Calculate a normalized weight for each rule
    max_usage = max(rule_usage.values()) if rule_usage else 1
    rule_weights = {}

    for rule_id, count in rule_usage.items():
        # Normalize weight between 0.1 and 1.0
        weight = 0.1 + 0.9 * (count / max_usage) if max_usage > 0 else 0.1
        rule_weights[rule_id] = {"count": count, "weight": weight}

    # Add weights for rules that weren't explicitly referenced
    for rule_id in all_rules.keys():
        if rule_id not in rule_weights:
            rule_weights[rule_id] = {"count": 0, "weight": 0.1}

    return all_rules, rule_weights


def make_rule_key(expansion):
    """Create a unique key for a rule based on its expansion."""
    return ','.join(expansion)


def process_grammar_rules(grammar_rules, all_rules, rule_id_counter, rule_usage):
    """
    Process grammar rules to find unique rules and track references.
    Updates all_rules and rule_id_counter in place.
    """
    # Create a mapping from original rule IDs to new rule IDs
    rule_mapping = {}

    # First pass: create rule keys and mapping
    for rule_id, expansion in grammar_rules.items():
        rule_key = make_rule_key(expansion)

        # Check if this expansion already exists
        existing_id = None
        for existing_rule_id, rule_data in all_rules.items():
            if rule_data["key"] == rule_key:
                existing_id = existing_rule_id
                break

        if existing_id:
            # Use the existing rule ID
            rule_mapping[rule_id] = existing_id
        else:
            # Create a new rule ID
            new_id = str(rule_id_counter)
            rule_id_counter += 1
            rule_mapping[rule_id] = new_id

            # Add the rule, but we'll process expansions in the second pass
            all_rules[new_id] = {
                "expansion": expansion,
                "key": rule_key,
                "original": rule_id == "0"  # Is this a top-level rule?
            }

    # Second pass: update expansions to use new rule IDs
    for rule_id, rule_data in list(all_rules.items()):
        new_expansion = []
        for symbol in rule_data["expansion"]:
            if symbol in grammar_rules:  # It's a rule reference
                new_expansion.append(rule_mapping[symbol])
                # Increment usage count
                rule_usage[rule_mapping[symbol]] += 1
            else:  # It's a terminal symbol
                new_expansion.append(symbol)

        rule_data["expansion"] = new_expansion
        rule_data["key"] = make_rule_key(new_expansion)


def format_grammar_output(rules, weights):
    """Format the grammar rules and weights for output."""
    output = []

    # Add header
    output.append("# Grammar generated by scikit-sequitur with rule usage counts")
    output.append("# Format: rule -> expansion (space-separated symbols) [usage_count, weight]")
    output.append("# Weights are normalized between 0.1 and 1.0 based on usage frequency")
    output.append("")

    # Add rules with usage statistics
    sorted_rules = sorted(rules.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))

    for rule_id in sorted_rules:
        expansion = rules[rule_id]["expansion"]
        count = weights[rule_id]["count"]
        weight = weights[rule_id]["weight"]

        line = f"{rule_id} -> {' '.join(expansion)} [count={count}, weight={weight:.2f}]"
        output.append(line)

    output.append("")
    output.append("# Usage Statistics Summary")
    output.append("# Format: rule_id: usage_count, normalized_weight")

    for rule_id in sorted_rules:
        count = weights[rule_id]["count"]
        weight = weights[rule_id]["weight"]
        output.append(f"# {rule_id}: {count}, {weight:.2f}")

    output.append("")
    output.append("# Raw Rule Expansions")
    for rule_id in sorted_rules:
        expansion = rules[rule_id]["expansion"]
        output.append(f"# {rule_id}: {' '.join(expansion)}")

    return "\n".join(output)


def main():
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python sequitur_independent_grammar.py <input_file> <output_file>")
        return 1

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return 1

    try:
        # Read the sequences
        print(f"Reading sequences from '{input_file}'...")
        sequences = read_sequences(input_file)
        print(f"Read {len(sequences)} sequences.")

        # Process sequences
        print("Processing sequences independently with scikit-sequitur...")
        rules, weights = process_sequences_independently(sequences)
        print(f"Generated {len(rules)} unique rules.")

        # Format and write grammar to output file
        print(f"Writing grammar to '{output_file}'...")
        grammar_output = format_grammar_output(rules, weights)

        with open(output_file, 'w') as f:
            f.write(grammar_output)

        print("Done!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())