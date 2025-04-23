#!/usr/bin/env python3
"""
SEQUITUR Grammar Generator with Token-Based Processing

This script reads sequences from an input file, tokenizes them,
processes each sequence independently with scikit-sequitur,
and generates both a grammar file and a GraphViz DOT file for visualization.

Usage:
    python sequitur_token_grammar.py input_sequences.txt grammar.txt grammar.dot
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


def tokenize(sequence):
    """
    Convert a string into a list of tokens.
    Splits on whitespace and treats each word as a single token.
    """
    # Simple tokenization by splitting on whitespace
    return sequence.split()


def process_sequences_independently(sequences):
    """
    Process each tokenized sequence independently and merge the resulting grammars.

    Returns a merged grammar with usage statistics.
    """
    all_rules = {}
    rule_id_counter = 1  # Start from 1, reserve 0 for the root rule

    # Keep track of rule usage
    rule_usage = Counter()

    # Process each sequence separately
    sequence_grammars = []
    for i, sequence in enumerate(sequences):
        # Tokenize the sequence
        tokens = tokenize(sequence)

        # Parse with scikit-sequitur (using tokens instead of characters)
        grammar = parse(tokens)
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
    output.append("# Tokens are preserved as-is; whitespace between tokens is not included in the grammar")
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


def generate_dot_graph(rules, weights, title="SEQUITUR Token Grammar Visualization"):
    """Generate a GraphViz DOT representation of the grammar."""
    dot = []

    # Header
    dot.append("digraph Grammar {")
    dot.append("  graph [fontname=\"Arial\", fontsize=12, label=\"" + title + "\", labelloc=t];")
    dot.append("  node [fontname=\"Arial\", fontsize=10, style=\"filled\"];")
    dot.append("  edge [fontname=\"Arial\", fontsize=8];")
    dot.append("")

    # Node definitions
    for rule_id, rule_data in rules.items():
        # Determine if this is a rule or a terminal
        is_rule = rule_id.isdigit()

        if is_rule:
            # Get weight info
            count = weights[rule_id]["count"] if rule_id in weights else 0
            weight = weights[rule_id]["weight"] if rule_id in weights else 0.1

            # Color intensity based on weight (more frequent = more intense)
            color_intensity = int(255 - (weight * 155))  # Range 100-255 (darker for important rules)
            color = f"\"#FFEC{color_intensity:02X}\""

            # Create node with weight info
            dot.append(
                f"  \"rule_{rule_id}\" [label=\"Rule {rule_id}\\nw={weight:.2f}, count={count}\", shape=box, fillcolor={color}];")
        else:
            # Terminal symbol (no rule expansion)
            # Escape special characters in the rule_id
            safe_id = rule_id.replace("\"", "\\\"").replace("\\", "\\\\")
            dot.append(f"  \"terminal_{safe_id}\" [label=\"{safe_id}\", shape=ellipse, fillcolor=\"#B3E5FC\"];")

    # Add terminal nodes for symbols that appear in expansions
    terminal_nodes = set()
    for rule_id, rule_data in rules.items():
        for symbol in rule_data["expansion"]:
            if not symbol.isdigit():
                terminal_nodes.add(symbol)

    for symbol in terminal_nodes:
        safe_symbol = symbol.replace("\"", "\\\"").replace("\\", "\\\\")
        dot.append(f"  \"terminal_{safe_symbol}\" [label=\"{safe_symbol}\", shape=ellipse, fillcolor=\"#B3E5FC\"];")

    dot.append("")

    # Edge definitions
    for rule_id, rule_data in rules.items():
        if not rule_id.isdigit():
            continue  # Skip non-numeric rules

        # Add connections from this rule to its components
        for i, symbol in enumerate(rule_data["expansion"]):
            if symbol.isdigit():
                # Connection to another rule
                target = f"\"rule_{symbol}\""

                # Get the weight of the target rule
                target_weight = weights[symbol]["weight"] if symbol in weights else 0.1

                # Edge thickness based on target weight
                penwidth = 1 + (target_weight * 3)

                dot.append(f"  \"rule_{rule_id}\" -> {target} [label=\"{i}\", penwidth={penwidth}];")
            else:
                # Connection to a terminal
                safe_symbol = symbol.replace("\"", "\\\"").replace("\\", "\\\\")
                dot.append(f"  \"rule_{rule_id}\" -> \"terminal_{safe_symbol}\" [label=\"{i}\"];")

    # Highlight the root rule
    dot.append("  \"rule_0\" [penwidth=2, color=\"#4CAF50\"];")

    dot.append("}")

    return "\n".join(dot)


def main():
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python sequitur_token_grammar.py <input_file> <grammar_file> [dot_file]")
        return 1

    input_file = sys.argv[1]
    grammar_file = sys.argv[2]
    dot_file = sys.argv[3] if len(sys.argv) > 3 else grammar_file.replace('.txt', '.dot')

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
        print(f"Writing grammar to '{grammar_file}'...")
        grammar_output = format_grammar_output(rules, weights)

        with open(grammar_file, 'w') as f:
            f.write(grammar_output)

        # Generate and write DOT file for GraphViz
        print(f"Writing GraphViz DOT file to '{dot_file}'...")
        title = f"SEQUITUR Grammar for {len(sequences)} Tokenized Sequences"
        dot_output = generate_dot_graph(rules, weights, title)

        with open(dot_file, 'w') as f:
            f.write(dot_output)

        print("Done!")
        print(f"To generate a visualization, run: dot -Tsvg {dot_file} -o {dot_file.replace('.dot', '.svg')}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())