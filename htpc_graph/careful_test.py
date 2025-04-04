"""
A careful test that processes one sentence at a time and continues
even if there are errors.
"""
from htpc_graph_builder import HTCPGraphBuilder
import traceback

# Create builder with debug mode
builder = HTCPGraphBuilder(debug=True)

# Configure thresholds
builder.thresholds = {
    "bigram": 2,  # Minimum frequency to create a bigram
    "phrase": 2,  # Minimum frequency to create a phrase
    "hierarchy": 1,  # Minimum frequency to create a hierarchy
    "discourse": 1  # Minimum frequency to create a discourse pattern
}


# Process sentences one by one
def process_file_safely(filename):
    print(f"Processing file: {filename}")

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        traceback.print_exc()
        return False

    success_count = 0
    error_count = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        print(f"\nProcessing line {i + 1}: {line}")
        try:
            # Manually process each part safely
            tokens = builder.tokenize(line)
            if not tokens:
                print("  No tokens found, skipping line")
                continue

            print(f"  Tokenized into: {tokens}")

            # Store for higher-level processing
            builder.sequence_buffer.append(tokens)

            # Process tokens safely
            print("  Processing tokens...")
            try:
                # Process each token
                token_nodes = {}
                for token in tokens:
                    try:
                        # Find or create token node
                        token_node_id = None
                        for node_id, node in builder.graph["nodes"].items():
                            if (node["level"] == "L1" and
                                    node["type"] == "AND" and
                                    node["value"] == token):
                                token_node_id = node_id
                                node["frequency"] += 1
                                break

                        if not token_node_id:
                            token_node_id = builder._create_node("AND", "L1", token)

                        token_nodes[token] = token_node_id
                    except Exception as e:
                        print(f"  Error processing token '{token}': {e}")
                        continue  # Skip this token

                # Process transitions between tokens
                for i in range(len(tokens) - 1):
                    try:
                        current_token = tokens[i]
                        next_token = tokens[i + 1]

                        if current_token not in token_nodes or next_token not in token_nodes:
                            continue  # Skip if either token was not processed

                        current_node_id = token_nodes[current_token]
                        next_node_id = token_nodes[next_token]

                        # Update tracking
                        builder.successors[current_node_id].add(next_node_id)
                        builder.predecessors[next_node_id].add(current_node_id)

                        # Add edge
                        builder._create_edge(current_node_id, next_node_id, "sequence")

                        # Track bigram
                        bigram_key = f"{current_token}_{next_token}"
                        builder.bigram_frequencies[bigram_key] = \
                            builder.bigram_frequencies.get(bigram_key, 0) + 1
                    except Exception as e:
                        print(f"  Error processing transition {tokens[i]} -> {tokens[i + 1]}: {e}")

                # Skip OR node creation for now - we'll do it at the end

                success_count += 1
                print(f"  Successfully processed line {i + 1}")
            except Exception as e:
                print(f"  Error in token processing for line {i + 1}: {e}")
                error_count += 1
                traceback.print_exc()
        except Exception as e:
            print(f"  Error processing line {i + 1}: {e}")
            error_count += 1
            traceback.print_exc()

    print("\nCreating OR nodes...")
    try:
        # Now try to create OR nodes from all the accumulated data
        create_or_nodes_safely(builder)
    except Exception as e:
        print(f"Error creating OR nodes: {e}")
        traceback.print_exc()

    print("\nBuilding higher levels...")
    try:
        # Try to build higher levels
        if len(builder.sequence_buffer) >= 2:
            try_build_higher_levels(builder)
    except Exception as e:
        print(f"Error building higher levels: {e}")
        traceback.print_exc()

    print(f"\nProcessing summary:")
    print(f"  Successfully processed: {success_count} lines")
    print(f"  Errors: {error_count} lines")

    # Update metadata
    builder.update_metadata()

    return success_count > 0


def create_or_nodes_safely(builder):
    """Safely create OR nodes without unpacking errors"""
    # Handle convergence points
    for node_id, pred_set in builder.predecessors.items():
        if len(pred_set) > 1:  # Multiple predecessors
            try:
                pred_list = list(pred_set)
                or_node_id = builder._create_node("OR", "L1", f"OR_in_{node_id}", components=pred_list)

                for pred_id in pred_list:
                    # Connect predecessor to OR node
                    builder._create_edge(pred_id, or_node_id, "alternative")

                # Connect OR node to target
                builder._create_edge(or_node_id, node_id, "sequence")
            except Exception as e:
                print(f"Error creating OR node for convergence at {node_id}: {e}")

    # Handle divergence points
    for node_id, succ_set in builder.successors.items():
        if len(succ_set) > 1:  # Multiple successors
            try:
                succ_list = list(succ_set)
                or_node_id = builder._create_node("OR", "L1", f"OR_out_{node_id}", components=succ_list)

                for succ_id in succ_list:
                    # Connect OR node to successor
                    builder._create_edge(or_node_id, succ_id, "alternative")

                # Connect source to OR node
                builder._create_edge(node_id, or_node_id, "sequence")
            except Exception as e:
                print(f"Error creating OR node for divergence at {node_id}: {e}")


def try_build_higher_levels(builder):
    """Try to build higher levels safely"""
    # Build bigrams
    print("  Building bigrams...")
    try:
        for bigram, freq in builder.bigram_frequencies.items():
            if freq < builder.thresholds["bigram"]:
                continue

            try:
                parts = bigram.split("_")
                if len(parts) < 2:
                    continue

                token1 = parts[0]
                token2 = "_".join(parts[1:]) if len(parts) > 2 else parts[1]

                # Find token nodes
                token1_node_id = None
                token2_node_id = None

                for node_id, node in builder.graph["nodes"].items():
                    if node["level"] == "L1" and node["type"] == "AND":
                        if node["value"] == token1:
                            token1_node_id = node_id
                        elif node["value"] == token2:
                            token2_node_id = node_id

                if not token1_node_id or not token2_node_id:
                    continue

                # Create bigram node
                bigram_node_id = builder._create_node(
                    "AND",
                    "L2",
                    f"{token1}_{token2}",
                    components=[token1_node_id, token2_node_id],
                    frequency=freq
                )

                # Connect tokens to bigram
                builder._create_edge(token1_node_id, bigram_node_id, "composition")
                builder._create_edge(token2_node_id, bigram_node_id, "composition")
            except Exception as e:
                print(f"Error processing bigram {bigram}: {e}")
    except Exception as e:
        print(f"Error building bigrams: {e}")

    # We'll skip higher levels for now since they're complex
    # and might have similar issues

    return True


# Run the careful test
if process_file_safely("test_input.txt"):
    print("\nGraph built successfully.")

    # Print statistics
    print(f"\nStatistics:")
    print(f"Total nodes: {len(builder.graph['nodes'])}")
    print(f"AND nodes: {builder.graph['metadata']['and_count']}")
    print(f"OR nodes: {builder.graph['metadata']['or_count']}")
    print(f"L1 nodes: {builder.graph['metadata']['levels']['L1']}")
    print(f"L2 nodes: {builder.graph['metadata']['levels']['L2']}")
    print(f"L3 nodes: {builder.graph['metadata']['levels']['L3']}")
    print(f"L4 nodes: {builder.graph['metadata']['levels']['L4']}")
    print(f"L5 nodes: {builder.graph['metadata']['levels']['L5']}")
    print(f"Total edges: {len(builder.graph['edges'])}")

    # Save graph
    if builder.save_to_json("htpc_model_careful.json"):
        print("\nGraph saved to htpc_model_careful.json")

    # Try visualization with limited nodes
    try:
        builder.visualize("htpc_graph_careful.png", max_nodes=30, show=False)
        print("Visualization saved to htpc_graph_careful.png")
    except Exception as e:
        print(f"Error creating visualization: {e}")
        traceback.print_exc()
else:
    print("Failed to build graph.")