"""
Debug script to find the exact location of the unpacking error.
This script tests processing a single problematic sentence.
"""

from htpc_graph_builder import HTCPGraphBuilder

# Create test sentence that causes the error
test_sentence = "The man and woman walked home together."

# Create builder instance with debug mode
builder = HTCPGraphBuilder(debug=True)

# Override methods with diagnostic versions
original_process_sequence = builder.process_sequence


def debug_process_sequence(self, sequence):
    print("\n===== DEBUG: Starting process_sequence =====")
    print(f"Sequence: {sequence}")
    try:
        tokens = self.tokenize(sequence)
        print(f"Tokens: {tokens}")

        if not tokens:
            print("No tokens found, returning")
            return

        # Store the sequence for higher-level processing
        self.sequence_buffer.append(tokens)
        print(f"Added to sequence buffer, length now: {len(self.sequence_buffer)}")

        # Process L1: Token transitions with detailed debugging
        print("\n----- Calling _process_tokens -----")
        self._debug_process_tokens(tokens)

        # Process higher levels if we have enough sequences
        if len(self.sequence_buffer) >= 5:
            print("\n----- Calling _build_higher_levels -----")
            # Keep only the most recent sequences for sliding window
            self.sequence_buffer = self.sequence_buffer[-5:]
    except Exception as e:
        print(f"ERROR in process_sequence: {e}")
        import traceback
        traceback.print_exc()


def debug_process_tokens(self, tokens):
    print("\n>> DEBUG _process_tokens")
    print(f"Tokens: {tokens}")

    try:
        # Track token nodes in this sequence
        sequence_nodes = []

        # First, ensure all tokens have nodes
        token_nodes = {}
        for token in tokens:
            print(f"\nProcessing token: '{token}'")
            # Update token frequency
            self.token_frequencies[token] += 1

            # Check if token already has a node
            token_node_id = None
            for node_id, node in self.graph["nodes"].items():
                if node["level"] == "L1" and node["type"] == "AND" and node["value"] == token:
                    token_node_id = node_id
                    node["frequency"] += 1
                    print(f"  Found existing token node: {node_id}")
                    break

            # Create token node if needed (token nodes are AND nodes)
            if not token_node_id:
                token_node_id = self._create_node("AND", "L1", token)
                print(f"  Created new token node: {token_node_id}")

            token_nodes[token] = token_node_id
            sequence_nodes.append(token_node_id)

        print("\nProcessing transitions between tokens:")
        # Process transitions between tokens
        for i in range(len(tokens) - 1):
            current_token = tokens[i]
            next_token = tokens[i + 1]
            print(f"  Transition: '{current_token}' -> '{next_token}'")

            current_node_id = token_nodes[current_token]
            next_node_id = token_nodes[next_token]

            # Update tracking for OR node creation
            self.successors[current_node_id].add(next_node_id)
            self.predecessors[next_node_id].add(current_node_id)

            # Create direct edge (will be replaced by OR nodes later)
            self._create_edge(current_node_id, next_node_id, "sequence")
            print(f"  Added edge: {current_node_id} -> {next_node_id}")

            # Track bigram frequency
            bigram_key = f"{current_token}_{next_token}"
            self.bigram_frequencies[bigram_key] += 1
            print(f"  Incremented bigram frequency for '{bigram_key}'")

        print("\nAbout to call _create_or_nodes")
        # Create OR nodes for convergence and divergence points
        self._debug_create_or_nodes()

    except Exception as e:
        print(f"ERROR in _process_tokens: {e}")
        import traceback
        traceback.print_exc()


def debug_create_or_nodes(self):
    print("\n>> DEBUG _create_or_nodes")
    try:
        # Handle convergence points (multiple paths leading to the same node)
        print("\nChecking convergence points (nodes with multiple predecessors):")
        for node_id, pred_set in self.predecessors.items():
            print(f"  Node {node_id} has {len(pred_set)} predecessors: {pred_set}")
            if len(pred_set) > 1:  # Multiple predecessors
                print(f"  Creating OR node for convergence at {node_id}")
                try:
                    # Create an OR node for this convergence point
                    or_node_id = self._create_node("OR", "L1", f"OR_in_{node_id}", components=list(pred_set))
                    print(f"  Created OR node: {or_node_id}")

                    # Connect predecessors to OR node instead of directly to the target
                    for pred_id in pred_set:
                        print(f"  Processing predecessor: {pred_id}")
                        # Remove direct edge
                        edges_to_remove = []
                        for i, edge in enumerate(self.graph["edges"]):
                            if edge["source"] == pred_id and edge["target"] == node_id:
                                edges_to_remove.append(i)
                                # Add edge from predecessor to OR node
                                self._create_edge(pred_id, or_node_id, "alternative")
                                print(f"  Added edge: {pred_id} -> {or_node_id} (alternative)")

                        # Remove edges in reverse order to avoid index issues
                        for i in sorted(edges_to_remove, reverse=True):
                            if i < len(self.graph["edges"]):  # Safety check
                                print(
                                    f"  Removing edge at index {i}: {self.graph['edges'][i]['source']} -> {self.graph['edges'][i]['target']}")
                                del self.graph["edges"][i]

                        # Remove from NetworkX graph if it exists
                        if self.nx_graph.has_edge(pred_id, node_id):
                            print(f"  Removing edge from NetworkX graph: {pred_id} -> {node_id}")
                            self.nx_graph.remove_edge(pred_id, node_id)

                    # Add edge from OR node to target
                    self._create_edge(or_node_id, node_id, "sequence")
                    print(f"  Added edge: {or_node_id} -> {node_id} (sequence)")
                except Exception as e:
                    print(f"ERROR creating convergence OR node for {node_id}: {e}")
                    import traceback
                    traceback.print_exc()

        # Handle divergence points (node with multiple possible next nodes)
        print("\nChecking divergence points (nodes with multiple successors):")
        for node_id, succ_set in self.successors.items():
            print(f"  Node {node_id} has {len(succ_set)} successors: {succ_set}")
            if len(succ_set) > 1:  # Multiple successors
                print(f"  Creating OR node for divergence at {node_id}")
                try:
                    # Create an OR node for this divergence point
                    or_node_id = self._create_node("OR", "L1", f"OR_out_{node_id}", components=list(succ_set))
                    print(f"  Created OR node: {or_node_id}")

                    # Connect OR node to successors instead of directly from the source
                    for succ_id in succ_set:
                        print(f"  Processing successor: {succ_id}")
                        # Remove direct edge
                        edges_to_remove = []
                        for i, edge in enumerate(self.graph["edges"]):
                            if edge["source"] == node_id and edge["target"] == succ_id:
                                edges_to_remove.append(i)
                                # Add edge from OR node to successor
                                self._create_edge(or_node_id, succ_id, "alternative")
                                print(f"  Added edge: {or_node_id} -> {succ_id} (alternative)")

                        # Remove edges in reverse order to avoid index issues
                        for i in sorted(edges_to_remove, reverse=True):
                            if i < len(self.graph["edges"]):  # Safety check
                                print(
                                    f"  Removing edge at index {i}: {self.graph['edges'][i]['source']} -> {self.graph['edges'][i]['target']}")
                                del self.graph["edges"][i]

                        # Remove from NetworkX graph if it exists
                        if self.nx_graph.has_edge(node_id, succ_id):
                            print(f"  Removing edge from NetworkX graph: {node_id} -> {succ_id}")
                            self.nx_graph.remove_edge(node_id, succ_id)

                    # Add edge from source to OR node
                    self._create_edge(node_id, or_node_id, "sequence")
                    print(f"  Added edge: {node_id} -> {or_node_id} (sequence)")
                except Exception as e:
                    print(f"ERROR creating divergence OR node for {node_id}: {e}")
                    import traceback
                    traceback.print_exc()
    except Exception as e:
        print(f"ERROR in _create_or_nodes: {e}")
        import traceback
        traceback.print_exc()


# Override methods with debug versions
builder.process_sequence = debug_process_sequence.__get__(builder)
builder._debug_process_tokens = debug_process_tokens.__get__(builder)
builder._debug_create_or_nodes = debug_create_or_nodes.__get__(builder)

# Try processing the problematic sentence
print("\n\n========== STARTING DEBUG TEST ==========")
builder.process_sequence(test_sentence)
print("========== DEBUG TEST COMPLETE ==========")