"""
Modified test script that handles the problematic _build_hierarchies method.
"""
from htpc_graph_builder import HTCPGraphBuilder

# Create builder instance with debug mode enabled
builder = HTCPGraphBuilder(debug=True)

# Configure thresholds
builder.thresholds = {
    "bigram": 2,     # Minimum frequency to create a bigram
    "phrase": 2,     # Minimum frequency to create a phrase
    "hierarchy": 1,  # Minimum frequency to create a hierarchy
    "discourse": 1   # Minimum frequency to create a discourse pattern
}

# Override the _build_hierarchies method with a fixed version
def safe_build_hierarchies(self):
    """Safe version of the _build_hierarchies method that avoids unpacking errors"""
    print("Using safe version of _build_hierarchies")

    # Create a map of phrases by sequence for co-occurrence analysis
    sequence_phrases = {}

    # Find phrases in each sequence
    for seq_idx, tokens in enumerate(self.sequence_buffer):
        sequence_phrases[seq_idx] = []
        seq_str = " ".join(tokens)

        # Check each phrase node to see if it appears in this sequence
        for node_id, node in self.graph["nodes"].items():
            if node["level"] == "L3" and node["type"] == "AND":
                phrase = node["value"].replace("_", " ")
                if phrase in seq_str:
                    sequence_phrases[seq_idx].append(node_id)

    # Find co-occurring phrases
    hierarchy_pairs = {}

    for seq_idx, phrase_nodes in sequence_phrases.items():
        # Create pairs of co-occurring phrases
        for i in range(len(phrase_nodes)):
            for j in range(i+1, len(phrase_nodes)):
                # Create a unique key for this pair using a pipe character
                pair_key = f"{phrase_nodes[i]}|{phrase_nodes[j]}"
                hierarchy_pairs[pair_key] = hierarchy_pairs.get(pair_key, 0) + 1

    # Create hierarchy nodes for frequent co-occurrences
    for pair_key, freq in hierarchy_pairs.items():
        if freq < self.thresholds["hierarchy"]:
            continue

        try:
            # Split safely using pipe character
            parts = pair_key.split("|")
            if len(parts) != 2:
                print(f"Invalid hierarchy pair key: {pair_key}")
                continue

            phrase1_id = parts[0]
            phrase2_id = parts[1]

            # Verify both phrase nodes exist
            if phrase1_id not in self.graph["nodes"] or phrase2_id not in self.graph["nodes"]:
                print(f"Cannot find one or both phrases: {phrase1_id}, {phrase2_id}")
                continue

            # Create a unique hierarchy node ID
            hierarchy_node_id = f"h_{self.next_node_id}"
            self.next_node_id += 1

            # Get phrase values
            phrase1_value = self.graph["nodes"][phrase1_id]["value"]
            phrase2_value = self.graph["nodes"][phrase2_id]["value"]

            # Create the hierarchy node
            self.graph["nodes"][hierarchy_node_id] = {
                "id": hierarchy_node_id,
                "level": "L4",
                "type": "AND",
                "value": f"H({phrase1_value},{phrase2_value})",
                "components": [phrase1_id, phrase2_id],
                "frequency": freq
            }

            # Update metadata
            self.graph["metadata"]["and_count"] += 1
            self.graph["metadata"]["levels"]["L4"] += 1

            # Add to NetworkX graph
            self.nx_graph.add_node(
                hierarchy_node_id,
                label=f"H({phrase1_value},{phrase2_value})",
                type="AND",
                level="L4",
                color=self._get_node_color("AND", "L4"),
                shape="box"
            )

            # Connect phrase nodes to hierarchy node
            for comp_id in [phrase1_id, phrase2_id]:
                edge = {
                    "source": comp_id,
                    "target": hierarchy_node_id,
                    "type": "composition",
                    "weight": 1
                }
                self.graph["edges"].append(edge)
                self.nx_graph.add_edge(
                    comp_id,
                    hierarchy_node_id,
                    type="composition",
                    weight=1
                )

            print(f"Created hierarchy node: {hierarchy_node_id}")
        except Exception as e:
            print(f"Error creating hierarchy for {pair_key}: {e}")
            import traceback
            traceback.print_exc()


# Override the _build_discourse_patterns method with a version that uses pipe separators
def safe_build_discourse_patterns(self):
    """Safe version of the _build_discourse_patterns method"""
    print("Using safe version of _build_discourse_patterns")

    # Track which hierarchies appear in which sequences
    hierarchy_by_seq = {}

    # Find hierarchies in each sequence
    for seq_idx, tokens in enumerate(self.sequence_buffer):
        hierarchy_by_seq[seq_idx] = []
        seq_text = " ".join(tokens)

        # Check each hierarchy node
        for node_id, node in self.graph["nodes"].items():
            if node["level"] == "L4" and node["type"] == "AND":
                if "components" not in node or len(node["components"]) < 2:
                    continue

                # Get the phrases in this hierarchy
                components = node["components"]
                phrase1_id = components[0]
                phrase2_id = components[1]

                if phrase1_id not in self.graph["nodes"] or phrase2_id not in self.graph["nodes"]:
                    continue

                # Get phrase values
                phrase1 = self.graph["nodes"][phrase1_id]["value"].replace("_", " ")
                phrase2 = self.graph["nodes"][phrase2_id]["value"].replace("_", " ")

                # Check if both phrases appear in the sequence
                if phrase1 in seq_text and phrase2 in seq_text:
                    hierarchy_by_seq[seq_idx].append(node_id)

    # Find discourse patterns (hierarchies that appear in consecutive sequences)
    discourse_patterns = {}

    for i in range(len(self.sequence_buffer) - 1):
        for h1 in hierarchy_by_seq.get(i, []):
            for h2 in hierarchy_by_seq.get(i+1, []):
                # Use pipe separator
                pattern_key = f"{h1}|{h2}"
                discourse_patterns[pattern_key] = discourse_patterns.get(pattern_key, 0) + 1

    # Create discourse nodes for frequent patterns
    for pattern_key, freq in discourse_patterns.items():
        if freq < self.thresholds["discourse"]:
            continue

        try:
            # Split safely
            parts = pattern_key.split("|")
            if len(parts) != 2:
                print(f"Invalid discourse pattern key: {pattern_key}")
                continue

            h1_id, h2_id = parts[0], parts[1]

            # Verify both hierarchy nodes exist
            if h1_id not in self.graph["nodes"] or h2_id not in self.graph["nodes"]:
                print(f"Cannot find one or both hierarchies: {h1_id}, {h2_id}")
                continue

            # Create a unique discourse node ID
            discourse_node_id = f"d_{self.next_node_id}"
            self.next_node_id += 1

            # Get hierarchy values
            h1_value = self.graph["nodes"][h1_id]["value"]
            h2_value = self.graph["nodes"][h2_id]["value"]

            # Create the discourse node directly
            self.graph["nodes"][discourse_node_id] = {
                "id": discourse_node_id,
                "level": "L5",
                "type": "AND",
                "value": f"D({h1_value}→{h2_value})",
                "components": [h1_id, h2_id],
                "frequency": freq
            }

            # Update metadata
            self.graph["metadata"]["and_count"] += 1
            self.graph["metadata"]["levels"]["L5"] += 1

            # Add to NetworkX graph
            self.nx_graph.add_node(
                discourse_node_id,
                label=f"D({h1_value}→{h2_value})",
                type="AND",
                level="L5",
                color=self._get_node_color("AND", "L5"),
                shape="box"
            )

            # Connect hierarchy nodes to discourse node
            for comp_id in [h1_id, h2_id]:
                edge = {
                    "source": comp_id,
                    "target": discourse_node_id,
                    "type": "composition",
                    "weight": 1
                }
                self.graph["edges"].append(edge)
                self.nx_graph.add_edge(
                    comp_id,
                    discourse_node_id,
                    type="composition",
                    weight=1
                )

            print(f"Created discourse node: {discourse_node_id}")
        except Exception as e:
            print(f"Error creating discourse for {pattern_key}: {e}")
            import traceback
            traceback.print_exc()


# Override the _build_higher_levels method to use our safer implementations
def safe_build_higher_levels(self):
    """Safe version of the _build_higher_levels method"""
    print("\nBuilding higher levels safely:")
    try:
        # L2: Process bigrams
        print("Building bigrams...")
        self._build_bigrams()

        # Create OR nodes for bigrams
        print("Creating OR nodes for bigrams...")
        self._create_bigram_or_nodes()

        # L3: Process phrases
        print("Building phrases...")
        self._build_phrases()

        # Create OR nodes for phrases
        print("Creating OR nodes for phrases...")
        self._create_phrase_or_nodes()

        # L4: Process hierarchies with our safe method
        print("Building hierarchies...")
        safe_build_hierarchies(self)

        # Create OR nodes for hierarchies
        print("Creating OR nodes for hierarchies...")
        self._create_hierarchy_or_nodes()

        # L5: Process discourse patterns with our safe method
        print("Building discourse patterns...")
        safe_build_discourse_patterns(self)

        # Create OR nodes for discourse patterns
        print("Creating OR nodes for discourse patterns...")
        self._create_discourse_or_nodes()
    except Exception as e:
        print(f"Error in safe_build_higher_levels: {e}")
        import traceback
        traceback.print_exc()


# Replace the problematic methods
builder._build_hierarchies = safe_build_hierarchies.__get__(builder)
builder._build_discourse_patterns = safe_build_discourse_patterns.__get__(builder)
builder._build_higher_levels = safe_build_higher_levels.__get__(builder)

# Build the graph from the example file
print("Building graph from test_input.txt...")
if builder.build_from_file("test_input.txt"):
    print("Graph built successfully.")

    # Print some statistics
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

    # Save the graph to a JSON file
    if builder.save_to_json("htpc_model_fixed.json"):
        print("\nGraph saved to htpc_model_fixed.json")
    else:
        print("\nFailed to save graph.")

    # Generate visualization
    print("\nGenerating visualization...")
    try:
        if builder.visualize("htpc_graph_fixed.png", max_nodes=50, show=False):
            print("Visualization saved to htpc_graph_fixed.png")
        else:
            print("Failed to create visualization.")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Failed to build graph.")