import json
import re
from collections import defaultdict, Counter
import traceback

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


class HTCPGraphBuilder:
    """
    Builds a hierarchical AND-OR graph model for HTPC with 5 levels:
    - L1: Token transitions with OR nodes for alternative paths
    - L2: Bigram memory (AND nodes with OR alternatives)
    - L3: Phrase memory (AND nodes with OR alternatives)
    - L4: Phrase hierarchy (AND nodes with OR alternatives)
    - L5: Discourse patterns (AND nodes with OR alternatives)
    """

    def __init__(self, debug=False):
        # Debug mode flag
        self.debug = debug

        # Initialize the graph structure
        self.graph = {
            "nodes": {},  # All nodes with their attributes
            "edges": [],  # All edges with their attributes
            "metadata": {
                "token_count": 0,
                "or_count": 0,
                "and_count": 0,
                "levels": {
                    "L1": 0,  # Token level
                    "L2": 0,  # Bigram level
                    "L3": 0,  # Phrase level
                    "L4": 0,  # Hierarchy level
                    "L5": 0  # Discourse level
                }
            }
        }
        # Thresholds for each level
        self.thresholds = {
            "bigram": 2,  # Minimum frequency to create a bigram
            "phrase": 2,  # Minimum frequency to create a phrase
            "hierarchy": 2,  # Minimum frequency to create a hierarchy
            "discourse": 2  # Minimum frequency to create a discourse pattern
        }
        # Tracking counters
        self.token_frequencies = Counter()
        self.bigram_frequencies = Counter()
        self.phrase_frequencies = Counter()
        self.hierarchy_frequencies = Counter()
        # Temporary storage for sequence processing
        self.sequence_buffer = []
        # Tracking node predecessors and successors for OR node creation
        self.predecessors = defaultdict(set)  # nodes that come before a given node
        self.successors = defaultdict(set)  # nodes that come after a given node
        # Node counter for unique IDs
        self.next_node_id = 0
        # NetworkX graph for visualization
        self.nx_graph = nx.DiGraph()

    def _debug(self, message):
        """Print debug message if debug mode is enabled"""
        if self.debug:
            print(f"DEBUG: {message}")

    def _create_node(self, node_type, level, value, components=None, frequency=1):
        """Create a new node in the graph with a unique ID"""
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1

        node = {
            "id": node_id,
            "type": node_type,  # "AND" or "OR"
            "level": level,  # L1, L2, L3, L4, L5
            "value": value,  # The actual content (token, phrase, etc.)
            "frequency": frequency
        }

        if components:
            node["components"] = components

        self.graph["nodes"][node_id] = node

        # Update metadata
        if node_type == "AND":
            self.graph["metadata"]["and_count"] += 1
        else:  # "OR"
            self.graph["metadata"]["or_count"] += 1

        self.graph["metadata"]["levels"][level] += 1

        # Add to NetworkX graph for visualization
        color = self._get_node_color(node_type, level)
        shape = "box" if node_type == "AND" else "ellipse"
        self.nx_graph.add_node(node_id,
                               label=str(value),
                               type=node_type,
                               level=level,
                               color=color,
                               shape=shape)

        return node_id

    def _create_edge(self, source_id, target_id, edge_type, weight=1):
        """Create a new edge in the graph"""
        edge = {
            "source": source_id,
            "target": target_id,
            "type": edge_type,
            "weight": weight
        }

        # Check if edge already exists
        for existing_edge in self.graph["edges"]:
            if (existing_edge["source"] == source_id and
                    existing_edge["target"] == target_id and
                    existing_edge["type"] == edge_type):
                # Update weight of existing edge
                existing_edge["weight"] += weight
                # Update NetworkX edge
                if source_id in self.nx_graph and target_id in self.nx_graph[source_id]:
                    self.nx_graph[source_id][target_id]["weight"] = existing_edge["weight"]
                return

        # Add new edge
        self.graph["edges"].append(edge)

        # Add to NetworkX graph
        self.nx_graph.add_edge(source_id, target_id,
                               type=edge_type,
                               weight=weight)

    def _get_node_color(self, node_type, level):
        """Get node color based on type and level for visualization"""
        # Base colors
        and_color = [0.2, 0.6, 0.9, 1.0]  # Blue
        or_color = [0.9, 0.4, 0.2, 1.0]  # Orange

        # Adjust color intensity based on level
        level_num = int(level[1])  # Extract number from L1, L2, etc.
        intensity = 0.5 + (level_num * 0.1)  # Darker for higher levels

        if node_type == "AND":
            color = [c * intensity for c in and_color[:-1]] + [and_color[-1]]
        else:  # "OR"
            color = [c * intensity for c in or_color[:-1]] + [or_color[-1]]

        return f"rgba({color[0] * 255:.0f}, {color[1] * 255:.0f}, {color[2] * 255:.0f}, {color[3]:.1f})"

    def tokenize(self, text):
        """Convert text to tokens, handling basic punctuation"""
        try:
            # Remove excess whitespace and convert to lowercase
            text = text.strip().lower()

            # Simple tokenization: split on whitespace and keep punctuation
            tokens = re.findall(r'\b\w+\b|[.,!?;]', text)

            if self.debug:
                print(f"Tokenized '{text}' into: {tokens}")

            return tokens
        except Exception as e:
            if self.debug:
                print(f"Error tokenizing text: {e}")
                traceback.print_exc()
            return []

    def process_sequence(self, sequence):
        """Process a single sequence (sentence) and update the graph"""
        try:
            tokens = self.tokenize(sequence)
            if not tokens:
                return

            # Store the sequence for higher-level processing
            self.sequence_buffer.append(tokens)

            # Process L1: Token transitions
            self._process_tokens(tokens)

            # Process higher levels if we have enough sequences
            if len(self.sequence_buffer) >= 5:  # Wait until we have enough context
                self._build_higher_levels()
                # Keep only the most recent sequences for sliding window
                self.sequence_buffer = self.sequence_buffer[-5:]
        except Exception as e:
            if self.debug:
                print(f"Error processing sequence: {sequence}")
                print(f"Error details: {e}")
                traceback.print_exc()

    def _process_tokens(self, tokens):
        """Process token-level (L1) structures including OR nodes for alternatives"""
        try:
            # Track token nodes in this sequence
            sequence_nodes = []

            # First, ensure all tokens have nodes
            token_nodes = {}
            for token in tokens:
                try:
                    # Update token frequency
                    self.token_frequencies[token] += 1

                    # Check if token already has a node
                    token_node_id = None
                    for node_id, node in self.graph["nodes"].items():
                        if node["level"] == "L1" and node["type"] == "AND" and node["value"] == token:
                            token_node_id = node_id
                            node["frequency"] += 1
                            break

                    # Create token node if needed (token nodes are AND nodes)
                    if not token_node_id:
                        token_node_id = self._create_node("AND", "L1", token)

                    token_nodes[token] = token_node_id
                    sequence_nodes.append(token_node_id)
                except Exception as e:
                    if self.debug:
                        print(f"Error processing token '{token}': {e}")
                        continue  # Skip this token but continue with others

            # Process transitions between tokens
            for i in range(len(tokens) - 1):
                try:
                    current_token = tokens[i]
                    next_token = tokens[i + 1]

                    if current_token not in token_nodes or next_token not in token_nodes:
                        continue  # Skip if either token was not processed

                    current_node_id = token_nodes[current_token]
                    next_node_id = token_nodes[next_token]

                    # Update tracking for OR node creation
                    self.successors[current_node_id].add(next_node_id)
                    self.predecessors[next_node_id].add(current_node_id)

                    # Create direct edge (will be replaced by OR nodes later)
                    self._create_edge(current_node_id, next_node_id, "sequence")

                    # Track bigram frequency
                    bigram_key = f"{current_token}_{next_token}"
                    self.bigram_frequencies[bigram_key] += 1
                except Exception as e:
                    if self.debug:
                        print(f"Error processing transition {tokens[i]} -> {tokens[i + 1]}: {e}")
                        continue  # Skip this transition but continue with others

            # Create OR nodes for convergence and divergence points
            self._create_or_nodes()
        except Exception as e:
            if self.debug:
                print(f"Error in _process_tokens: {e}")
                traceback.print_exc()

    def _create_or_nodes(self):
        """Create OR nodes for points with multiple predecessors or successors"""
        try:
            # Handle convergence points (multiple paths leading to the same node)
            for node_id, pred_set in self.predecessors.items():
                if len(pred_set) > 1:  # Multiple predecessors
                    try:
                        # Create an OR node for this convergence point
                        or_node_id = self._create_node("OR", "L1", f"OR_in_{node_id}", components=list(pred_set))

                        # Connect predecessors to OR node instead of directly to the target
                        for pred_id in pred_set:
                            # Remove direct edge
                            edges_to_remove = []
                            for i, edge in enumerate(self.graph["edges"]):
                                if edge["source"] == pred_id and edge["target"] == node_id:
                                    edges_to_remove.append(i)
                                    # Add edge from predecessor to OR node
                                    self._create_edge(pred_id, or_node_id, "alternative")

                            # Remove edges in reverse order to avoid index issues
                            for i in sorted(edges_to_remove, reverse=True):
                                if i < len(self.graph["edges"]):  # Safety check
                                    del self.graph["edges"][i]

                            # Remove from NetworkX graph if it exists
                            if pred_id in self.nx_graph and node_id in self.nx_graph[pred_id]:
                                self.nx_graph.remove_edge(pred_id, node_id)

                        # Add edge from OR node to target
                        self._create_edge(or_node_id, node_id, "sequence")
                    except Exception as e:
                        if self.debug:
                            print(f"Error creating convergence OR node for {node_id}: {e}")

            # Handle divergence points (node with multiple possible next nodes)
            for node_id, succ_set in self.successors.items():
                if len(succ_set) > 1:  # Multiple successors
                    try:
                        # Create an OR node for this divergence point
                        or_node_id = self._create_node("OR", "L1", f"OR_out_{node_id}", components=list(succ_set))

                        # Connect OR node to successors instead of directly from the source
                        for succ_id in succ_set:
                            # Remove direct edge
                            edges_to_remove = []
                            for i, edge in enumerate(self.graph["edges"]):
                                if edge["source"] == node_id and edge["target"] == succ_id:
                                    edges_to_remove.append(i)
                                    # Add edge from OR node to successor
                                    self._create_edge(or_node_id, succ_id, "alternative")

                            # Remove edges in reverse order to avoid index issues
                            for i in sorted(edges_to_remove, reverse=True):
                                if i < len(self.graph["edges"]):  # Safety check
                                    del self.graph["edges"][i]

                            # Remove from NetworkX graph if it exists
                            if node_id in self.nx_graph and succ_id in self.nx_graph[node_id]:
                                self.nx_graph.remove_edge(node_id, succ_id)

                        # Add edge from source to OR node
                        self._create_edge(node_id, or_node_id, "sequence")
                    except Exception as e:
                        if self.debug:
                            print(f"Error creating divergence OR node for {node_id}: {e}")
        except Exception as e:
            if self.debug:
                print(f"Error in _create_or_nodes: {e}")
                traceback.print_exc()

    def _build_higher_levels(self):
        """Build higher-level structures (L2-L5) based on collected data"""
        try:
            # Only proceed if there are enough sequences to work with
            if len(self.sequence_buffer) < 2:
                if self.debug:
                    print("Not enough sequences in buffer to build higher levels")
                return

            # Process each level sequentially
            self._build_bigrams()

            # Create OR nodes for bigrams (L2)
            self._create_bigram_or_nodes()

            self._build_phrases()

            # Create OR nodes for phrases (L3)
            self._create_phrase_or_nodes()

            self._build_hierarchies()

            # Create OR nodes for hierarchies (L4)
            self._create_hierarchy_or_nodes()

            self._build_discourse_patterns()

            # Create OR nodes for discourse patterns (L5)
            self._create_discourse_or_nodes()

        except Exception as e:
            if self.debug:
                print(f"Error in _build_higher_levels: {e}")
                traceback.print_exc()

    def _build_bigrams(self):
        """Build L2: Bigram nodes (AND nodes) from token transitions"""
        try:
            for bigram, freq in self.bigram_frequencies.items():
                if freq < self.thresholds["bigram"]:
                    continue

                # Parse the bigram key back into tokens
                # Handle case when bigram key might have multiple underscores
                parts = bigram.split("_")
                if len(parts) < 2:
                    continue  # Skip if invalid format
                token1 = parts[0]
                token2 = parts[1]  # This takes the second token, ignoring any additional splits

                # Find token node IDs
                token1_node_id = None
                token2_node_id = None

                for node_id, node in self.graph["nodes"].items():
                    if node["level"] == "L1" and node["type"] == "AND":
                        if node["value"] == token1:
                            token1_node_id = node_id
                        elif node["value"] == token2:
                            token2_node_id = node_id

                if not token1_node_id or not token2_node_id:
                    continue  # Skip if token nodes not found

                # Check if bigram already exists
                bigram_node_id = None
                for node_id, node in self.graph["nodes"].items():
                    if (node["level"] == "L2" and
                            node["type"] == "AND" and
                            "components" in node and
                            set(node["components"]) == {token1_node_id, token2_node_id}):
                        bigram_node_id = node_id
                        node["frequency"] += freq
                        break

                # Create new bigram node if needed
                if not bigram_node_id:
                    try:
                        bigram_node_id = self._create_node(
                            "AND",
                            "L2",
                            f"{token1}_{token2}",
                            components=[token1_node_id, token2_node_id],
                            frequency=freq
                        )

                        # Connect token nodes to bigram node
                        self._create_edge(token1_node_id, bigram_node_id, "composition")
                        self._create_edge(token2_node_id, bigram_node_id, "composition")
                    except Exception as e:
                        if self.debug:
                            print(f"Error creating bigram node for {token1}_{token2}: {e}")
                            traceback.print_exc()
        except Exception as e:
            if self.debug:
                print(f"Error in _build_bigrams: {e}")
                traceback.print_exc()

    def _build_phrases(self):
        """Build L3: Phrase nodes (AND nodes) from sequences of tokens"""
        try:
            # Process each sequence in the buffer to find phrases
            for tokens in self.sequence_buffer:
                # Find all potential phrases (3+ tokens)
                if len(tokens) < 3:
                    continue

                # Generate all possible phrases from the sequence
                for i in range(len(tokens) - 2):
                    for j in range(i + 2, min(i + 6, len(tokens))):  # Limit phrase length
                        try:
                            phrase_tokens = tokens[i:j + 1]
                            phrase_key = "_".join(phrase_tokens)

                            # Update phrase frequency
                            self.phrase_frequencies[phrase_key] += 1

                            # Check if this meets our threshold
                            if self.phrase_frequencies[phrase_key] >= self.thresholds["phrase"]:
                                # Find token node IDs in this phrase
                                token_node_ids = []
                                all_tokens_found = True

                                for token in phrase_tokens:
                                    token_found = False
                                    for node_id, node in self.graph["nodes"].items():
                                        if (node["level"] == "L1" and
                                                node["type"] == "AND" and
                                                node["value"] == token):
                                            token_node_ids.append(node_id)
                                            token_found = True
                                            break

                                    if not token_found:
                                        all_tokens_found = False
                                        break

                                if not all_tokens_found or len(token_node_ids) != len(phrase_tokens):
                                    continue  # Skip if not all tokens found

                                # Check if phrase already exists
                                phrase_node_id = None
                                for node_id, node in self.graph["nodes"].items():
                                    if (node["level"] == "L3" and
                                            node["type"] == "AND" and
                                            "components" in node and
                                            len(node["components"]) == len(token_node_ids) and
                                            all(c1 == c2 for c1, c2 in zip(node["components"], token_node_ids))):
                                        phrase_node_id = node_id
                                        node["frequency"] += 1
                                        break

                                # Create new phrase node if needed
                                if not phrase_node_id:
                                    phrase_node_id = self._create_node(
                                        "AND",
                                        "L3",
                                        phrase_key,
                                        components=token_node_ids,
                                        frequency=self.phrase_frequencies[phrase_key]
                                    )

                                    # Connect token nodes to phrase node
                                    for token_node_id in token_node_ids:
                                        self._create_edge(token_node_id, phrase_node_id, "composition")
                        except Exception as e:
                            if self.debug:
                                print(f"Error processing phrase {phrase_tokens}: {e}")
                                continue  # Skip this phrase but continue with others
        except Exception as e:
            if self.debug:
                print(f"Error in _build_phrases: {e}")
                traceback.print_exc()

    def _build_hierarchies(self):
        """Build L4: Hierarchy nodes (AND nodes) connecting phrases"""
        try:
            # Create a map of phrases by sequence for co-occurrence analysis
            sequence_phrases = defaultdict(list)

            # Find phrases in each sequence
            for seq_idx, tokens in enumerate(self.sequence_buffer):
                seq_str = " ".join(tokens)

                # Check each phrase node to see if it appears in this sequence
                for node_id, node in self.graph["nodes"].items():
                    if node["level"] == "L3" and node["type"] == "AND":
                        phrase = node["value"].replace("_", " ")
                        if phrase in seq_str:
                            sequence_phrases[seq_idx].append(node_id)

            # Find co-occurring phrases
            hierarchy_pairs = Counter()

            for seq_idx, phrase_nodes in sequence_phrases.items():
                # Create pairs of co-occurring phrases
                for i in range(len(phrase_nodes)):
                    for j in range(i + 1, len(phrase_nodes)):
                        # Create a unique key for this pair that doesn't use underscore
                        # since phrase IDs might already contain underscores
                        pair_key = f"{phrase_nodes[i]}|{phrase_nodes[j]}"
                        hierarchy_pairs[pair_key] += 1

            # Create hierarchy nodes for frequent co-occurrences
            for pair_key, freq in hierarchy_pairs.items():
                if freq < self.thresholds["hierarchy"]:
                    continue

                # Split the key safely
                parts = pair_key.split("|")
                if len(parts) != 2:
                    if self.debug:
                        print(f"Invalid hierarchy pair key: {pair_key}")
                    continue

                phrase1_id, phrase2_id = parts[0], parts[1]

                # Verify both phrase nodes exist
                if phrase1_id not in self.graph["nodes"] or phrase2_id not in self.graph["nodes"]:
                    if self.debug:
                        print(f"Cannot find one or both phrases: {phrase1_id}, {phrase2_id}")
                    continue

                # Check if hierarchy already exists
                hierarchy_node_id = None
                for node_id, node in self.graph["nodes"].items():
                    if (node["level"] == "L4" and
                            node["type"] == "AND" and
                            "components" in node and
                            len(node["components"]) == 2 and
                            ((node["components"][0] == phrase1_id and node["components"][1] == phrase2_id) or
                             (node["components"][0] == phrase2_id and node["components"][1] == phrase1_id))):
                        hierarchy_node_id = node_id
                        node["frequency"] += freq
                        break

                # Create new hierarchy node if needed
                if not hierarchy_node_id:
                    try:
                        phrase1_value = self.graph["nodes"][phrase1_id]["value"]
                        phrase2_value = self.graph["nodes"][phrase2_id]["value"]

                        hierarchy_node_id = self._create_node(
                            "AND",
                            "L4",
                            f"H({phrase1_value},{phrase2_value})",
                            components=[phrase1_id, phrase2_id],
                            frequency=freq
                        )

                        # Connect phrase nodes to hierarchy node
                        self._create_edge(phrase1_id, hierarchy_node_id, "composition")
                        self._create_edge(phrase2_id, hierarchy_node_id, "composition")
                    except Exception as e:
                        if self.debug:
                            print(f"Error creating hierarchy node for {phrase1_id}|{phrase2_id}: {e}")
                            traceback.print_exc()
        except Exception as e:
            if self.debug:
                print(f"Error in _build_hierarchies: {e}")
                traceback.print_exc()

    def _build_discourse_patterns(self):
        """Build L5: Discourse nodes (AND nodes) for patterns across sentences"""
        try:
            # Track which hierarchies appear in which sequences
            hierarchy_by_seq = defaultdict(list)

            # Find hierarchies in each sequence
            for seq_idx, tokens in enumerate(self.sequence_buffer):
                seq_text = " ".join(tokens)

                # Check each hierarchy node
                for node_id, node in self.graph["nodes"].items():
                    if node["level"] == "L4" and node["type"] == "AND":
                        # Get the phrases in this hierarchy
                        if "components" not in node or len(node["components"]) < 2:
                            continue  # Skip if components missing or insufficient

                        components = node["components"]
                        phrase1_id = components[0]
                        phrase2_id = components[1]

                        # Verify these phrase nodes exist
                        if phrase1_id not in self.graph["nodes"] or phrase2_id not in self.graph["nodes"]:
                            continue

                        # Get phrase values
                        phrase1 = self.graph["nodes"][phrase1_id]["value"].replace("_", " ")
                        phrase2 = self.graph["nodes"][phrase2_id]["value"].replace("_", " ")

                        # Check if both phrases appear in the sequence
                        if phrase1 in seq_text and phrase2 in seq_text:
                            hierarchy_by_seq[seq_idx].append(node_id)

            # Find discourse patterns (hierarchies that appear in consecutive sequences)
            discourse_patterns = Counter()

            for i in range(len(self.sequence_buffer) - 1):
                for h1 in hierarchy_by_seq.get(i, []):
                    for h2 in hierarchy_by_seq.get(i + 1, []):
                        # Use pipe as separator instead of underscore to avoid confusion
                        pattern_key = f"{h1}|{h2}"
                        discourse_patterns[pattern_key] += 1

            # Create discourse nodes for frequent patterns
            for pattern_key, freq in discourse_patterns.items():
                if freq < self.thresholds["discourse"]:
                    continue

                # Split safely
                parts = pattern_key.split("|")
                if len(parts) != 2:
                    if self.debug:
                        print(f"Invalid discourse pattern key: {pattern_key}")
                    continue

                h1_id, h2_id = parts[0], parts[1]

                # Verify both hierarchy nodes exist
                if h1_id not in self.graph["nodes"] or h2_id not in self.graph["nodes"]:
                    if self.debug:
                        print(f"Cannot find one or both hierarchies: {h1_id}, {h2_id}")
                    continue

                # Check if discourse pattern already exists
                discourse_node_id = None
                for node_id, node in self.graph["nodes"].items():
                    if (node["level"] == "L5" and
                            node["type"] == "AND" and
                            "components" in node and
                            len(node["components"]) == 2 and
                            node["components"][0] == h1_id and
                            node["components"][1] == h2_id):
                        discourse_node_id = node_id
                        node["frequency"] += freq
                        break

                # Create new discourse node if needed
                if not discourse_node_id:
                    try:
                        h1_value = self.graph["nodes"][h1_id]["value"]
                        h2_value = self.graph["nodes"][h2_id]["value"]

                        discourse_node_id = self._create_node(
                            "AND",
                            "L5",
                            f"D({h1_value}→{h2_value})",
                            components=[h1_id, h2_id],
                            frequency=freq
                        )

                        # Connect hierarchy nodes to discourse node
                        self._create_edge(h1_id, discourse_node_id, "composition")
                        self._create_edge(h2_id, discourse_node_id, "composition")
                    except Exception as e:
                        if self.debug:
                            print(f"Error creating discourse node for {h1_id}|{h2_id}: {e}")
                            traceback.print_exc()
        except Exception as e:
            if self.debug:
                print(f"Error in _build_discourse_patterns: {e}")
                traceback.print_exc()

    def _create_bigram_or_nodes(self):
        """Create OR nodes for bigrams (L2) that share components"""
        try:
            if self.debug:
                print("Creating OR nodes for bigrams (L2)...")

            # Collect bigrams by their component tokens
            bigrams_by_first_token = defaultdict(set)
            bigrams_by_second_token = defaultdict(set)

            # Group bigrams by their components
            for node_id, node in self.graph["nodes"].items():
                if node["level"] == "L2" and node["type"] == "AND" and "components" in node:
                    if len(node["components"]) >= 2:
                        first_token = node["components"][0]
                        second_token = node["components"][1]

                        bigrams_by_first_token[first_token].add(node_id)
                        bigrams_by_second_token[second_token].add(node_id)

            # Create OR nodes for bigrams sharing the same first token (divergence)
            for first_token, bigram_set in bigrams_by_first_token.items():
                if len(bigram_set) > 1:  # Multiple bigrams with the same first token
                    try:
                        # Create an OR node
                        or_node_id = self._create_node(
                            "OR",
                            "L2",
                            f"OR_out_L2_{first_token}",
                            components=list(bigram_set)
                        )

                        # Connect first token to OR node
                        # Connect first token to OR node
                        self._create_edge(first_token, or_node_id, "composition")

                        # Connect OR node to the bigrams
                        for bigram_id in bigram_set:
                            self._create_edge(or_node_id, bigram_id, "alternative")
                    except Exception as e:
                        if self.debug:
                            print(f"Error creating L2 divergence OR node for {first_token}: {e}")

            # Create OR nodes for bigrams sharing the same second token (convergence)
            for second_token, bigram_set in bigrams_by_second_token.items():
                if len(bigram_set) > 1:  # Multiple bigrams with the same second token
                    try:
                        # Create an OR node
                        or_node_id = self._create_node(
                            "OR",
                            "L2",
                            f"OR_in_L2_{second_token}",
                            components=list(bigram_set)
                        )

                        # Connect OR node to second token
                        self._create_edge(or_node_id, second_token, "composition")

                        # Connect bigrams to OR node
                        for bigram_id in bigram_set:
                            self._create_edge(bigram_id, or_node_id, "alternative")
                    except Exception as e:
                        if self.debug:
                            print(f"Error creating L2 convergence OR node for {second_token}: {e}")
        except Exception as e:
            if self.debug:
                print(f"Error in _create_bigram_or_nodes: {e}")
                traceback.print_exc()

    def _create_phrase_or_nodes(self):
        """Create OR nodes for phrases (L3) that share similar patterns"""
        try:
            if self.debug:
                print("Creating OR nodes for phrases (L3)...")

            # Collect phrases that start with the same token
            phrases_by_start = defaultdict(set)
            # Collect phrases that end with the same token
            phrases_by_end = defaultdict(set)
            # Collect phrases that contain the same bigrams
            phrases_by_bigram = defaultdict(set)

            # Group phrases
            for node_id, node in self.graph["nodes"].items():
                if node["level"] == "L3" and node["type"] == "AND" and "components" in node:
                    if len(node["components"]) >= 3:  # Phrase has at least 3 tokens
                        start_token = node["components"][0]
                        end_token = node["components"][-1]

                        phrases_by_start[start_token].add(node_id)
                        phrases_by_end[end_token].add(node_id)

                        # Track bigrams within this phrase
                        for i in range(len(node["components"]) - 1):
                            token1 = node["components"][i]
                            token2 = node["components"][i + 1]

                            # Find matching bigram node
                            for bigram_id, bigram_node in self.graph["nodes"].items():
                                if (bigram_node["level"] == "L2" and
                                        bigram_node["type"] == "AND" and
                                        "components" in bigram_node and
                                        len(bigram_node["components"]) == 2 and
                                        bigram_node["components"][0] == token1 and
                                        bigram_node["components"][1] == token2):
                                    phrases_by_bigram[bigram_id].add(node_id)
                                    break

            # Create OR nodes for phrases with the same start token
            for start_token, phrase_set in phrases_by_start.items():
                if len(phrase_set) > 1:
                    try:
                        # Create an OR node
                        or_node_id = self._create_node(
                            "OR",
                            "L3",
                            f"OR_start_L3_{start_token}",
                            components=list(phrase_set)
                        )

                        # Connect start token to OR node
                        self._create_edge(start_token, or_node_id, "composition")

                        # Connect OR node to phrases
                        for phrase_id in phrase_set:
                            self._create_edge(or_node_id, phrase_id, "alternative")
                    except Exception as e:
                        if self.debug:
                            print(f"Error creating L3 start OR node for {start_token}: {e}")

            # Create OR nodes for phrases with the same end token
            for end_token, phrase_set in phrases_by_end.items():
                if len(phrase_set) > 1:
                    try:
                        # Create an OR node
                        or_node_id = self._create_node(
                            "OR",
                            "L3",
                            f"OR_end_L3_{end_token}",
                            components=list(phrase_set)
                        )

                        # Connect phrases to OR node
                        for phrase_id in phrase_set:
                            self._create_edge(phrase_id, or_node_id, "alternative")

                        # Connect OR node to end token
                        self._create_edge(or_node_id, end_token, "composition")
                    except Exception as e:
                        if self.debug:
                            print(f"Error creating L3 end OR node for {end_token}: {e}")

            # Create OR nodes for phrases that share significant bigrams
            for bigram_id, phrase_set in phrases_by_bigram.items():
                if len(phrase_set) > 1:
                    try:
                        # Create an OR node
                        or_node_id = self._create_node(
                            "OR",
                            "L3",
                            f"OR_bigram_L3_{bigram_id}",
                            components=list(phrase_set)
                        )

                        # Connect bigram to OR node
                        self._create_edge(bigram_id, or_node_id, "composition")

                        # Connect OR node to phrases
                        for phrase_id in phrase_set:
                            self._create_edge(or_node_id, phrase_id, "alternative")
                    except Exception as e:
                        if self.debug:
                            print(f"Error creating L3 bigram OR node for {bigram_id}: {e}")
        except Exception as e:
            if self.debug:
                print(f"Error in _create_phrase_or_nodes: {e}")
                traceback.print_exc()

    def _create_hierarchy_or_nodes(self):
        """Create OR nodes for hierarchy structures (L4) that share components"""
        try:
            if self.debug:
                print("Creating OR nodes for hierarchies (L4)...")

            # Collect hierarchies by their component phrases
            hierarchies_by_phrase = defaultdict(set)

            # Group hierarchies by their component phrases
            for node_id, node in self.graph["nodes"].items():
                if node["level"] == "L4" and node["type"] == "AND" and "components" in node:
                    if len(node["components"]) >= 2:
                        for phrase_id in node["components"]:
                            hierarchies_by_phrase[phrase_id].add(node_id)

            # Create OR nodes for hierarchies sharing the same phrase
            for phrase_id, hierarchy_set in hierarchies_by_phrase.items():
                if len(hierarchy_set) > 1:  # Multiple hierarchies containing this phrase
                    try:
                        # Create an OR node
                        or_node_id = self._create_node(
                            "OR",
                            "L4",
                            f"OR_phrase_L4_{phrase_id}",
                            components=list(hierarchy_set)
                        )

                        # Connect phrase to OR node
                        self._create_edge(phrase_id, or_node_id, "composition")

                        # Connect OR node to hierarchies
                        for hierarchy_id in hierarchy_set:
                            self._create_edge(or_node_id, hierarchy_id, "alternative")
                    except Exception as e:
                        if self.debug:
                            print(f"Error creating L4 phrase OR node for {phrase_id}: {e}")
        except Exception as e:
            if self.debug:
                print(f"Error in _create_hierarchy_or_nodes: {e}")
                traceback.print_exc()

    def _create_discourse_or_nodes(self):
        """Create OR nodes for discourse patterns (L5) that share components"""
        try:
            if self.debug:
                print("Creating OR nodes for discourse patterns (L5)...")

            # Collect discourse patterns by their component hierarchies
            discourse_by_hierarchy = defaultdict(set)

            # Group discourse patterns by their components
            for node_id, node in self.graph["nodes"].items():
                if node["level"] == "L5" and node["type"] == "AND" and "components" in node:
                    if len(node["components"]) >= 2:
                        for hierarchy_id in node["components"]:
                            discourse_by_hierarchy[hierarchy_id].add(node_id)

            # Create OR nodes for discourse patterns sharing the same hierarchy
            for hierarchy_id, discourse_set in discourse_by_hierarchy.items():
                if len(discourse_set) > 1:  # Multiple discourse patterns with this hierarchy
                    try:
                        # Create an OR node
                        or_node_id = self._create_node(
                            "OR",
                            "L5",
                            f"OR_hierarchy_L5_{hierarchy_id}",
                            components=list(discourse_set)
                        )

                        # Connect hierarchy to OR node
                        self._create_edge(hierarchy_id, or_node_id, "composition")

                        # Connect OR node to discourse patterns
                        for discourse_id in discourse_set:
                            self._create_edge(or_node_id, discourse_id, "alternative")
                    except Exception as e:
                        if self.debug:
                            print(f"Error creating L5 hierarchy OR node for {hierarchy_id}: {e}")
        except Exception as e:
            if self.debug:
                print(f"Error in _create_discourse_or_nodes: {e}")
                traceback.print_exc()

    def update_metadata(self):
        """Update the metadata with current counts"""
        self.graph["metadata"]["token_count"] = sum(1 for node in self.graph["nodes"].values()
                                                    if node["level"] == "L1" and node["type"] == "AND")

    def build_from_file(self, filepath):
        """Build the graph model from sequences in a text file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            self.process_sequence(line)
                        except Exception as e:
                            print(f"Error processing line {line_num}: {line}")
                            print(f"Error details: {e}")
                            # Continue with next line instead of failing completely

            try:
                # Ensure we process any remaining sequences
                self._build_higher_levels()
            except Exception as e:
                print(f"Error building higher levels: {e}")
                traceback.print_exc()
                # Continue to metadata update

            # Update metadata
            self.update_metadata()

            return True
        except Exception as e:
            print(f"Error processing file: {e}")
            traceback.print_exc()
            return False

    def save_to_json(self, output_path):
        """Save the graph model to a JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(self.graph, file, indent=2)
            return True
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False

    def visualize(self, output_path=None, max_nodes=100, show=True):
        """
        Visualize the graph using NetworkX and matplotlib

        Args:
            output_path: Path to save the visualization image (optional)
            max_nodes: Maximum number of nodes to display (for readability)
            show: Whether to display the graph
        """
        if len(self.nx_graph) == 0:
            print("Graph is empty, nothing to visualize.")
            return False

        # Limit display to max_nodes for readability
        if len(self.nx_graph) > max_nodes:
            print(f"Graph has {len(self.nx_graph)} nodes, limiting visualization to {max_nodes} nodes.")

            # Create a subgraph with important nodes
            subgraph_nodes = []

            # Include higher level nodes first
            for level in ["L5", "L4", "L3", "L2", "L1"]:
                level_nodes = [n for n, d in self.nx_graph.nodes(data=True)
                               if d.get("level") == level]
                subgraph_nodes.extend(level_nodes)
                if len(subgraph_nodes) >= max_nodes:
                    break

            # Limit to max_nodes
            subgraph_nodes = subgraph_nodes[:max_nodes]

            # Create subgraph
            subgraph = self.nx_graph.subgraph(subgraph_nodes)
        else:
            subgraph = self.nx_graph

        # Create position layout
        pos = nx.spring_layout(subgraph, k=0.3, iterations=50)

        # Setup figure
        plt.figure(figsize=(15, 10))

        # Draw nodes
        node_colors = []
        node_sizes = []
        node_shapes = []

        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            level = node_data.get("level", "L1")
            node_type = node_data.get("type", "AND")

            # Determine node size based on level
            level_num = int(level[1])
            size = 300 + (level_num * 100)  # Larger nodes for higher levels
            node_sizes.append(size)

            # Determine node color
            if node_type == "AND":
                color = (0.2, 0.6, 0.9, 1.0)  # Blue for AND
            else:
                color = (0.9, 0.4, 0.2, 1.0)  # Orange for OR

            # Adjust color intensity for level
            intensity = 0.5 + (level_num * 0.1)
            color = (color[0] * intensity, color[1] * intensity, color[2] * intensity, color[3])
            node_colors.append(color)

            # Determine shape
            shape = "s" if node_type == "AND" else "o"  # Square for AND, Circle for OR
            node_shapes.append(shape)

        # Draw edges
        edge_colors = []
        edge_widths = []

        for u, v, data in subgraph.edges(data=True):
            edge_type = data.get("type", "sequence")

            if edge_type == "composition":
                color = "green"
            elif edge_type == "alternative":
                color = "red"
            else:  # sequence
                color = "gray"

            edge_colors.append(color)

            # Width based on weight
            width = data.get("weight", 1) * 0.5
            edge_widths.append(min(width, 3.0))  # Cap width for readability

        # Draw the graph
        for i, node in enumerate(subgraph.nodes()):
            nx.draw_networkx_nodes(
                subgraph, pos,
                nodelist=[node],
                node_color=[node_colors[i]],
                node_size=node_sizes[i],
                node_shape=node_shapes[i],
                alpha=0.8
            )

        nx.draw_networkx_edges(
            subgraph, pos,
            width=edge_widths,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=10,
            connectionstyle="arc3,rad=0.1"
        )

        # Draw labels
        nx.draw_networkx_labels(
            subgraph, pos,
            labels={n: d.get("label", n) for n, d in subgraph.nodes(data=True)},
            font_size=8,
            font_color="black"
        )

        # Add legend
        and_patch = plt.Line2D([0], [0], marker='s', color='w',
                               markerfacecolor=(0.2, 0.6, 0.9), markersize=10, label='AND')
        or_patch = plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=(0.9, 0.4, 0.2), markersize=10, label='OR')

        l1_line = plt.Line2D([0], [0], color='w', marker='o',
                             markerfacecolor=(0.2, 0.6, 0.9, 0.6), markersize=10, label='L1')
        l2_line = plt.Line2D([0], [0], color='w', marker='o',
                             markerfacecolor=(0.2, 0.6, 0.9, 0.7), markersize=12, label='L2')
        l3_line = plt.Line2D([0], [0], color='w', marker='o',
                             markerfacecolor=(0.2, 0.6, 0.9, 0.8), markersize=14, label='L3')
        l4_line = plt.Line2D([0], [0], color='w', marker='o',
                             markerfacecolor=(0.2, 0.6, 0.9, 0.9), markersize=16, label='L4')
        l5_line = plt.Line2D([0], [0], color='w', marker='o',
                             markerfacecolor=(0.2, 0.6, 0.9, 1.0), markersize=18, label='L5')

        seq_line = plt.Line2D([0], [0], color='gray', lw=2, label='Sequence')
        comp_line = plt.Line2D([0], [0], color='green', lw=2, label='Composition')
        alt_line = plt.Line2D([0], [0], color='red', lw=2, label='Alternative')

        plt.legend(handles=[and_patch, or_patch, l1_line, l2_line, l3_line, l4_line, l5_line,
                            seq_line, comp_line, alt_line], loc='upper left', bbox_to_anchor=(1, 1))

        plt.title("HTPC AND-OR Graph Visualization")
        plt.axis('off')
        plt.tight_layout()

        # Save if path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Visualization saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return True


# Example usage
if __name__ == "__main__":
    import sys

    # if len(sys.argv) < 3:
    #     print("Usage: python htpc_graph_builder.py <input_file> <output_file>")
    #     sys.exit(1)

    input_file = 'test_input.txt'
    output_file = 'htpc_model.json'

    builder = HTCPGraphBuilder(debug=True)
    print(f"Building graph from {input_file}...")

    if builder.build_from_file(input_file):
        print("Graph built successfully.")

        if builder.save_to_json(output_file):
            print(f"Graph saved to {output_file}")

            # Generate visualization
            viz_file = output_file.replace(".json", ".png")
            if builder.visualize(viz_file, max_nodes=50, show=False):
                print(f"Visualization saved to {viz_file}")
            else:
                print("Failed to create visualization.")
        else:
            print("Failed to save graph.")
    else:
        print("Failed to build graph.")