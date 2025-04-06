import numpy as np
from collections import defaultdict, Counter
import math
from typing import List, Dict, Tuple, Set, Optional, Any
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import math

class POSGraphBase:
    """
    Base class for POS graph implementations with common functionality.
    """

    def __init__(self, predefined_boundaries: Optional[Dict[Tuple[str, str], float]] = None):
        """
        Initialize the base POS graph.

        Args:
            predefined_boundaries: Optional dictionary of predefined boundary probabilities
        """
        # Graph for storing transitions
        self.graph = nx.DiGraph()

        # Track n-gram counts for training
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(lambda: defaultdict(Counter))

        # Boundary probabilities
        self.boundary_probs = defaultdict(float)

        # Predefined linguistic rules
        self.predefined_boundaries = predefined_boundaries or {
            ('NOUN', 'VERB'): 0.9,  # NP to VP transition
            ('VERB', 'DET'): 0.8,  # VP to NP transition
            ('PUNCT', 'DET'): 0.95,  # Punctuation followed by determiner
            ('NOUN', 'PREP'): 0.7,  # NP to PP transition
            ('VERB', 'PREP'): 0.6,  # VP to PP transition
            ('ADJ', 'NOUN'): 0.2,  # Within NP (low boundary probability)
            ('DET', 'ADJ'): 0.1,  # Within NP (very low boundary probability)
        }

        # Thresholds
        self.hard_boundary_threshold = 0.75
        self.soft_boundary_threshold = 0.4

        # Add special start and end nodes
        self.graph.add_node("<START>", pos_type="special")
        self.graph.add_node("<END>", pos_type="special")

    def _build_initial_graph(self, pos_sequences: List[List[str]]):
        """Build the initial graph structure and collect statistics."""
        # First pass - add all nodes and count statistics
        for sequence in pos_sequences:
            # Add nodes for each unique POS tag
            for pos in sequence:
                if not self.graph.has_node(pos):
                    self.graph.add_node(pos, pos_type="basic")
                self.unigram_counts[pos] += 1

            # Count bigrams and add edges
            for i in range(len(sequence) - 1):
                pos1, pos2 = sequence[i], sequence[i + 1]
                self.bigram_counts[pos1][pos2] += 1

                # Ensure edge exists (weight will be calculated later)
                if not self.graph.has_edge(pos1, pos2):
                    self.graph.add_edge(pos1, pos2, weight=0, count=0, boundary_prob=0)

                # Increment edge count
                self.graph[pos1][pos2]["count"] += 1

            # Add connections from start and to end
            if sequence:
                if not self.graph.has_edge("<START>", sequence[0]):
                    self.graph.add_edge("<START>", sequence[0], weight=0, count=0, boundary_prob=0)
                self.graph["<START>"][sequence[0]]["count"] += 1

                if not self.graph.has_edge(sequence[-1], "<END>"):
                    self.graph.add_edge(sequence[-1], "<END>", weight=0, count=0, boundary_prob=0)
                self.graph[sequence[-1]]["<END>"]["count"] += 1

            # Count trigrams
            for i in range(len(sequence) - 2):
                pos1, pos2, pos3 = sequence[i], sequence[i + 1], sequence[i + 2]
                self.trigram_counts[pos1][pos2][pos3] += 1

    def _calculate_edge_weights(self):
        """Calculate edge weights (transition probabilities) based on counts."""
        # For each node, calculate outgoing transition probabilities
        for node in self.graph.nodes():
            if node == "<END>":
                continue  # End node has no outgoing edges

            # Get total count of outgoing transitions
            outgoing_edges = list(self.graph.out_edges(node, data=True))
            total_count = sum(data["count"] for _, _, data in outgoing_edges)

            if total_count > 0:
                # Calculate probability for each outgoing edge
                for _, target, data in outgoing_edges:
                    prob = data["count"] / total_count
                    self.graph[node][target]["weight"] = prob

    def _calculate_boundary_probabilities(self):
        """Calculate boundary probabilities for each edge based on transition statistics."""
        for source, target, data in self.graph.edges(data=True):
            if source in ("<START>", "<END>") or target in ("<START>", "<END>"):
                continue  # Skip special nodes

            # Calculate surprisal for this transition
            prob = data.get("weight", 0)
            if prob > 0:
                surprisal = -math.log2(prob)

                # Normalize surprisal to a boundary probability between 0 and 1
                # Higher surprisal = higher boundary probability
                boundary_prob = 1 / (1 + math.exp(-(surprisal - 1)))

                # Consider predefined boundaries if available
                if (source, target) in self.predefined_boundaries:
                    predefined_prob = self.predefined_boundaries[(source, target)]
                    alpha = 0.3  # Weight for predefined rules
                    boundary_prob = alpha * predefined_prob + (1 - alpha) * boundary_prob

                # Store in graph and in lookup dictionary
                self.graph[source][target]["boundary_prob"] = boundary_prob
                self.boundary_probs[(source, target)] = boundary_prob

    def segment(self, pos_sequence: List[str]) -> List[List[str]]:
        """
        Segment a POS sequence into chunks based on boundary probabilities.

        Args:
            pos_sequence: List of POS tags for a sentence

        Returns:
            List of chunks, where each chunk is a list of POS tags
        """
        chunks = []
        current_chunk = [pos_sequence[0]]

        for i in range(1, len(pos_sequence)):
            pos1, pos2 = pos_sequence[i - 1], pos_sequence[i]

            # Get boundary probability
            boundary_prob = self.boundary_probs.get((pos1, pos2), 0.2)  # Default if unseen

            if boundary_prob > self.hard_boundary_threshold:
                # Hard boundary - create a new chunk
                chunks.append(current_chunk)
                current_chunk = [pos2]
            else:
                # Continue current chunk
                current_chunk.append(pos2)

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def predict_next_pos(self, context: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Predict the next POS tag given a context sequence.

        Args:
            context: List of preceding POS tags
            top_n: Number of top predictions to return

        Returns:
            List of (pos_tag, probability) pairs, sorted by probability
        """
        if not context:
            # No context, use connections from start node
            predictions = []
            for target, data in self.graph.out_edges("<START>", data=True):
                predictions.append((target, data["weight"]))
            return sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

        # Use the last tag for prediction
        last_pos = context[-1]

        if self.graph.has_node(last_pos):
            # Get all outgoing edges
            predictions = []
            for _, target, data in self.graph.out_edges(last_pos, data=True):
                if target != "<END>":  # Skip end node in predictions
                    predictions.append((target, data["weight"]))

            return sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
        else:
            # Unseen POS tag
            return [("<UNK>", 1.0)]  # Return unknown with full probability

    def visualize_pos_graph(self, filename: str = "pos_graph.png"):
        """
        Visualize the POS transition graph.

        Args:
            filename: Output file name
        """
        # Check if graph is empty
        if len(self.graph) <= 2:  # Only START and END nodes
            print("POS graph is empty or contains only special nodes - no visualization created")
            return

        # Create a copy without special nodes for cleaner visualization
        g = self.graph.copy()

        # Only remove special nodes if they exist
        if "<START>" in g:
            g.remove_node("<START>")
        if "<END>" in g:
            g.remove_node("<END>")

        if len(g.edges()) == 0:
            print("POS graph has no edges - no visualization created")
            return

        # Set up the plot
        plt.figure(figsize=(12, 10))

        # Define node positions using spring layout
        pos = nx.spring_layout(g, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(g, pos, node_size=500, node_color="lightblue")

        # Prepare edge attributes
        edge_width = []
        edge_color = []

        for _, _, data in g.edges(data=True):
            # Default to 0.1 if weight is missing or zero
            weight = data.get("weight", 0.1)
            if weight == 0:
                weight = 0.1
            edge_width.append(weight * 5)

            # Default to 0.5 if boundary_prob is missing
            edge_color.append(data.get("boundary_prob", 0.5))

        # Draw edges
        nx.draw_networkx_edges(
            g, pos, width=edge_width,
            edge_color=edge_color, edge_cmap=plt.cm.Reds,
            connectionstyle="arc3,rad=0.1"
        )

        # Add labels
        nx.draw_networkx_labels(g, pos, font_size=10)

        # Edge labels (probabilities)
        edge_labels = {}
        for u, v, d in g.edges(data=True):
            if "weight" in d:
                edge_labels[(u, v)] = f"{d['weight']:.2f}"
            else:
                edge_labels[(u, v)] = "0.00"

        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8)

        # Add a color bar for boundary probabilities
        fig = plt.gcf()
        ax = plt.gca()
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Boundary Probability")

        plt.title("POS Transition Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename)
        print(f"POS graph visualization saved to {filename}")
        plt.close()