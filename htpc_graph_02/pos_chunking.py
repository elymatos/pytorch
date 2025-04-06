from typing import List, Dict, Tuple, Set, Optional, Any
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import math

class POSChunking:
    """
    Mixin class for chunk detection and management.
    """

    def __init__(self):
        # Higher-order chunk graph for learned patterns
        self.chunk_graph = nx.DiGraph()

        # Discovered chunks
        self.common_chunks = {}

        # Cohesion threshold
        self.cohesion_threshold = 0.7

    def _identify_common_chunks(self, pos_sequences: List[List[str]], boundary_probs: Dict[Tuple[str, str], float]):
        """
        Identify common chunks based on frequency and internal cohesion.

        Args:
            pos_sequences: List of POS sequences
            boundary_probs: Dictionary of boundary probabilities
        """
        # Use a sliding window approach to find potential chunks
        chunk_candidates = Counter()

        # Try different chunk sizes
        for size in range(2, 5):  # 2-grams to 4-grams
            for sequence in pos_sequences:
                if len(sequence) < size:
                    continue

                for i in range(len(sequence) - size + 1):
                    chunk = tuple(sequence[i:i + size])
                    chunk_candidates[chunk] += 1

        print(f"Found {len(chunk_candidates)} potential chunks")

        # For small training sets, lower the threshold
        total_sentences = len(pos_sequences)
        min_occurrences = max(2, int(total_sentences * 0.05))

        # Lower the cohesion threshold for small datasets
        self.cohesion_threshold = 0.6 if total_sentences < 20 else 0.7

        # Count qualifying chunks
        qualifying_chunks = 0
        for chunk, count in chunk_candidates.items():
            if count >= min_occurrences:
                qualifying_chunks += 1

                # Calculate internal cohesion
                internal_boundaries = 0

                for i in range(len(chunk) - 1):
                    pos1, pos2 = chunk[i], chunk[i + 1]

                    # Get boundary probability
                    boundary_prob = boundary_probs.get((pos1, pos2), 0.5)

                    # Accumulate boundary probability
                    internal_boundaries += boundary_prob

                avg_internal_boundary = internal_boundaries / (len(chunk) - 1)
                cohesion = 1 - avg_internal_boundary

                # Only keep reasonably cohesive chunks
                if cohesion > self.cohesion_threshold:
                    chunk_name = f"{'_'.join(chunk)}"
                    self.common_chunks[chunk] = {
                        "name": chunk_name,
                        "elements": chunk,
                        "count": count,
                        "cohesion": cohesion,
                        "activation": 0.0  # Initial activation level
                    }

        print(f"{qualifying_chunks} chunks met frequency criteria, {len(self.common_chunks)} met cohesion criteria")

    def _build_chunk_graph(self, trigram_counts: Dict):
        """
        Build higher-order graph representing transitions between chunks.

        Args:
            trigram_counts: Dictionary of trigram counts
        """
        # Add nodes for each chunk
        for chunk_tuple, chunk_info in self.common_chunks.items():
            chunk_name = chunk_info["name"]
            self.chunk_graph.add_node(
                chunk_name,
                pos_type="chunk",
                elements=chunk_info["elements"],
                cohesion=chunk_info["cohesion"]
            )

        # Connect chunks that can follow each other
        for chunk1_tuple, chunk1_info in self.common_chunks.items():
            for chunk2_tuple, chunk2_info in self.common_chunks.items():
                # Check if chunk2 can follow chunk1 (overlap or adjacency)
                if self._can_follow(chunk1_tuple, chunk2_tuple):
                    # Calculate transition probability
                    # This is simplified - would need corpus analysis for accurate probabilities
                    transition_prob = 0.1  # Default low probability

                    # If we have trigram data, use it to estimate transition probability
                    if len(chunk1_tuple) >= 2 and len(chunk2_tuple) >= 1:
                        last1, last2 = chunk1_tuple[-2], chunk1_tuple[-1]
                        first = chunk2_tuple[0]

                        if last1 in trigram_counts and last2 in trigram_counts[last1]:
                            total = sum(trigram_counts[last1][last2].values())
                            if total > 0:
                                count = trigram_counts[last1][last2].get(first, 0)
                                transition_prob = count / total

                    # Add edge with weight
                    chunk1_name = chunk1_info["name"]
                    chunk2_name = chunk2_info["name"]
                    self.chunk_graph.add_edge(
                        chunk1_name,
                        chunk2_name,
                        weight=transition_prob
                    )

    def _can_follow(self, chunk1: Tuple[str, ...], chunk2: Tuple[str, ...], graph=None) -> bool:
        """
        Determine if chunk2 can follow chunk1 in a sequence.
        Either through overlap or adjacency.

        Args:
            chunk1: First chunk
            chunk2: Second chunk
            graph: Graph to check for adjacency (if None, just check overlap)

        Returns:
            Boolean indicating if chunk2 can follow chunk1
        """
        # Check if there's an overlap
        for overlap_size in range(1, min(len(chunk1), len(chunk2))):
            if chunk1[-overlap_size:] == chunk2[:overlap_size]:
                return True

        # If no graph provided, only check for overlap
        if graph is None:
            return False

        # Check if there's an edge from the last element of chunk1 to the first of chunk2
        last_of_chunk1 = chunk1[-1]
        first_of_chunk2 = chunk2[0]

        return graph.has_edge(last_of_chunk1, first_of_chunk2)

    def recognize_chunks(self, pos_sequence: List[str]) -> List[Dict[str, Any]]:
        """
        Recognize known chunks in a POS sequence.

        Args:
            pos_sequence: List of POS tags

        Returns:
            List of recognized chunks with their properties
        """
        recognized = []

        # Try to match chunks of different sizes
        for i in range(len(pos_sequence)):
            for size in range(4, 1, -1):  # Try larger chunks first (4, 3, 2)
                if i + size <= len(pos_sequence):
                    chunk_tuple = tuple(pos_sequence[i:i + size])
                    if chunk_tuple in self.common_chunks:
                        # Calculate activation based on cohesion
                        chunk_info = self.common_chunks[chunk_tuple]
                        activation = chunk_info["cohesion"]

                        recognized.append({
                            "chunk": chunk_info,
                            "start": i,
                            "end": i + size,
                            "activation": activation
                        })

        # Sort by start position
        recognized.sort(key=lambda x: x["start"])

        return recognized

    def _resolve_chunk_overlaps(self, chunks: List[Dict[str, Any]], seq_length: int) -> List[Dict[str, Any]]:
        """
        Resolve overlapping chunks by selecting the most activated ones.

        Args:
            chunks: List of recognized chunks
            seq_length: Length of the original sequence

        Returns:
            List of non-overlapping chunks
        """
        # If no chunks, return empty list
        if not chunks:
            return []

        # Sort by activation to prioritize strongest chunks
        sorted_chunks = sorted(chunks, key=lambda x: x["activation"], reverse=True)

        # Track which positions are covered
        covered = [False] * seq_length

        # Select non-overlapping chunks
        selected = []

        for chunk in sorted_chunks:
            start, end = chunk["start"], chunk["end"]

            # Check if this chunk overlaps with already selected ones
            overlap = False
            for i in range(start, end):
                if covered[i]:
                    overlap = True
                    break

            if not overlap:
                # Add chunk and mark positions as covered
                selected.append(chunk)
                for i in range(start, end):
                    covered[i] = True

        # Sort by start position
        selected.sort(key=lambda x: x["start"])

        return selected

    def _create_final_segmentation(self, pos_sequence: List[str], chunks: List[Dict[str, Any]], segment_func) -> List[
        List[str]]:
        """
        Create final segmentation based on recognized chunks and boundary probabilities.

        Args:
            pos_sequence: Original POS sequence
            chunks: Non-overlapping chunks
            segment_func: Function to use for segmenting (e.g., self.segment)

        Returns:
            List of segments (chunks)
        """
        # If no chunks recognized, fall back to boundary-based segmentation
        if not chunks:
            return segment_func(pos_sequence)

        # Create segmentation based on recognized chunks and boundaries
        segmentation = []
        current_pos = 0

        for chunk in chunks:
            start, end = chunk["start"], chunk["end"]

            # If there's a gap before this chunk, segment it using boundaries
            if start > current_pos:
                gap_sequence = pos_sequence[current_pos:start]
                gap_segments = segment_func(gap_sequence)
                segmentation.extend(gap_segments)

            # Add the recognized chunk
            segmentation.append(pos_sequence[start:end])
            current_pos = end

        # Handle any remaining sequence after the last chunk
        if current_pos < len(pos_sequence):
            remaining = pos_sequence[current_pos:]
            remaining_segments = segment_func(remaining)
            segmentation.extend(remaining_segments)

        return segmentation

    def visualize_chunk_graph(self, filename: str = "chunk_graph.png"):
        """
        Visualize the chunk transition graph.

        Args:
            filename: Output file name
        """
        if len(self.chunk_graph) == 0:
            print("Chunk graph is empty - no visualization created")
            return

        if len(self.chunk_graph.edges()) == 0:
            print("Chunk graph has no edges - adding artificial edges for visualization")
            # Create some artificial edges just for visualization
            nodes = list(self.chunk_graph.nodes())
            if len(nodes) > 1:
                for i in range(len(nodes) - 1):
                    self.chunk_graph.add_edge(nodes[i], nodes[i + 1], weight=0.1)

        # Set up the plot
        plt.figure(figsize=(14, 12))

        # Define node positions using spring layout
        pos = nx.spring_layout(self.chunk_graph, seed=42)

        # Draw nodes with size based on cohesion
        node_sizes = []
        for node in self.chunk_graph.nodes():
            cohesion = self.chunk_graph.nodes[node].get("cohesion", 0.5)
            if cohesion <= 0:
                cohesion = 0.5  # Ensure minimum size
            node_sizes.append(cohesion * 800)

        nx.draw_networkx_nodes(
            self.chunk_graph, pos,
            node_size=node_sizes,
            node_color="lightgreen"
        )

        # Draw edges with width proportional to weight
        if len(self.chunk_graph.edges()) > 0:
            edge_width = []
            for _, _, data in self.chunk_graph.edges(data=True):
                weight = data.get("weight", 0.1)
                if weight <= 0:
                    weight = 0.1  # Ensure minimum width
                edge_width.append(weight * 8)

            nx.draw_networkx_edges(
                self.chunk_graph, pos, width=edge_width,
                edge_color="gray", alpha=0.6,
                connectionstyle="arc3,rad=0.1"
            )

        # Add labels
        nx.draw_networkx_labels(self.chunk_graph, pos, font_size=9)

        plt.title("Chunk Transition Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Chunk graph visualization saved to {filename}")
        plt.close()