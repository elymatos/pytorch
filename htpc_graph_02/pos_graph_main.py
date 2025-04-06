import numpy as np
from collections import defaultdict, Counter
import math
from typing import List, Dict, Tuple, Set, Optional, Any
import networkx as nx
import matplotlib.pyplot as plt
import random
import time

# Import our modules
from pos_base import POSGraphBase
from pos_chunking import POSChunking
from pos_attention import POSAttention
from pos_bidirectional import POSBidirectional


class POSGraphBidirectional(POSGraphBase, POSChunking, POSAttention, POSBidirectional):
    """
    Complete implementation of a POS graph with bidirectional processing and attention mechanisms.
    This class combines the functionality from all modules.
    """

    def __init__(self, predefined_boundaries: Optional[Dict[Tuple[str, str], float]] = None):
        """
        Initialize the bidirectional POS graph with attention.

        Args:
            predefined_boundaries: Optional dictionary of predefined boundary probabilities
        """
        # Initialize all parent classes
        POSGraphBase.__init__(self, predefined_boundaries)
        POSChunking.__init__(self)
        POSAttention.__init__(self)
        POSBidirectional.__init__(self)

        # Special attributes for this combined class
        self.backward_bigram_counts = defaultdict(Counter)  # For backward transitions

    def train(self, pos_sequences: List[List[str]], epochs: int = 1):
        """
        Train the POS graph on a corpus of POS tag sequences with attention
        and bidirectional processing.

        Args:
            pos_sequences: List of POS tag sequences, each representing a sentence
            epochs: Number of training epochs
        """
        print(f"Training on {len(pos_sequences)} sequences for {epochs} epochs")
        start_time = time.time()

        # 1. Initialize attention weights uniformly
        self._initialize_attention_weights(pos_sequences)

        # 2. Build initial graphs (forward and backward) and collect statistics
        self._build_bidirectional_graphs(pos_sequences)

        # 3. Iterative training with attention modulation and bidirectional processing
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Shuffle sequences for stochastic training
            random.shuffle(pos_sequences)

            epoch_surprisal = 0.0

            # Process each sequence with attention
            for sequence in pos_sequences:
                # Forward pass - calculate predictions and errors
                sequence_surprisal, prediction_errors = self._forward_pass_with_attention(sequence, self.forward_graph)
                epoch_surprisal += sequence_surprisal

                # Backward pass - process the sequence in reverse
                reversed_sequence = list(reversed(sequence))
                backward_surprisal, backward_errors = self._backward_pass(reversed_sequence)
                epoch_surprisal += backward_surprisal

                # Update attention weights based on both forward and backward errors
                self._update_attention_weights_bidirectional(
                    sequence, prediction_errors, reversed_sequence, backward_errors)

                # Update graph precision based on errors
                self._update_graph_precision(sequence, prediction_errors, self.forward_graph)
                self._update_graph_precision(reversed_sequence, backward_errors, self.backward_graph)

            avg_surprisal = epoch_surprisal / (len(pos_sequences) * 2)  # Both directions
            self.surprisal_history.append(avg_surprisal)
            print(f"  Average surprisal: {avg_surprisal:.4f}")

            # Recalculate edge weights based on updated counts
            self._calculate_bidirectional_edge_weights()

            # After each epoch, update boundary probabilities (both directions)
            self._calculate_boundary_probabilities_with_attention(self.forward_graph, self.forward_boundary_probs)
            self._calculate_boundary_probabilities_with_attention(self.backward_graph, self.backward_boundary_probs)

            # Combine boundary probabilities from both directions
            self._combine_boundary_probabilities()

        # 4. Identify common chunks with attention influence
        self._identify_common_chunks(pos_sequences, self.combined_boundary_probs)

        # 5. Update chunk attention weights based on component POS tags
        self._update_chunk_attention_weights()

        # 6. Build chunk graph with attention-weighted connections
        self._build_chunk_graph(self.trigram_counts)

        training_time = time.time() - start_time
        print(f"Training complete in {training_time:.2f} seconds.")
        print(f"Forward graph has {len(self.forward_graph.nodes)} nodes and {len(self.forward_graph.edges)} edges")
        print(f"Backward graph has {len(self.backward_graph.nodes)} nodes and {len(self.backward_graph.edges)} edges")
        print(f"Chunk graph has {len(self.chunk_graph.nodes)} nodes and {len(self.chunk_graph.edges)} edges")

        # Report top attention weights
        self._report_attention_weights()

    def predictive_processing(self, pos_sequence: List[str]) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
        """
        Process a sequence using bidirectional predictive coding with attention.

        Args:
            pos_sequence: List of POS tags

        Returns:
            Tuple of (recognized chunks, segmented sequence)
        """
        # First pass: recognize chunks with attention
        recognized_chunks = self._recognize_chunks_with_attention(pos_sequence)

        # Second pass: resolve overlaps with attention-weighted resolution
        non_overlapping = self._resolve_chunk_overlaps(recognized_chunks, len(pos_sequence))

        # Third pass: bidirectional segmentation
        segmentation = self.bidirectional_segment(pos_sequence)

        return non_overlapping, segmentation

    def predict_next_pos(self, context: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Predict the next POS tag with bidirectional attention-modulated probabilities.

        Args:
            context: List of preceding POS tags
            top_n: Number of top predictions to return

        Returns:
            List of (pos_tag, probability) pairs, sorted by probability
        """
        return self.predict_next_pos_bidirectional(context, top_n)


# Example usage
if __name__ == "__main__":
    # Sample POS sequences for training - Create more examples with repeating patterns
    # to increase likelihood of chunk detection
    training_data = [
        ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"],
        ["PRON", "VERB", "PREP", "DET", "NOUN"],
        ["DET", "NOUN", "VERB", "ADV", "ADJ"],
        ["DET", "ADJ", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"],
        ["PRON", "VERB", "DET", "NOUN", "CONJ", "VERB", "ADV"],
        # More examples with repeating patterns to help chunk detection
        ["DET", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"],  # Repeat
        ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"],  # Repeat
        ["DET", "NOUN", "VERB", "PREP", "DET", "ADJ", "NOUN"],
        ["PRON", "VERB", "ADV", "CONJ", "VERB", "DET", "NOUN"],
        ["DET", "ADJ", "NOUN", "VERB", "ADV", "PREP", "PRON"],
        ["NOUN", "VERB", "DET", "ADJ", "NOUN", "PREP", "DET", "NOUN"],
        ["DET", "NOUN", "VERB", "ADJ", "CONJ", "ADV"],
        # Even more repetition of common patterns
        ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"],  # Repeat
        ["PRON", "VERB", "PREP", "DET", "NOUN"],  # Repeat
        ["DET", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"],  # Repeat
    ]

    print(f"Training on {len(training_data)} sentences")

    # Initialize and train the graph with bidirectional processing and attention mechanisms
    pos_graph = POSGraphBidirectional()
    pos_graph.train(training_data, epochs=3)  # Multiple epochs to allow attention to develop

    # Test on a new sentence
    test_sentence = ["DET", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"]

    print("\nTest sentence:", test_sentence)

    # Recognition and segmentation
    chunks, segments = pos_graph.predictive_processing(test_sentence)

    print("\nRecognized chunks:")
    if chunks:
        for chunk in chunks:
            print(f"  {chunk['chunk']['elements']} (Position {chunk['start']}-{chunk['end']}, " +
                  f"Activation: {chunk['activation']:.3f})")
    else:
        print("  No chunks recognized")

    print("\nBidirectional segmentation:", segments)

    # Compare segmentation approaches
    comparison = pos_graph.compare_segmentation_approaches(test_sentence)

    print("\nSegmentation comparison:")
    print(f"Forward segments: {comparison['forward_segments']}")
    print(f"Backward segments: {comparison['backward_segments']}")
    print(f"Bidirectional segments: {comparison['bidirectional_segments']}")
    print(f"Forward-backward agreement: {comparison['forward_backward_agreement']:.2f}")

    # Predictions
    context = ["DET", "ADJ"]
    predictions = pos_graph.predict_next_pos(context)
    print(f"\nTop predictions after {context}:")
    for pos, prob in predictions:
        print(f"  {pos}: {prob:.2f} (Attention: {pos_graph.attention_weights.get(pos, 1.0):.2f})")

    # Visualize all graphs
    pos_graph.visualize_graphs()