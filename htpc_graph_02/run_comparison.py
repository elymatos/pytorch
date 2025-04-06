"""
Script to compare bidirectional and unidirectional POS processing models.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict
import random
# Import our model classes
from pos_graph_main import POSGraphBidirectional
from pos_base import POSGraphBase
from pos_chunking import POSChunking
from pos_attention import POSAttention


# Create a unidirectional model using only some of the mixins
class POSGraphUnidirectional(POSGraphBase, POSChunking, POSAttention):
    """
    Unidirectional POS graph with attention but without bidirectional processing.
    """

    def __init__(self, predefined_boundaries=None):
        POSGraphBase.__init__(self, predefined_boundaries)
        POSChunking.__init__(self)
        POSAttention.__init__(self)

    def train(self, pos_sequences, epochs=1):
        """
        Train the unidirectional model.
        """
        print(f"Training unidirectional model on {len(pos_sequences)} sequences for {epochs} epochs")

        # Initialize attention weights
        self._initialize_attention_weights(pos_sequences)

        # Build initial graph
        self._build_initial_graph(pos_sequences)

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Shuffle sequences
            random.shuffle(pos_sequences)

            epoch_surprisal = 0.0

            # Process each sequence
            for sequence in pos_sequences:
                # Forward pass with attention
                surprisal, errors = self._forward_pass_with_attention(sequence, self.graph)
                epoch_surprisal += surprisal

                # Update attention weights
                self._update_attention_weights(sequence, errors)

                # Update graph precision
                self._update_graph_precision(sequence, errors, self.graph)

            # Calculate average surprisal
            avg_surprisal = epoch_surprisal / len(pos_sequences)
            self.surprisal_history.append(avg_surprisal)
            print(f"  Average surprisal: {avg_surprisal:.4f}")

            # Update weights and boundary probabilities
            self._calculate_edge_weights()
            self._calculate_boundary_probabilities_with_attention(self.graph, self.boundary_probs)

        # Identify chunks and build chunk graph
        self._identify_common_chunks(pos_sequences, self.boundary_probs)
        self._update_chunk_attention_weights()
        self._build_chunk_graph(self.trigram_counts)

        # Report stats
        print(f"Unidirectional model has {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        print(f"Chunk graph has {len(self.chunk_graph.nodes)} nodes and {len(self.chunk_graph.edges)} edges")

        # Report attention weights
        self._report_attention_weights()

    def predictive_processing(self, pos_sequence):
        """
        Process a sequence using unidirectional predictive coding with attention.
        """
        # Recognize chunks with attention
        recognized_chunks = self._recognize_chunks_with_attention(pos_sequence)

        # Resolve overlaps
        non_overlapping = self._resolve_chunk_overlaps(recognized_chunks, len(pos_sequence))

        # Create segmentation
        segmentation = self.segment_with_attention(pos_sequence, self.boundary_probs, self.hard_boundary_threshold)

        return non_overlapping, segmentation


def train_and_compare(training_data, test_sentences, epochs=3):
    """
    Train both models and compare their performance.

    Args:
        training_data: List of POS sequences for training
        test_sentences: List of POS sequences for testing
        epochs: Number of training epochs

    Returns:
        Dictionary of comparison results
    """
    # Train bidirectional model
    print("\n=== Training Bidirectional Model ===")
    bidirectional_model = POSGraphBidirectional()
    bidirectional_model.train(training_data, epochs=epochs)

    # Train unidirectional model
    print("\n=== Training Unidirectional Model ===")
    unidirectional_model = POSGraphUnidirectional()
    unidirectional_model.train(training_data, epochs=epochs)

    # Compare learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(bidirectional_model.surprisal_history, 'b-o', label="Bidirectional")
    plt.plot(unidirectional_model.surprisal_history, 'r-o', label="Unidirectional")
    plt.title("Learning Progress Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Average Surprisal")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_comparison.png")
    plt.close()

    # Test on sentences
    results = []

    for i, sentence in enumerate(test_sentences):
        print(f"\n--- Testing Sentence {i + 1}: {sentence} ---")

        # Process with both models
        bi_chunks, bi_segments = bidirectional_model.predictive_processing(sentence)
        uni_chunks, uni_segments = unidirectional_model.predictive_processing(sentence)

        # Get predictions
        context = sentence[:2] if len(sentence) >= 2 else sentence
        bi_predictions = bidirectional_model.predict_next_pos(context)
        uni_predictions = unidirectional_model.predict_next_pos_with_attention(
            context, unidirectional_model.graph)

        # Compare results
        bi_chunk_count = len(bi_chunks)
        uni_chunk_count = len(uni_chunks)

        bi_segment_count = len(bi_segments)
        uni_segment_count = len(uni_segments)

        # Calculate segment agreement
        segment_agreement = calculate_segmentation_agreement(bi_segments, uni_segments)

        # Calculate chunk overlap
        chunk_overlap = calculate_chunk_overlap(bi_chunks, uni_chunks)

        # Calculate prediction similarity
        prediction_similarity = calculate_prediction_similarity(bi_predictions, uni_predictions)

        # Store results
        results.append({
            "sentence": sentence,
            "bi_chunks": bi_chunks,
            "uni_chunks": uni_chunks,
            "bi_segments": bi_segments,
            "uni_segments": uni_segments,
            "bi_chunk_count": bi_chunk_count,
            "uni_chunk_count": uni_chunk_count,
            "bi_segment_count": bi_segment_count,
            "uni_segment_count": uni_segment_count,
            "segment_agreement": segment_agreement,
            "chunk_overlap": chunk_overlap,
            "prediction_similarity": prediction_similarity,
            "bi_predictions": bi_predictions,
            "uni_predictions": uni_predictions
        })

        # Print comparison
        print(f"Bidirectional segments: {bi_segments}")
        print(f"Unidirectional segments: {uni_segments}")
        print(f"Segment agreement: {segment_agreement:.2f}")
        print(f"Chunk overlap: {chunk_overlap:.2f}")
        print(f"Prediction similarity: {prediction_similarity:.2f}")

    # Calculate summary statistics
    summary = calculate_summary_statistics(results)

    # Visualize model graphs for comparison
    bidirectional_model.visualize_pos_graph_with_attention(
        bidirectional_model.forward_graph, "bidirectional_graph.png")
    unidirectional_model.visualize_pos_graph_with_attention(
        unidirectional_model.graph, "unidirectional_graph.png")

    # Return all results
    return {
        "bidirectional_model": bidirectional_model,
        "unidirectional_model": unidirectional_model,
        "test_results": results,
        "summary": summary
    }


def calculate_segmentation_agreement(segments1, segments2):
    """Calculate agreement between two segmentations using Jaccard index."""
    if not segments1 or not segments2:
        return 0.0

    # Create boundary sets
    boundaries1 = set()
    boundaries2 = set()

    pos1 = 0
    for segment in segments1[:-1]:  # All except last
        pos1 += len(segment)
        boundaries1.add(pos1)

    pos2 = 0
    for segment in segments2[:-1]:  # All except last
        pos2 += len(segment)
        boundaries2.add(pos2)

    # Calculate agreement using Jaccard index
    intersection = len(boundaries1.intersection(boundaries2))
    union = len(boundaries1.union(boundaries2))

    if union == 0:
        return 1.0  # Both have no boundaries, perfect agreement

    return intersection / union


def calculate_chunk_overlap(chunks1, chunks2):
    """Calculate overlap between recognized chunks from two models."""
    if not chunks1 or not chunks2:
        return 0.0

    # Create sets of chunk positions
    positions1 = set()
    positions2 = set()

    for chunk in chunks1:
        start, end = chunk["start"], chunk["end"]
        for pos in range(start, end):
            positions1.add(pos)

    for chunk in chunks2:
        start, end = chunk["start"], chunk["end"]
        for pos in range(start, end):
            positions2.add(pos)

    # Calculate overlap
    intersection = len(positions1.intersection(positions2))
    union = len(positions1.union(positions2))

    if union == 0:
        return 1.0  # No chunks in either model

    return intersection / union


def calculate_prediction_similarity(pred1, pred2):
    """Calculate similarity between two sets of predictions."""
    if not pred1 or not pred2:
        return 0.0

    # Convert to dictionaries
    dict1 = {tag: prob for tag, prob in pred1}
    dict2 = {tag: prob for tag, prob in pred2}

    # Get all tags
    all_tags = set(dict1.keys()) | set(dict2.keys())

    # Calculate cosine similarity
    dot_product = sum(dict1.get(tag, 0) * dict2.get(tag, 0) for tag in all_tags)
    magnitude1 = sum(dict1.get(tag, 0) ** 2 for tag in all_tags) ** 0.5
    magnitude2 = sum(dict2.get(tag, 0) ** 2 for tag in all_tags) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def calculate_summary_statistics(results):
    """Calculate summary statistics across all test results."""
    # Average statistics
    avg_segment_agreement = np.mean([r["segment_agreement"] for r in results])
    avg_chunk_overlap = np.mean([r["chunk_overlap"] for r in results])
    avg_prediction_similarity = np.mean([r["prediction_similarity"] for r in results])

    # Count statistics
    avg_bi_chunks = np.mean([r["bi_chunk_count"] for r in results])
    avg_uni_chunks = np.mean([r["uni_chunk_count"] for r in results])
    avg_bi_segments = np.mean([r["bi_segment_count"] for r in results])
    avg_uni_segments = np.mean([r["uni_segment_count"] for r in results])

    return {
        "avg_segment_agreement": avg_segment_agreement,
        "avg_chunk_overlap": avg_chunk_overlap,
        "avg_prediction_similarity": avg_prediction_similarity,
        "avg_bi_chunks": avg_bi_chunks,
        "avg_uni_chunks": avg_uni_chunks,
        "avg_bi_segments": avg_bi_segments,
        "avg_uni_segments": avg_uni_segments
    }


if __name__ == "__main__":
    # Sample training data
    training_data = [
        ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"],
        ["PRON", "VERB", "PREP", "DET", "NOUN"],
        ["DET", "NOUN", "VERB", "ADV", "ADJ"],
        ["DET", "ADJ", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"],
        ["PRON", "VERB", "DET", "NOUN", "CONJ", "VERB", "ADV"],
        # More examples with repeating patterns
        ["DET", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"],
        ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"],
        ["DET", "NOUN", "VERB", "PREP", "DET", "ADJ", "NOUN"],
        ["PRON", "VERB", "ADV", "CONJ", "VERB", "DET", "NOUN"],
        ["DET", "ADJ", "NOUN", "VERB", "ADV", "PREP", "PRON"],
        ["NOUN", "VERB", "DET", "ADJ", "NOUN", "PREP", "DET", "NOUN"],
        ["DET", "NOUN", "VERB", "ADJ", "CONJ", "ADV"],
        # Repetition for better learning
        ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"],
        ["PRON", "VERB", "PREP", "DET", "NOUN"],
        ["DET", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"],
    ]

    # Test sentences
    test_sentences = [
        ["DET", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"],
        ["PRON", "VERB", "DET", "ADJ", "NOUN"],
        ["DET", "NOUN", "VERB", "ADV", "PREP", "DET", "NOUN"],
        ["ADJ", "NOUN", "VERB", "ADV", "CONJ", "VERB", "PREP", "NOUN"]
    ]

    # Run comparison
    results = train_and_compare(training_data, test_sentences, epochs=3)

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for metric, value in results["summary"].items():
        print(f"{metric}: {value:.4f}")

    # Generate performance comparison plot
    metrics = ["avg_segment_agreement", "avg_chunk_overlap", "avg_prediction_similarity"]
    values = [results["summary"][m] for m in metrics]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=["blue", "green", "orange"])
    plt.title("Performance Comparison: Bidirectional vs Unidirectional")
    plt.ylabel("Score (0-1)")
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig("performance_comparison.png")

    print("\nComparison complete. Visualizations saved as PNG files.")