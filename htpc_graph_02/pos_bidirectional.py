from typing import List, Dict, Tuple, Set, Optional, Any
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from collections import Counter, defaultdict
import math


class POSBidirectional:
    """
    Mixin class for bidirectional processing in POS graph models.
    """

    def __init__(self):
        # Forward and backward graphs
        self.forward_graph = nx.DiGraph()
        self.backward_graph = nx.DiGraph()

        # Boundary probabilities
        self.forward_boundary_probs = defaultdict(float)  # Forward direction
        self.backward_boundary_probs = defaultdict(float)  # Backward direction
        self.combined_boundary_probs = defaultdict(float)  # Combined

        # Bidirectional weights
        self.forward_weight = 0.6  # Weight for forward processing (typically higher)
        self.backward_weight = 0.4  # Weight for backward processing

    def _build_bidirectional_graphs(self, pos_sequences: List[List[str]]):
        """Build forward and backward graphs from the training sequences."""
        # Forward graph
        for sequence in pos_sequences:
            # Add nodes for each unique POS tag
            for pos in sequence:
                if not self.forward_graph.has_node(pos):
                    self.forward_graph.add_node(pos, pos_type="basic", precision=self.base_precision)
                self.unigram_counts[pos] += 1

            # Count bigrams and add edges (forward)
            for i in range(len(sequence) - 1):
                pos1, pos2 = sequence[i], sequence[i + 1]
                self.bigram_counts[pos1][pos2] += 1

                # Ensure edge exists (weight will be calculated later)
                if not self.forward_graph.has_edge(pos1, pos2):
                    self.forward_graph.add_edge(pos1, pos2, weight=0, count=0, boundary_prob=0,
                                                precision=self.base_precision)

                # Increment edge count
                self.forward_graph[pos1][pos2]["count"] += 1

            # Add connections from start and to end (forward)
            if sequence:
                if not self.forward_graph.has_edge("<START>", sequence[0]):
                    self.forward_graph.add_edge("<START>", sequence[0], weight=0, count=0, boundary_prob=0,
                                                precision=self.base_precision)
                self.forward_graph["<START>"][sequence[0]]["count"] += 1

                if not self.forward_graph.has_edge(sequence[-1], "<END>"):
                    self.forward_graph.add_edge(sequence[-1], "<END>", weight=0, count=0, boundary_prob=0,
                                                precision=self.base_precision)
                self.forward_graph[sequence[-1]]["<END>"]["count"] += 1

            # Count trigrams
            for i in range(len(sequence) - 2):
                pos1, pos2, pos3 = sequence[i], sequence[i + 1], sequence[i + 2]
                self.trigram_counts[pos1][pos2][pos3] += 1

        # Backward graph (reversed sequences)
        self.backward_bigram_counts = defaultdict(Counter)  # For backward transitions

        for sequence in pos_sequences:
            reversed_seq = list(reversed(sequence))

            # Add nodes for each unique POS tag in backward graph
            for pos in reversed_seq:
                if not self.backward_graph.has_node(pos):
                    self.backward_graph.add_node(pos, pos_type="basic", precision=self.base_precision)

            # Count bigrams and add edges (backward)
            for i in range(len(reversed_seq) - 1):
                pos1, pos2 = reversed_seq[i], reversed_seq[i + 1]
                self.backward_bigram_counts[pos1][pos2] += 1

                # Ensure edge exists in backward graph
                if not self.backward_graph.has_edge(pos1, pos2):
                    self.backward_graph.add_edge(pos1, pos2, weight=0, count=0, boundary_prob=0,
                                                 precision=self.base_precision)

                # Increment edge count
                self.backward_graph[pos1][pos2]["count"] += 1

            # Add connections from start and to end (backward)
            if reversed_seq:
                if not self.backward_graph.has_edge("<START>", reversed_seq[0]):
                    self.backward_graph.add_edge("<START>", reversed_seq[0], weight=0, count=0, boundary_prob=0,
                                                 precision=self.base_precision)
                self.backward_graph["<START>"][reversed_seq[0]]["count"] += 1

                if not self.backward_graph.has_edge(reversed_seq[-1], "<END>"):
                    self.backward_graph.add_edge(reversed_seq[-1], "<END>", weight=0, count=0, boundary_prob=0,
                                                 precision=self.base_precision)
                self.backward_graph[reversed_seq[-1]]["<END>"]["count"] += 1

    def _backward_pass(self, reversed_sequence: List[str]) -> Tuple[float, List[float]]:
        """
        Process a sequence in backward direction and calculate prediction errors.

        Args:
            reversed_sequence: A reversed sequence of POS tags

        Returns:
            Tuple of (total_surprisal, list_of_prediction_errors)
        """
        total_surprisal = 0.0
        prediction_errors = []

        # Start with START node
        current_pos = "<START>"

        # Process each position in the sequence
        for pos in reversed_sequence:
            # Calculate prediction probability for this position
            prediction_prob = 0.0
            if self.backward_graph.has_edge(current_pos, pos):
                prediction_prob = self.backward_graph[current_pos][pos].get("weight", 0.0)

            # Calculate surprisal (-log probability)
            if prediction_prob > 0:
                surprisal = -math.log2(prediction_prob)
            else:
                surprisal = 10.0  # High surprisal for unseen transitions

            # Get current precision for this transition
            precision = self.base_precision
            if self.backward_graph.has_edge(current_pos, pos):
                precision = self.backward_graph[current_pos][pos].get("precision", self.base_precision)

            # Calculate precision-weighted prediction error
            prediction_error = surprisal * precision

            prediction_errors.append(prediction_error)
            total_surprisal += surprisal

            # Update current position
            current_pos = pos

        return total_surprisal, prediction_errors

    def _update_attention_weights_bidirectional(self, sequence: List[str], forward_errors: List[float],
                                                reversed_sequence: List[str], backward_errors: List[float]):
        """
        Update attention weights based on prediction errors from both directions.

        Args:
            sequence: The forward POS sequence
            forward_errors: Prediction errors for forward transitions
            reversed_sequence: The reversed POS sequence
            backward_errors: Prediction errors for backward transitions
        """
        # Combine forward and backward errors for each position
        combined_errors = {}

        # Process forward errors
        if forward_errors:
            max_forward = max(forward_errors)
            if max_forward > 0:
                for i, pos in enumerate(sequence):
                    if i < len(forward_errors):
                        error = forward_errors[i] / max_forward
                        combined_errors[pos] = combined_errors.get(pos, 0) + error * self.forward_weight

        # Process backward errors (need to re-reverse to align with original positions)
        if backward_errors:
            max_backward = max(backward_errors)
            if max_backward > 0:
                backward_positions = list(reversed(reversed_sequence))  # Re-reverse to match original
                for i, pos in enumerate(backward_positions):
                    if i < len(backward_errors):
                        error_idx = len(backward_errors) - i - 1  # Map to correct backward error
                        if error_idx >= 0 and error_idx < len(backward_errors):
                            error = backward_errors[error_idx] / max_backward
                            combined_errors[pos] = combined_errors.get(pos, 0) + error * self.backward_weight

        # Update attention weights based on combined errors
        for pos, error in combined_errors.items():
            current_attention = self.attention_weights.get(pos, 1.0)
            # Update with learning rate
            self.attention_weights[pos] = (1 - self.attention_learning_rate) * current_attention + \
                                          self.attention_learning_rate * error

            # Ensure attention weights stay in reasonable range
            self.attention_weights[pos] = max(0.2, min(3.0, self.attention_weights[pos]))

    def _calculate_bidirectional_edge_weights(self):
        """Calculate edge weights for both forward and backward graphs."""
        # Forward graph
        for node in self.forward_graph.nodes():
            if node == "<END>":
                continue  # End node has no outgoing edges

            # Get total count of outgoing transitions
            outgoing_edges = list(self.forward_graph.out_edges(node, data=True))
            total_count = sum(data["count"] for _, _, data in outgoing_edges)

            if total_count > 0:
                # Calculate probability for each outgoing edge
                for _, target, data in outgoing_edges:
                    prob = data["count"] / total_count
                    self.forward_graph[node][target]["weight"] = prob

        # Backward graph
        for node in self.backward_graph.nodes():
            if node == "<END>":
                continue  # End node has no outgoing edges

            # Get total count of outgoing transitions
            outgoing_edges = list(self.backward_graph.out_edges(node, data=True))
            total_count = sum(data["count"] for _, _, data in outgoing_edges)

            if total_count > 0:
                # Calculate probability for each outgoing edge
                for _, target, data in outgoing_edges:
                    prob = data["count"] / total_count
                    self.backward_graph[node][target]["weight"] = prob

    def _calculate_boundary_probabilities_bidirectional(self):
        """Calculate boundary probabilities for both forward and backward graphs."""
        # Forward direction
        for source, target, data in self.forward_graph.edges(data=True):
            if source in ("<START>", "<END>") or target in ("<START>", "<END>"):
                continue  # Skip special nodes

            # Calculate surprisal for this transition
            prob = data.get("weight", 0)
            if prob > 0:
                surprisal = -math.log2(prob)

                # Get attention weights for source and target
                source_attention = self.attention_weights.get(source, 1.0)
                target_attention = self.attention_weights.get(target, 1.0)

                # Attention-modulated boundary probability
                # Higher attention at either end means more salient boundary
                attention_factor = (source_attention + target_attention) / 2

                # Normalize surprisal to a boundary probability between 0 and 1
                raw_boundary_prob = 1 / (1 + math.exp(-(surprisal - 1)))

                # Adjust boundary probability based on attention
                boundary_prob = raw_boundary_prob * attention_factor

                # Consider predefined boundaries if available
                if hasattr(self, 'predefined_boundaries') and (source, target) in self.predefined_boundaries:
                    predefined_prob = self.predefined_boundaries[(source, target)]
                    alpha = 0.3  # Weight for predefined rules
                    boundary_prob = alpha * predefined_prob + (1 - alpha) * boundary_prob

                # Store in graph and in lookup dictionary
                boundary_prob = max(0.0, min(1.0, boundary_prob))  # Ensure [0,1] range
                self.forward_graph[source][target]["boundary_prob"] = boundary_prob
                self.forward_boundary_probs[(source, target)] = boundary_prob

        # Backward direction
        for source, target, data in self.backward_graph.edges(data=True):
            if source in ("<START>", "<END>") or target in ("<START>", "<END>"):
                continue  # Skip special nodes

            # Calculate surprisal for this transition
            prob = data.get("weight", 0)
            if prob > 0:
                surprisal = -math.log2(prob)

                # Get attention weights for source and target
                source_attention = self.attention_weights.get(source, 1.0)
                target_attention = self.attention_weights.get(target, 1.0)

                # Attention-modulated boundary probability
                attention_factor = (source_attention + target_attention) / 2

                # Normalize surprisal to a boundary probability between 0 and 1
                raw_boundary_prob = 1 / (1 + math.exp(-(surprisal - 1)))

                # Adjust boundary probability based on attention
                boundary_prob = raw_boundary_prob * attention_factor

                # Handle predefined boundaries - note reversed order for backward graph
                if hasattr(self, 'predefined_boundaries') and (
                target, source) in self.predefined_boundaries:  # Reversed for backward direction
                    predefined_prob = self.predefined_boundaries[(target, source)]
                    alpha = 0.3  # Weight for predefined rules
                    boundary_prob = alpha * predefined_prob + (1 - alpha) * boundary_prob

                # Store in graph and in lookup dictionary
                boundary_prob = max(0.0, min(1.0, boundary_prob))  # Ensure [0,1] range
                self.backward_graph[source][target]["boundary_prob"] = boundary_prob
                self.backward_boundary_probs[(source, target)] = boundary_prob

    def _combine_boundary_probabilities(self):
        """Combine forward and backward boundary probabilities."""
        # First combine all the transitions found in either direction
        all_transitions = set(self.forward_boundary_probs.keys()) | set(self.backward_boundary_probs.keys())

        # For each transition, create a combined boundary probability
        for source, target in all_transitions:
            # Get forward boundary probability
            forward_prob = self.forward_boundary_probs.get((source, target), 0.0)

            # Get backward boundary probability (note: need to reverse direction)
            backward_prob = self.backward_boundary_probs.get((target, source), 0.0)

            # Weighted combination (can adjust weights as needed)
            combined_prob = self.forward_weight * forward_prob + self.backward_weight * backward_prob

            # Store the combined probability
            self.combined_boundary_probs[(source, target)] = combined_prob

    def _can_follow_bidirectional(self, chunk1: Tuple[str, ...], chunk2: Tuple[str, ...]) -> bool:
        """
        Determine if chunk2 can follow chunk1 in a sequence using bidirectional evidence.
        Either through overlap or adjacency.

        Args:
            chunk1: First chunk
            chunk2: Second chunk

        Returns:
            Boolean indicating if chunk2 can follow chunk1
        """
        # Check if there's an overlap
        for overlap_size in range(1, min(len(chunk1), len(chunk2))):
            if chunk1[-overlap_size:] == chunk2[:overlap_size]:
                return True

        # Check if there's an edge from the last element of chunk1 to the first of chunk2
        last_of_chunk1 = chunk1[-1]
        first_of_chunk2 = chunk2[0]

        # Check in both forward and backward graphs
        forward_connection = self.forward_graph.has_edge(last_of_chunk1, first_of_chunk2)
        backward_connection = self.backward_graph.has_edge(first_of_chunk2, last_of_chunk1)  # Note reversed order

        return forward_connection or backward_connection

    def bidirectional_segment(self, pos_sequence: List[str]) -> List[List[str]]:
        """
        Segment a POS sequence using bidirectional processing.

        Args:
            pos_sequence: List of POS tags for a sentence

        Returns:
            List of segments based on bidirectional evidence
        """
        # 1. Forward pass - left to right
        forward_chunks = self.segment_direction(pos_sequence, "forward")
        forward_boundaries = set()

        # Track boundary positions from forward segmentation
        position = 0
        for chunk in forward_chunks[:-1]:  # All but the last chunk
            position += len(chunk)
            forward_boundaries.add(position)

        # 2. Backward pass - right to left
        reversed_sequence = list(reversed(pos_sequence))
        backward_chunks = self.segment_direction(reversed_sequence, "backward")

        # Convert backward chunks to original sequence order
        aligned_backward_chunks = []
        for chunk in backward_chunks:
            aligned_backward_chunks.insert(0, list(reversed(chunk)))

        # Track boundary positions from backward segmentation
        backward_boundaries = set()
        position = 0
        for chunk in aligned_backward_chunks[:-1]:
            position += len(chunk)
            backward_boundaries.add(position)

        # 3. Combine evidence from both passes
        # Boundaries supported by both passes are strongest
        strong_boundaries = forward_boundaries.intersection(backward_boundaries)

        # Boundaries suggested by only one pass are candidates for review
        forward_only = forward_boundaries - backward_boundaries
        backward_only = backward_boundaries - forward_boundaries

        # Evaluate weak boundaries using attention weights
        accepted_weak_boundaries = set()

        # Check each forward-only boundary
        for boundary in forward_only:
            if boundary > 0 and boundary < len(pos_sequence):
                before_pos = pos_sequence[boundary - 1]
                after_pos = pos_sequence[boundary]

                # Consider attention weights at the boundary
                before_attention = self.attention_weights.get(before_pos, 1.0)
                after_attention = self.attention_weights.get(after_pos, 1.0)

                # Higher attention suggests more important boundary
                if (before_attention + after_attention) / 2 > 1.2:  # Threshold for accepting weak boundary
                    accepted_weak_boundaries.add(boundary)

        # Check each backward-only boundary
        for boundary in backward_only:
            if boundary > 0 and boundary < len(pos_sequence):
                before_pos = pos_sequence[boundary - 1]
                after_pos = pos_sequence[boundary]

                # Consider attention weights at the boundary
                before_attention = self.attention_weights.get(before_pos, 1.0)
                after_attention = self.attention_weights.get(after_pos, 1.0)

                # Higher attention suggests more important boundary
                if (before_attention + after_attention) / 2 > 1.2:  # Threshold for accepting weak boundary
                    accepted_weak_boundaries.add(boundary)

        # Combine strong and accepted weak boundaries
        final_boundaries = strong_boundaries.union(accepted_weak_boundaries)

        # 4. Create final segmentation
        final_chunks = []
        current_chunk = []

        for i, pos in enumerate(pos_sequence):
            current_chunk.append(pos)

            if i + 1 in final_boundaries:
                # Boundary found, end current chunk
                final_chunks.append(current_chunk)
                current_chunk = []

        # Add any remaining tokens
        if current_chunk:
            final_chunks.append(current_chunk)

        return final_chunks

    def segment_direction(self, pos_sequence: List[str], direction: str = "forward") -> List[List[str]]:
        """
        Segment a POS sequence in a specific direction.

        Args:
            pos_sequence: List of POS tags
            direction: Either "forward" or "backward"

        Returns:
            List of chunks
        """
        chunks = []
        current_chunk = [pos_sequence[0]]

        for i in range(1, len(pos_sequence)):
            pos1, pos2 = pos_sequence[i - 1], pos_sequence[i]

            # Get boundary probability based on direction
            if direction == "forward":
                boundary_prob = self.forward_boundary_probs.get((pos1, pos2), 0.2)
            else:  # backward
                boundary_prob = self.backward_boundary_probs.get((pos1, pos2), 0.2)

            # Apply attention weighting
            pos1_attention = self.attention_weights.get(pos1, 1.0)
            pos2_attention = self.attention_weights.get(pos2, 1.0)
            attention_factor = (pos1_attention + pos2_attention) / 2

            # Attention-modulated boundary decision
            effective_boundary = boundary_prob * attention_factor

            if effective_boundary > self.hard_boundary_threshold:
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

    def compare_segmentation_approaches(self, pos_sequence: List[str]):
        """
        Compare different segmentation approaches and return evaluation metrics.

        Args:
            pos_sequence: A POS sequence to segment

        Returns:
            Dictionary of metrics comparing segmentation approaches
        """
        # Get segmentations using different approaches
        forward_segments = self.segment_direction(pos_sequence, "forward")

        reversed_sequence = list(reversed(pos_sequence))
        backward_segments = self.segment_direction(reversed_sequence, "backward")

        # Convert backward segments to original direction
        aligned_backward = []
        for segment in backward_segments:
            aligned_backward.insert(0, list(reversed(segment)))

        # Get bidirectional segmentation
        bidirectional_segments = self.bidirectional_segment(pos_sequence)

        # Calculate number of boundaries for each approach
        forward_boundary_count = len(forward_segments) - 1
        backward_boundary_count = len(aligned_backward) - 1
        bidirectional_boundary_count = len(bidirectional_segments) - 1

        # Calculate average segment length
        forward_avg_length = len(pos_sequence) / len(forward_segments) if forward_segments else 0
        backward_avg_length = len(pos_sequence) / len(aligned_backward) if aligned_backward else 0
        bidirectional_avg_length = len(pos_sequence) / len(bidirectional_segments) if bidirectional_segments else 0

        # Calculate agreement metrics
        forward_backward_agreement = self._calculate_segmentation_agreement(forward_segments, aligned_backward)

        # Return comparison metrics
        return {
            "forward_segments": forward_segments,
            "backward_segments": aligned_backward,
            "bidirectional_segments": bidirectional_segments,
            "forward_boundary_count": forward_boundary_count,
            "backward_boundary_count": backward_boundary_count,
            "bidirectional_boundary_count": bidirectional_boundary_count,
            "forward_avg_length": forward_avg_length,
            "backward_avg_length": backward_avg_length,
            "bidirectional_avg_length": bidirectional_avg_length,
            "forward_backward_agreement": forward_backward_agreement
        }

    def _calculate_segmentation_agreement(self, segments1: List[List[str]], segments2: List[List[str]]) -> float:
        """
        Calculate agreement between two segmentations.

        Args:
            segments1: First segmentation
            segments2: Second segmentation

        Returns:
            Agreement score (0-1)
        """
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

    def predict_next_pos_bidirectional(self, context: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Predict the next POS tag with bidirectional attention-modulated probabilities.

        Args:
            context: List of preceding POS tags
            top_n: Number of top predictions to return

        Returns:
            List of (pos_tag, probability) pairs, sorted by probability
        """
        # Update context history
        self.context_history.append(context)
        if len(self.context_history) > 5:  # Keep only recent history
            self.context_history = self.context_history[-5:]

        if not context:
            # No context, use connections from start node
            predictions = []
            for target, data in self.forward_graph.out_edges("<START>", data=True):
                if target != "<END>":
                    # Apply attention weighting
                    base_prob = data.get("weight", 0.0)
                    target_attention = self.attention_weights.get(target, 1.0)
                    adjusted_prob = base_prob * target_attention
                    predictions.append((target, adjusted_prob))

            # Normalize probabilities
            total_prob = sum(prob for _, prob in predictions)
            if total_prob > 0:
                predictions = [(tag, prob / total_prob) for tag, prob in predictions]

            return sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

        # Use the last tag for prediction
        last_pos = context[-1]

        # Forward predictions (from forward graph)
        forward_predictions = {}
        if self.forward_graph.has_node(last_pos):
            source_attention = self.attention_weights.get(last_pos, 1.0)

            for _, target, data in self.forward_graph.out_edges(last_pos, data=True):
                if target != "<END>":
                    base_prob = data.get("weight", 0.0)
                    target_attention = self.attention_weights.get(target, 1.0)
                    combined_attention = math.sqrt(source_attention * target_attention)
                    adjusted_prob = base_prob * combined_attention
                    forward_predictions[target] = adjusted_prob * self.forward_weight

        # Backward predictions (looking at what comes before the context in backward graph)
        backward_predictions = {}
        if len(context) > 1 and self.backward_graph.has_node(last_pos):
            prev_pos = context[-2]  # Second-to-last position

            if self.backward_graph.has_node(prev_pos):
                prev_attention = self.attention_weights.get(prev_pos, 1.0)

                for _, target, data in self.backward_graph.out_edges(last_pos, data=True):
                    if target != "<END>" and target != prev_pos:
                        # This shows what might come after last_pos (in original sequence)
                        # based on backward graph
                        base_prob = data.get("weight", 0.0)
                        target_attention = self.attention_weights.get(target, 1.0)
                        combined_attention = math.sqrt(prev_attention * target_attention)
                        adjusted_prob = base_prob * combined_attention
                        backward_predictions[target] = adjusted_prob * self.backward_weight

        # Chunk-based predictions
        chunk_predictions = self._predict_from_chunks(context)
        chunk_pred_dict = {tag: prob * 0.5 for tag, prob in chunk_predictions}  # 50% weight for chunks

        # Combine all prediction sources
        combined_predictions = {}

        # Add forward predictions
        for tag, prob in forward_predictions.items():
            combined_predictions[tag] = combined_predictions.get(tag, 0) + prob

        # Add backward predictions
        for tag, prob in backward_predictions.items():
            combined_predictions[tag] = combined_predictions.get(tag, 0) + prob

        # Add chunk predictions
        for tag, prob in chunk_pred_dict.items():
            combined_predictions[tag] = combined_predictions.get(tag, 0) + prob

        # Convert to list and normalize
        predictions = [(tag, prob) for tag, prob in combined_predictions.items()]
        total_prob = sum(prob for _, prob in predictions)

        if total_prob > 0:
            predictions = [(tag, prob / total_prob) for tag, prob in predictions]

        return sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    def visualize_graphs(self, prefix="bidirectional_"):
        """Visualize all graphs (forward, backward, and chunk)."""
        self.visualize_pos_graph_with_attention(self.forward_graph,
                                                f"{prefix}forward_graph.png", "Forward")
        self.visualize_pos_graph_with_attention(self.backward_graph,
                                                f"{prefix}backward_graph.png", "Backward")
        self.visualize_chunk_graph_with_attention(f"{prefix}chunk_graph.png")

        # Plot learning progress
        self.plot_learning_progress(f"{prefix}learning_progress.png")