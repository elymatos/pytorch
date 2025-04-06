from typing import List, Dict, Tuple, Set, Optional, Any
import math
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import math

class POSAttention:
    """
    Mixin class for attention mechanisms in POS graph processing.
    """

    def __init__(self):
        # Attention mechanisms
        self.attention_weights = {}  # POS tag attention weights
        self.chunk_attention_weights = {}  # Chunk attention weights
        self.surprisal_history = []  # Track surprisal for adaptive attention
        self.attention_learning_rate = 0.05  # For updating attention weights

        # Precision weighting parameters
        self.base_precision = 1.0
        self.max_precision = 5.0
        self.min_precision = 0.2

        # Context tracking
        self.context_history = []  # Track recent contexts for attention modulation

    def _initialize_attention_weights(self, pos_sequences: List[List[str]]):
        """Initialize attention weights for all POS tags."""
        # Extract all unique POS tags from sequences
        unique_pos = set()
        for sequence in pos_sequences:
            unique_pos.update(sequence)

        # Initialize with uniform weights
        for pos in unique_pos:
            self.attention_weights[pos] = 1.0

        # Special start and end nodes
        self.attention_weights["<START>"] = 1.0
        self.attention_weights["<END>"] = 1.0

    def _update_attention_weights(self, sequence: List[str], prediction_errors: List[float]):
        """
        Update attention weights based on prediction errors.
        Higher errors lead to increased attention.

        Args:
            sequence: The POS sequence
            prediction_errors: Corresponding prediction errors for transitions
        """
        # Normalize prediction errors to [0,1] range for attention updates
        if prediction_errors:
            max_error = max(prediction_errors)
            if max_error > 0:
                normalized_errors = [error / max_error for error in prediction_errors]
            else:
                normalized_errors = [0.0] * len(prediction_errors)

            # Update attention for START node and first token
            self.attention_weights["<START>"] = (1 - self.attention_learning_rate) * self.attention_weights["<START>"] + \
                                                self.attention_learning_rate * normalized_errors[0]

            # Update attention weights for each POS tag based on prediction errors
            for i, pos in enumerate(sequence):
                # Current position's error influences its attention
                if i < len(normalized_errors):
                    error_weight = normalized_errors[i]

                    # Update attention weight with learning rate
                    self.attention_weights[pos] = (1 - self.attention_learning_rate) * self.attention_weights.get(pos,
                                                                                                                  1.0) + \
                                                  self.attention_learning_rate * error_weight

                    # Ensure attention weights stay in reasonable range
                    self.attention_weights[pos] = max(0.2, min(3.0, self.attention_weights[pos]))

    def _forward_pass_with_attention(self, sequence: List[str], graph) -> Tuple[float, List[float]]:
        """
        Process a sequence and calculate prediction errors with attention modulation.

        Args:
            sequence: A sequence of POS tags
            graph: The graph to use for predictions

        Returns:
            Tuple of (total_surprisal, list_of_prediction_errors)
        """
        total_surprisal = 0.0
        prediction_errors = []

        # Start with START node
        current_pos = "<START>"

        # Process each position in the sequence
        for pos in sequence:
            # Calculate prediction probability for this position
            prediction_prob = 0.0
            if graph.has_edge(current_pos, pos):
                prediction_prob = graph[current_pos][pos].get("weight", 0.0)

            # Calculate surprisal (-log probability)
            if prediction_prob > 0:
                surprisal = -math.log2(prediction_prob)
            else:
                surprisal = 10.0  # High surprisal for unseen transitions

            # Get current precision for this transition
            precision = self.base_precision
            if graph.has_edge(current_pos, pos):
                precision = graph[current_pos][pos].get("precision", self.base_precision)

            # Calculate precision-weighted prediction error
            prediction_error = surprisal * precision

            prediction_errors.append(prediction_error)
            total_surprisal += surprisal

            # Update current position
            current_pos = pos

        return total_surprisal, prediction_errors

    def _update_graph_precision(self, sequence: List[str], prediction_errors: List[float], graph):
        """
        Update graph edge precision based on attention-modulated learning.

        Args:
            sequence: The POS sequence
            prediction_errors: Corresponding prediction errors
            graph: The graph to update
        """
        current_pos = "<START>"

        for i, pos in enumerate(sequence):
            # Get attention weights for current and next position
            current_attention = self.attention_weights.get(current_pos, 1.0)
            target_attention = self.attention_weights.get(pos, 1.0)

            # Combined attention effect (geometric mean)
            combined_attention = math.sqrt(current_attention * target_attention)

            # Update precision for this transition based on prediction error
            if i < len(prediction_errors):
                error = prediction_errors[i]
                new_precision = self.base_precision * (1 + 0.1 * error)
                new_precision = max(self.min_precision, min(self.max_precision, new_precision))

                if graph.has_edge(current_pos, pos):
                    # Slowly update precision for this edge
                    old_precision = graph[current_pos][pos].get("precision", self.base_precision)
                    updated_precision = 0.9 * old_precision + 0.1 * new_precision
                    graph[current_pos][pos]["precision"] = updated_precision

            # Update current position
            current_pos = pos

    def _calculate_boundary_probabilities_with_attention(self, graph, boundary_probs):
        """
        Calculate boundary probabilities with attention influence.

        Args:
            graph: The graph to process
            boundary_probs: Dictionary to store resulting boundary probabilities
        """
        for source, target, data in graph.edges(data=True):
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
                graph[source][target]["boundary_prob"] = boundary_prob
                boundary_probs[(source, target)] = boundary_prob

    def segment_with_attention(self, pos_sequence: List[str], boundary_probs, hard_boundary_threshold) -> List[
        List[str]]:
        """
        Segment a POS sequence using attention-weighted boundary probabilities.

        Args:
            pos_sequence: List of POS tags
            boundary_probs: Dictionary of boundary probabilities
            hard_boundary_threshold: Threshold for creating a boundary

        Returns:
            List of chunks
        """
        chunks = []
        current_chunk = [pos_sequence[0]]

        for i in range(1, len(pos_sequence)):
            pos1, pos2 = pos_sequence[i - 1], pos_sequence[i]

            # Get boundary probability
            boundary_prob = boundary_probs.get((pos1, pos2), 0.2)  # Default if unseen

            # Apply attention weighting
            pos1_attention = self.attention_weights.get(pos1, 1.0)
            pos2_attention = self.attention_weights.get(pos2, 1.0)
            attention_factor = (pos1_attention + pos2_attention) / 2

            # Attention-modulated boundary decision
            effective_boundary = boundary_prob * attention_factor

            if effective_boundary > hard_boundary_threshold:
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

    def _update_chunk_attention_weights(self):
        """Update attention weights for chunks based on component POS tags."""
        for chunk_tuple in self.common_chunks:
            # Calculate chunk attention as average of component attentions
            component_attentions = [self.attention_weights.get(pos, 1.0) for pos in chunk_tuple]
            avg_attention = sum(component_attentions) / len(component_attentions)

            # Store chunk attention
            self.chunk_attention_weights[chunk_tuple] = avg_attention

            # Also update attention field in common_chunks dictionary
            self.common_chunks[chunk_tuple]["attention"] = avg_attention

    def _recognize_chunks_with_attention(self, pos_sequence: List[str]) -> List[Dict[str, Any]]:
        """
        Recognize known chunks in a POS sequence with attention modulation.

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
                        # Calculate activation based on cohesion and attention
                        chunk_info = self.common_chunks[chunk_tuple]
                        base_activation = chunk_info["cohesion"]
                        chunk_attention = self.chunk_attention_weights.get(chunk_tuple, 1.0)

                        # Attention-modulated activation
                        activation = base_activation * chunk_attention

                        recognized.append({
                            "chunk": chunk_info,
                            "start": i,
                            "end": i + size,
                            "activation": activation
                        })

        # Sort by start position
        recognized.sort(key=lambda x: x["start"])

        return recognized

    def predict_next_pos_with_attention(self, context: List[str], graph, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Predict the next POS tag with attention-modulated probabilities.

        Args:
            context: List of preceding POS tags
            graph: Graph to use for predictions
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
            for target, data in graph.out_edges("<START>", data=True):
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

        if graph.has_node(last_pos):
            # Apply attention to outgoing predictions
            predictions = []

            # Get attention for the source position
            source_attention = self.attention_weights.get(last_pos, 1.0)

            # First try to use chunk-based prediction if we have matching chunks
            chunk_predictions = self._predict_from_chunks(context)

            if chunk_predictions:
                # If we have chunk-based predictions, give them more weight
                # but also include some direct edge predictions
                direct_predictions = []
                for _, target, data in graph.out_edges(last_pos, data=True):
                    if target != "<END>":
                        base_prob = data.get("weight", 0.0)
                        target_attention = self.attention_weights.get(target, 1.0)
                        # Combined attention effect (geometric mean)
                        combined_attention = math.sqrt(source_attention * target_attention)
                        adjusted_prob = base_prob * combined_attention
                        direct_predictions.append((target, adjusted_prob))

                # Normalize direct predictions
                total_direct = sum(prob for _, prob in direct_predictions)
                if total_direct > 0:
                    direct_predictions = [(tag, prob / total_direct) for tag, prob in direct_predictions]

                # Combine chunk-based and direct predictions with 3:1 weighting
                combined = {}
                for tag, prob in chunk_predictions:
                    combined[tag] = prob * 0.75

                for tag, prob in direct_predictions:
                    if tag in combined:
                        combined[tag] += prob * 0.25
                    else:
                        combined[tag] = prob * 0.25

                predictions = [(tag, prob) for tag, prob in combined.items()]
            else:
                # No chunk predictions, use direct edge predictions
                for _, target, data in graph.out_edges(last_pos, data=True):
                    if target != "<END>":
                        base_prob = data.get("weight", 0.0)
                        target_attention = self.attention_weights.get(target, 1.0)
                        # Combined attention effect
                        combined_attention = math.sqrt(source_attention * target_attention)
                        adjusted_prob = base_prob * combined_attention
                        predictions.append((target, adjusted_prob))

            # Normalize probabilities
            total_prob = sum(prob for _, prob in predictions)
            if total_prob > 0:
                predictions = [(tag, prob / total_prob) for tag, prob in predictions]

            return sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
        else:
            # Unseen POS tag
            return [("<UNK>", 1.0)]  # Return unknown with full probability

    def _predict_from_chunks(self, context: List[str]) -> List[Tuple[str, float]]:
        """
        Generate predictions based on chunk matching.

        Args:
            context: The context sequence

        Returns:
            List of (pos_tag, probability) tuples
        """
        if len(context) < 1 or not hasattr(self, 'common_chunks'):
            return []

        # Try to match the end of the context with the beginning of chunks
        matches = []
        max_match_length = 0

        for chunk_tuple, chunk_info in self.common_chunks.items():
            for match_length in range(min(len(context), len(chunk_tuple)), 0, -1):
                if context[-match_length:] == chunk_tuple[:match_length]:
                    if match_length > max_match_length:
                        max_match_length = match_length
                        matches = [(chunk_tuple, chunk_info, match_length)]
                    elif match_length == max_match_length:
                        matches.append((chunk_tuple, chunk_info, match_length))

        if not matches:
            return []

        # Generate predictions based on matched chunks
        predictions = {}
        total_weight = 0

        for chunk_tuple, chunk_info, match_length in matches:
            # If match is complete, this chunk can't help with prediction
            if match_length >= len(chunk_tuple):
                continue

            # The next element in the chunk is the prediction
            next_pos = chunk_tuple[match_length]

            # Weight by chunk cohesion and attention
            weight = chunk_info["cohesion"]
            if "attention" in chunk_info:
                weight *= chunk_info["attention"]

            if next_pos in predictions:
                predictions[next_pos] += weight
            else:
                predictions[next_pos] = weight

            total_weight += weight

        # Normalize predictions
        if total_weight > 0:
            return [(pos, weight / total_weight) for pos, weight in predictions.items()]
        else:
            return []

    def visualize_pos_graph_with_attention(self, graph, filename: str = "pos_graph_attention.png",
                                           title_prefix: str = ""):
        """
        Visualize the POS transition graph with attention weighting.

        Args:
            graph: The graph to visualize
            filename: Output file name
            title_prefix: Optional prefix for the plot title
        """
        # Check if graph is empty
        if len(graph) <= 2:  # Only START and END nodes
            print("POS graph is empty or contains only special nodes - no visualization created")
            return

        # Create a copy without special nodes for cleaner visualization
        g = graph.copy()

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

        # Node sizes based on attention weights
        node_sizes = []
        node_colors = []
        for node in g.nodes():
            attention = self.attention_weights.get(node, 1.0)
            node_sizes.append(400 + 300 * attention)  # Base size + attention effect

            # Color gradient from cool to warm based on attention
            # Low attention: blue (0.0), High attention: red (1.0)
            node_colors.append((min(1.0, attention / 2), 0.2, max(0.0, 1.0 - attention / 2)))

        # Draw nodes with size and color based on attention
        nx.draw_networkx_nodes(g, pos, node_size=node_sizes, node_color=node_colors)

        # Prepare edge attributes
        edge_width = []
        edge_color = []

        for source, target, data in g.edges(data=True):
            # Default to 0.1 if weight is missing or zero
            weight = data.get("weight", 0.1)
            if weight == 0:
                weight = 0.1

            # Adjust width by precision weighting
            precision = data.get("precision", self.base_precision)
            precision_factor = precision / self.base_precision

            edge_width.append(weight * 5 * precision_factor)

            # Edge color based on boundary probability
            edge_color.append(data.get("boundary_prob", 0.5))

        # Draw edges
        nx.draw_networkx_edges(
            g, pos, width=edge_width,
            edge_color=edge_color, edge_cmap=plt.cm.Reds,
            connectionstyle="arc3,rad=0.1"
        )

        # Add labels
        nx.draw_networkx_labels(g, pos, font_size=10)

        # Edge labels (probability + precision)
        edge_labels = {}
        for u, v, d in g.edges(data=True):
            weight = d.get("weight", 0.0)
            precision = d.get("precision", self.base_precision)
            edge_labels[(u, v)] = f"{weight:.2f}\n(p:{precision:.1f})"

        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8)

        # Add a color bar for boundary probabilities
        fig = plt.gcf()
        ax = plt.gca()
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Boundary Probability")

        title = "POS Transition Graph with Attention"
        if title_prefix:
            title = f"{title_prefix} {title}"
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename)
        print(f"POS graph visualization saved to {filename}")
        plt.close()

    def visualize_chunk_graph_with_attention(self, filename: str = "chunk_graph_attention.png"):
        """
        Visualize the chunk transition graph with attention weighting.

        Args:
            filename: Output file name
        """
        if not hasattr(self, 'chunk_graph') or len(self.chunk_graph) == 0:
            print("Chunk graph is empty - no visualization created")
            return

        if len(self.chunk_graph.edges()) == 0:
            print("Chunk graph has no edges - adding artificial edges for visualization")
            # Create some artificial edges just for visualization
            nodes = list(self.chunk_graph.nodes())
            if len(nodes) > 1:
                for i in range(len(nodes) - 1):
                    self.chunk_graph.add_edge(nodes[i], nodes[i + 1], weight=0.1, attention=1.0)

        # Set up the plot
        plt.figure(figsize=(14, 12))

        # Define node positions using spring layout
        pos = nx.spring_layout(self.chunk_graph, seed=42)

        # Draw nodes with size and color based on attention and cohesion
        node_sizes = []
        node_colors = []
        for node in self.chunk_graph.nodes():
            cohesion = self.chunk_graph.nodes[node].get("cohesion", 0.5)
            attention = self.chunk_graph.nodes[node].get("attention", 1.0)

            if cohesion <= 0:
                cohesion = 0.5  # Ensure minimum size

            # Size based on cohesion and attention
            node_sizes.append(cohesion * attention * 1000)

            # Color based on attention (from cool to warm)
            node_colors.append((min(1.0, attention / 2), 0.4, max(0.0, 1.0 - attention / 2)))

        nx.draw_networkx_nodes(
            self.chunk_graph, pos,
            node_size=node_sizes,
            node_color=node_colors
        )

        # Draw edges with width and color based on weight and attention
        if len(self.chunk_graph.edges()) > 0:
            edge_width = []
            edge_color = []

            for _, _, data in self.chunk_graph.edges(data=True):
                weight = data.get("weight", 0.1)
                attention = data.get("attention", 1.0)

                if weight <= 0:
                    weight = 0.1  # Ensure minimum width

                # Width affected by both weight and attention
                edge_width.append(weight * attention * 10)

                # Edge color based on attention
                edge_color.append(attention)

            nx.draw_networkx_edges(
                self.chunk_graph, pos, width=edge_width,
                edge_color=edge_color, edge_cmap=plt.cm.YlOrRd, alpha=0.7,
                connectionstyle="arc3,rad=0.1"
            )

        # Add labels
        nx.draw_networkx_labels(self.chunk_graph, pos, font_size=9)

        # Add a color bar for edge attention
        fig = plt.gcf()
        ax = plt.gca()
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Attention Weight")

        plt.title("Chunk Transition Graph with Attention")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Chunk graph visualization saved to {filename}")
        plt.close()

    def _report_attention_weights(self):
        """Report the top and bottom attention weights."""
        # Sort attention weights
        sorted_pos = sorted(self.attention_weights.items(), key=lambda x: x[1], reverse=True)

        # Report top attention weights
        print("\nTop 5 POS tags by attention:")
        for pos, weight in sorted_pos[:5]:
            print(f"  {pos}: {weight:.3f}")

        # Report bottom attention weights if we have enough
        if len(sorted_pos) > 5:
            print("\nBottom 5 POS tags by attention:")
            for pos, weight in sorted_pos[-5:]:
                print(f"  {pos}: {weight:.3f}")

        # Report chunk attention if available
        if self.chunk_attention_weights:
            sorted_chunks = sorted(self.chunk_attention_weights.items(),
                                   key=lambda x: x[1], reverse=True)

            print("\nTop 5 chunks by attention:")
            for chunk, weight in sorted_chunks[:min(5, len(sorted_chunks))]:
                print(f"  {chunk}: {weight:.3f}")

    def plot_learning_progress(self, filename="learning_progress.png"):
        """
        Plot the learning progress (surprisal history).

        Args:
            filename: Output file name
        """
        if not self.surprisal_history:
            print("No learning progress to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.surprisal_history, 'b-o')
        plt.title('Learning Progress - Average Surprisal per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Surprisal')
        plt.grid(True)
        plt.savefig(filename)
        print(f"Learning progress plot saved to {filename}")
        plt.close()