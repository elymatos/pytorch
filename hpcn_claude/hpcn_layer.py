import numpy as np
import typing
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class HPCNLayer:
    """
    Base class for Hierarchical Predictive Coding Network layers
    Implements core HPCN principles: prediction, error tracking, and hierarchical abstraction
    """
    layer_id: int
    dim: int
    learning_rate: float = 0.01

    def __post_init__(self):
        """
        Initialize layer components
        """
        # Use Xavier/Glorot initialization for weights
        self.weights = np.random.randn(self.dim, self.dim) * np.sqrt(2.0 / (2 * self.dim))
        self.bias = np.zeros(self.dim)
        self.prediction_error = np.zeros(self.dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer

        :param x: Input vector
        :return: Output vector
        """
        # Ensure x is a 2D array
        x = np.atleast_2d(x)

        # Compute layer output
        output = np.dot(x, self.weights) + self.bias

        # Apply non-linearity (ReLU)
        return np.maximum(output, 0)

    def compute_prediction_error(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute prediction error

        :param prediction: Predicted vector
        :param target: Target vector
        :return: Prediction error
        """
        # Ensure consistent shapes
        prediction = np.atleast_1d(prediction).flatten()
        target = np.atleast_1d(target).flatten()

        # Truncate to minimum length to avoid shape mismatches
        min_len = min(len(prediction), len(target))
        prediction = prediction[:min_len]
        target = target[:min_len]

        # Compute error
        self.prediction_error = target - prediction
        return self.prediction_error

    def update_weights(self, x: np.ndarray, error: np.ndarray):
        """
        Update weights using prediction error

        :param x: Input vector
        :param error: Prediction error
        """
        # Ensure x is 2D
        x = np.atleast_2d(x)

        # Ensure error is 1D
        error = np.atleast_1d(error).flatten()

        # Compute weight gradient
        weight_gradient = np.outer(x.flatten(), error)

        # Update weights and bias
        self.weights += self.learning_rate * weight_gradient.T
        self.bias += self.learning_rate * error


class POSSequenceHPCN:
    """
    Hierarchical Predictive Coding Network for POS Sequence Processing
    """

    def __init__(self, pos_sequences: List[List[str]], embedding_dim: int = 64):
        """
        Initialize HPCN for POS sequence processing

        :param pos_sequences: List of POS tag sequences
        :param embedding_dim: Dimension of embedding space
        """
        self.pos_sequences = pos_sequences
        self.embedding_dim = embedding_dim

        # Create embedding layer
        self.pos_embeddings = self._create_pos_embeddings()

        # Initialize HPCN layers
        self.layers = [
            HPCNLayer(layer_id=1, dim=embedding_dim),
            HPCNLayer(layer_id=2, dim=embedding_dim),
            HPCNLayer(layer_id=3, dim=embedding_dim)
        ]

    def _create_pos_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Create embedding vectors for unique POS tags

        :return: Dictionary of POS tag embeddings
        """
        # Collect unique POS tags
        unique_pos = set(tag for sequence in self.pos_sequences for tag in sequence)

        # Generate random embeddings
        return {
            pos: np.random.randn(self.embedding_dim)
            for pos in unique_pos
        }

    def process_sequence(self, pos_sequence: List[str]):
        """
        Process a single POS sequence through HPCN layers

        :param pos_sequence: List of POS tags
        """
        # Convert POS sequence to embeddings
        sequence_embeddings = [self.pos_embeddings[pos] for pos in pos_sequence]

        # Process through layers
        current_input = sequence_embeddings[0]

        for i, layer in enumerate(self.layers):
            # Predict next state
            prediction = layer.forward(current_input)

            # Determine actual input for error computation
            if i + 1 < len(sequence_embeddings):
                actual = sequence_embeddings[i + 1]
            else:
                actual = prediction  # For the last layer or sequence end

            # Compute prediction error
            prediction_error = layer.compute_prediction_error(prediction, actual)

            # Update layer weights
            layer.update_weights(current_input, prediction_error)

            # Update current input for next layer
            current_input = prediction

    def train(self, epochs: int = 10):
        """
        Train the HPCN on all POS sequences

        :param epochs: Number of training epochs
        """
        for _ in range(epochs):
            for sequence in self.pos_sequences:
                self.process_sequence(sequence)

    def predict_next_pos(self, context: List[str]) -> List[str]:
        """
        Predict possible next POS tags based on context

        :param context: Context POS sequence
        :return: Predicted next POS tags
        """
        # Convert context to embeddings
        context_embeddings = [self.pos_embeddings[pos] for pos in context]

        # Process through layers
        current_input = context_embeddings[-1]
        for layer in self.layers:
            prediction = layer.forward(current_input)
            current_input = prediction

        # Find nearest embedding in POS embedding space
        predictions = sorted(
            self.pos_embeddings.items(),
            key=lambda x: np.linalg.norm(x[1] - current_input)
        )

        # Return top k predictions
        return [pos for pos, _ in predictions[:3]]


# Example usage
def main():
    # Sample POS sequences
    pos_sequences = [
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

    # Initialize and train HPCN
    hpcn = POSSequenceHPCN(pos_sequences)
    hpcn.train(epochs=20)

    # Predict next POS tags
    #context = ['NOUN', 'VERB']
    context = ['DET']
    predictions = hpcn.predict_next_pos(context)
    print(f"Predicted next POS tags for context {context}: {predictions}")


if __name__ == '__main__':
    main()