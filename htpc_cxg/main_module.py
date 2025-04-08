"""
Main Module for the Predictive Coding Construction Grammar System.

This module integrates all components through multiple inheritance
and provides the main interface for using the system.
"""

from base_module import BaseModule
from construction_module import ConstructionModule
from attention_module import AttentionModule
from bidirectional_module import BidirectionalModule
from predictive_coding_module import PredictiveCodingModule


class MainModule(BaseModule, ConstructionModule, AttentionModule,
                 BidirectionalModule, PredictiveCodingModule):
    def __init__(self, predefined_constructions=None, min_chunk_size=1):
        """
        Initialize the main module with all components.

        Args:
            predefined_constructions: List of predefined constructions
            min_chunk_size: Minimum size of a construction
        """
        # Initialize all parent classes
        BaseModule.__init__(self)
        ConstructionModule.__init__(self, min_chunk_size, predefined_constructions)
        AttentionModule.__init__(self, self.construction_registry)
        BidirectionalModule.__init__(self)
        PredictiveCodingModule.__init__(self, self.construction_registry)

        # Additional state for the main module
        self.processing_history = []

    def process_sequence(self, pos_sequence, bidirectional=True):
        """
        Process a POS sequence with the full system.

        Args:
            pos_sequence: List of POS tags
            bidirectional: Whether to use bidirectional processing

        Returns:
            dict: Results of processing the sequence
        """
        if not pos_sequence:
            return {'error': 'Empty sequence'}

        # Update transition probabilities
        BaseModule.process_sequence(self, pos_sequence)

        if bidirectional:
            # Process in both directions using the bidirectional module
            results = self.process_bidirectional_sequence(pos_sequence)
        else:
            # Process only in forward direction
            results = self.process_forward_sequence(pos_sequence)

        # Add to processing history
        self.processing_history.append({
            'sequence': pos_sequence,
            'results': results
        })

        return results

    def process_bidirectional_sequence(self, pos_sequence):
        """
        Process a sequence in both directions.

        Args:
            pos_sequence: List of POS tags

        Returns:
            dict: Combined results from both directions
        """
        # Use the bidirectional module to process in both directions
        results = self.process_bidirectional(pos_sequence, self.process_forward_sequence)

        # Update direction weights based on prediction errors
        if 'forward' in results and 'backward' in results:
            forward_error = results['forward'].get('prediction_error', {}).get('total_error', 0.0)
            backward_error = results['backward'].get('prediction_error', {}).get('total_error', 0.0)

            self.update_direction_weights(forward_error, backward_error)

        return results

    def process_forward_sequence(self, pos_sequence):
        """
        Process a sequence in the forward direction.

        Args:
            pos_sequence: List of POS tags

        Returns:
            dict: Results of processing the sequence
        """
        # 1. Identify constructions
        constructions = self.identify_constructions(pos_sequence)

        # 2. Calculate attention
        attention = self.calculate_attention(
            pos_sequence,
            constructions,
            self.transition_counts
        )

        # 3. Generate predictions and calculate errors
        prediction_results = self.process_sequence_with_predictive_coding(
            pos_sequence,
            constructions,
            attention
        )

        # 4. Combine results
        results = {
            'constructions': constructions,
            'attention': attention,
            'predictions': prediction_results['predictions'],
            'prediction_error': prediction_results['prediction_errors']
        }

        return results

    def process_sequence_with_predictive_coding(self, pos_sequence, constructions, attention):
        """
        Process a sequence using the predictive coding module.

        Args:
            pos_sequence: List of POS tags
            constructions: Dictionary of identified constructions
            attention: Dictionary of attention weights

        Returns:
            dict: Results of predictive coding processing
        """
        return PredictiveCodingModule.process_sequence(
            self, pos_sequence, constructions, attention
        )

    def add_predefined_construction(self, pos_sequence):
        """
        Add a new predefined construction.

        Args:
            pos_sequence: List or tuple of POS tags

        Returns:
            str: ID of the new construction
        """
        # Generate a new predefined construction ID
        const_id = f"pre_{len(self.construction_registry)}"

        # Register the construction
        self.construction_registry[const_id] = {
            'pos_sequence': tuple(pos_sequence),
            'predefined': True,
            'frequency': 0,
            'confidence': 1.0,
            'entropy': 0.0,
            'cohesion': 0.0
        }

        return const_id

    def get_most_frequent_constructions(self, n=10):
        """
        Get the most frequent constructions.

        Args:
            n: Number of constructions to return

        Returns:
            list: List of (construction_id, frequency) pairs
        """
        # Sort constructions by frequency
        sorted_constructions = sorted(
            self.construction_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_constructions[:n]

    def get_construction_details(self, const_id):
        """
        Get detailed information about a construction.

        Args:
            const_id: Construction ID

        Returns:
            dict: Construction details
        """
        if const_id not in self.construction_registry:
            return {'error': 'Construction not found'}

        const_info = self.construction_registry[const_id]

        # Add additional information
        details = dict(const_info)

        # Add component constructions if this is a composite
        if const_info.get('composite', False):
            components = self.get_component_constructions(const_id)
            component_details = []

            for comp_id in components:
                if comp_id in self.construction_registry:
                    comp_details = {
                        'id': comp_id,
                        'pos_sequence': self.construction_registry[comp_id]['pos_sequence'],
                        'predefined': self.construction_registry[comp_id].get('predefined', False)
                    }
                    component_details.append(comp_details)

            details['component_details'] = component_details

        # Add specialization relations
        if const_id in self.specialization_relations:
            details['specializations'] = self.specialization_relations[const_id]

        # Add function annotation if available
        if 'function' not in details:
            function = self._annotate_construction_function(const_id)
            details['function'] = function

        return details

    def predict_for_partial_sequence(self, partial_sequence):
        """
        Generate predictions for a partial sequence.

        Args:
            partial_sequence: List of POS tags (partial sequence)

        Returns:
            dict: Prediction results
        """
        # Process the partial sequence
        results = self.process_sequence(partial_sequence)

        # Get the next tag predictions
        predictions = results.get('combined', {}).get('predictions', {}).get('next_pos', {})

        # Sort by probability
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        return {
            'next_pos_predictions': sorted_predictions,
            'constructions': results.get('combined', {}).get('constructions', {})
        }

    def reset(self):
        """
        Reset the system state.

        Returns:
            bool: True if reset was successful
        """
        # Reset base module
        self.forward_transitions.clear()
        self.backward_transitions.clear()
        self.pos_frequencies.clear()
        self.total_pos_count = 0
        self.transition_counts.clear()
        self.total_transitions = 0

        # Reset construction module (but keep predefined constructions)
        predefined = {}
        for const_id, const_info in self.construction_registry.items():
            if const_info.get('predefined', False):
                predefined[const_id] = const_info

        self.construction_registry = predefined
        self.hierarchical_relations.clear()
        self.specialization_relations.clear()
        self.construction_frequencies.clear()
        self.construction_transitions.clear()

        # Reset attention module
        self.pos_attention.clear()
        self.construction_attention.clear()
        self.cross_level_attention.clear()

        # Reset predictive coding module
        self.prediction_models['pos_level'].clear()
        self.prediction_models['construction_level'].clear()
        self.prediction_models['cross_level'].clear()

        self.prediction_errors['pos_level'] = []
        self.prediction_errors['construction_level'] = []
        self.prediction_errors['cross_level'] = []

        # Reset bidirectional module
        self.forward_weights = 0.5
        self.backward_weights = 0.5
        self.direction_confidence = {'forward': 0.5, 'backward': 0.5}

        # Reset processing history
        self.processing_history = []

        return True