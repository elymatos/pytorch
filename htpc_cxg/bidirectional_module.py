"""
Bidirectional Module for the Predictive Coding Construction Grammar System.

This module handles bidirectional processing of POS sequences, combining
evidence from both forward and backward directions for improved accuracy.
"""

import numpy as np

class BidirectionalModule:
    def __init__(self):
        """
        Initialize the bidirectional module.
        """
        self.forward_weights = 0.5  # Default equal weighting
        self.backward_weights = 0.5
        self.direction_confidence = {'forward': 0.5, 'backward': 0.5}

    def process_bidirectional(self, pos_sequence, process_function, **kwargs):
        """
        Process a POS sequence in both directions.

        Args:
            pos_sequence: List of POS tags
            process_function: Function to process the sequence
            **kwargs: Additional arguments for the process function

        Returns:
            dict: Dictionary with combined results from both directions
        """
        # Forward pass
        forward_results = process_function(pos_sequence, **kwargs)

        # Backward pass
        reversed_sequence = list(reversed(pos_sequence))
        backward_results = process_function(reversed_sequence, **kwargs)

        # Adjust results from backward direction to match forward direction indices
        adjusted_backward = self._adjust_backward_results(backward_results, len(pos_sequence))

        # Combine results
        combined_results = self._combine_results(forward_results, adjusted_backward)

        return {
            'forward': forward_results,
            'backward': adjusted_backward,
            'combined': combined_results
        }

    def _adjust_backward_results(self, backward_results, sequence_length):
        """
        Adjust indices in backward results to match forward direction.

        Args:
            backward_results: Results from backward processing
            sequence_length: Length of the original sequence

        Returns:
            dict: Adjusted backward results
        """
        adjusted = {}

        # Handle different result types

        # For construction results
        if 'predefined' in backward_results:
            # Adjust construction indices
            adjusted_constructions = {}

            for key in ['predefined', 'new', 'composite', 'all']:
                if key in backward_results:
                    adjusted_constructions[key] = []

                    for const in backward_results[key]:
                        # Create a copy of the construction
                        adjusted_const = dict(const)

                        # Adjust start and end indices
                        adjusted_const['start'] = sequence_length - const['end']
                        adjusted_const['end'] = sequence_length - const['start']

                        adjusted_constructions[key].append(adjusted_const)

            # Add adjusted constructions to result
            adjusted.update(adjusted_constructions)

        # For attention results
        if 'pos' in backward_results:
            adjusted['pos'] = backward_results['pos']  # POS attention doesn't need adjustment

            # Adjust integrated attention positions
            if 'integrated' in backward_results:
                integrated = {}
                for pos, value in backward_results['integrated'].items():
                    # Only adjust numeric indices
                    if isinstance(pos, int):
                        integrated[sequence_length - 1 - pos] = value
                    else:
                        integrated[pos] = value

                adjusted['integrated'] = integrated

        # For prediction results
        if 'next_pos' in backward_results:
            # Convert "next_pos" in backward to "prev_pos" in forward
            adjusted['prev_pos'] = backward_results['next_pos']

            # Adjust position-specific predictions
            if 'position_predictions' in backward_results:
                position_predictions = {}
                for pos, preds in backward_results['position_predictions'].items():
                    position_predictions[sequence_length - 1 - pos] = preds

                adjusted['position_predictions'] = position_predictions

        # Add any other fields that don't need adjustment
        for key, value in backward_results.items():
            if key not in adjusted and key not in ['predefined', 'new', 'composite', 'all',
                                                 'pos', 'integrated', 'next_pos', 'position_predictions']:
                adjusted[key] = value

        return adjusted

    def _combine_results(self, forward_results, backward_results):
        """
        Combine results from forward and backward processing.

        Args:
            forward_results: Results from forward processing
            backward_results: Adjusted results from backward processing

        Returns:
            dict: Combined results
        """
        combined = {}

        # Combine construction results
        if 'predefined' in forward_results and 'predefined' in backward_results:
            combined['constructions'] = self._combine_constructions(
                forward_results,
                backward_results
            )
        elif 'constructions' in forward_results:
            combined['constructions'] = forward_results['constructions']

        # Combine attention results
        if 'pos' in forward_results and 'pos' in backward_results:
            combined['attention'] = self._combine_attention(
                forward_results,
                backward_results
            )
        elif 'attention' in forward_results:
            combined['attention'] = forward_results['attention']

        # Combine prediction results
        if 'next_pos' in forward_results and 'prev_pos' in backward_results:
            combined['predictions'] = self._combine_predictions(
                forward_results,
                backward_results
            )
        elif 'predictions' in forward_results:
            combined['predictions'] = forward_results['predictions']

        # Add prediction error if available
        if 'prediction_error' in forward_results:
            combined['prediction_error'] = forward_results['prediction_error']

        return combined

    def _combine_constructions(self, forward_results, backward_results):
        """
        Combine construction results from both directions.

        Args:
            forward_results: Construction results from forward direction
            backward_results: Construction results from backward direction

        Returns:
            dict: Combined construction results
        """
        # Start with forward constructions
        combined = {
            'all': [],
            'predefined': forward_results.get('predefined', []).copy(),
            'new': forward_results.get('new', []).copy(),
            'composite': forward_results.get('composite', []).copy()
        }

        # Add backward-only constructions (those not overlapping with forward ones)
        forward_spans = [(c['start'], c['end']) for c in forward_results.get('all', [])]

        for const_type in ['predefined', 'new', 'composite']:
            for const in backward_results.get(const_type, []):
                span = (const['start'], const['end'])

                # Check if this span overlaps with any forward span
                if not any(max(span[0], f_span[0]) < min(span[1], f_span[1])
                          for f_span in forward_spans):
                    # No overlap, add this construction
                    combined[const_type].append(const)
                    combined['all'].append(const)

        # Ensure 'all' has all constructions
        if not combined['all']:
            combined['all'] = (combined['predefined'] +
                              combined['new'] +
                              combined['composite'])

        return combined

    def _combine_attention(self, forward_results, backward_results):
        """
        Combine attention results from both directions.

        Args:
            forward_results: Attention results from forward direction
            backward_results: Attention results from backward direction

        Returns:
            dict: Combined attention results
        """
        combined = {}

        # Combine POS attention
        if 'pos' in forward_results and 'pos' in backward_results:
            pos_combined = {}
            all_pos = set(list(forward_results['pos'].keys()) + list(backward_results['pos'].keys()))

            for pos in all_pos:
                fw_value = forward_results['pos'].get(pos, 0.5)
                bw_value = backward_results['pos'].get(pos, 0.5)

                # Weighted combination
                pos_combined[pos] = (self.forward_weights * fw_value +
                                    self.backward_weights * bw_value)

            combined['pos'] = pos_combined

        # Combine integrated attention
        if 'integrated' in forward_results and 'integrated' in backward_results:
            integrated_combined = {}
            all_positions = set(list(forward_results['integrated'].keys()) +
                               list(backward_results['integrated'].keys()))

            for pos in all_positions:
                if isinstance(pos, int):  # Only combine position-based attention
                    fw_value = forward_results['integrated'].get(pos, 0.5)
                    bw_value = backward_results['integrated'].get(pos, 0.5)

                    # Weighted combination
                    integrated_combined[pos] = (self.forward_weights * fw_value +
                                               self.backward_weights * bw_value)

            combined['integrated'] = integrated_combined

        return combined

    def _combine_predictions(self, forward_results, backward_results):
        """
        Combine prediction results from both directions.

        Args:
            forward_results: Prediction results from forward direction
            backward_results: Prediction results from backward direction

        Returns:
            dict: Combined prediction results
        """
        combined = {}

        # Combine next and previous predictions
        next_pos = {}
        if 'next_pos' in forward_results:
            next_pos = dict(forward_results['next_pos'])

        prev_pos = {}
        if 'prev_pos' in backward_results:
            prev_pos = dict(backward_results['prev_pos'])

        combined['next_pos'] = next_pos
        combined['prev_pos'] = prev_pos

        # Combine position-specific predictions
        if ('position_predictions' in forward_results and
            'position_predictions' in backward_results):
            position_combined = {}
            all_positions = set(list(forward_results['position_predictions'].keys()) +
                              list(backward_results['position_predictions'].keys()))

            for pos in all_positions:
                fw_preds = forward_results['position_predictions'].get(pos, {})
                bw_preds = backward_results['position_predictions'].get(pos, {})

                # Combine predictions
                pos_combined = {}
                all_tags = set(list(fw_preds.keys()) + list(bw_preds.keys()))

                for tag in all_tags:
                    fw_value = fw_preds.get(tag, 0.0)
                    bw_value = bw_preds.get(tag, 0.0)

                    # Weighted combination
                    pos_combined[tag] = (self.forward_weights * fw_value +
                                        self.backward_weights * bw_value)

                # Normalize
                total = sum(pos_combined.values()) or 1.0
                pos_combined = {tag: val/total for tag, val in pos_combined.items()}

                position_combined[pos] = pos_combined

            combined['position_predictions'] = position_combined

        return combined

    def update_direction_weights(self, forward_error, backward_error):
        """
        Update weights for combining forward and backward results based on prediction errors.

        Args:
            forward_error: Error in forward direction predictions
            backward_error: Error in backward direction predictions
        """
        if forward_error == 0 and backward_error == 0:
            # Equal weighting if both are perfect
            self.forward_weights = 0.5
            self.backward_weights = 0.5
            return

        # Invert errors to get confidence
        forward_conf = 1.0 / (1.0 + forward_error)
        backward_conf = 1.0 / (1.0 + backward_error)

        # Update direction confidence
        self.direction_confidence['forward'] = forward_conf
        self.direction_confidence['backward'] = backward_conf

        # Normalize to get weights
        total_conf = forward_conf + backward_conf
        self.forward_weights = forward_conf / total_conf
        self.backward_weights = backward_conf / total_conf

    def get_direction_weights(self):
        """
        Get the current weights for combining directions.

        Returns:
            tuple: (forward_weight, backward_weight)
        """
        return (self.forward_weights, self.backward_weights)