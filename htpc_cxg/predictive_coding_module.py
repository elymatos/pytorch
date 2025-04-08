"""
Predictive Coding Module for the Construction Grammar System.

This module implements the core predictive coding framework, generating
predictions at multiple levels and calculating prediction errors.
"""

import numpy as np
from collections import defaultdict

class PredictiveCodingModule:
    def __init__(self, construction_registry=None):
        """
        Initialize the predictive coding module.

        Args:
            construction_registry: Dictionary of constructions
        """
        self.construction_registry = construction_registry or {}
        self.prediction_models = {
            'pos_level': defaultdict(dict),  # POS transition probabilities
            'construction_level': defaultdict(dict),  # Construction transition probabilities
            'cross_level': defaultdict(dict)  # How POS patterns predict constructions and vice versa
        }
        self.prediction_errors = {
            'pos_level': [],
            'construction_level': [],
            'cross_level': []
        }
        self.learning_rate = 0.1  # Rate for updating prediction models

    def process_sequence(self, pos_sequence, identified_constructions, attention_weights=None):
        """
        Process a sequence using predictive coding.

        Args:
            pos_sequence: List of POS tags
            identified_constructions: Dictionary of identified constructions
            attention_weights: Dictionary of attention weights (optional)

        Returns:
            dict: Dictionary with predictions and prediction errors
        """
        # Generate predictions at multiple levels
        predictions = self.generate_predictions(pos_sequence, identified_constructions)

        # Calculate prediction errors
        prediction_errors = self.calculate_prediction_errors(
            predictions, pos_sequence, identified_constructions
        )

        # Update models based on prediction errors
        self.update_models(prediction_errors, pos_sequence, identified_constructions,
                          attention_weights)

        return {
            'predictions': predictions,
            'prediction_errors': prediction_errors
        }

    def generate_predictions(self, pos_sequence, identified_constructions):
        """
        Generate predictions at multiple levels.

        Args:
            pos_sequence: List of POS tags
            identified_constructions: Dictionary of identified constructions

        Returns:
            dict: Dictionary with predictions at different levels
        """
        # Current position in the sequence
        sequence_length = len(pos_sequence)

        # Predictions for each position in the sequence
        position_predictions = {}

        for pos in range(sequence_length):
            # Get context for this position
            context = self._get_context(pos_sequence, pos)

            # 1. Generate POS-level predictions
            pos_predictions = self._predict_next_pos(context)

            # 2. Generate construction-level predictions
            construction_predictions = self._predict_from_constructions(
                identified_constructions, pos
            )

            # 3. Generate predictions using hierarchical structure
            hierarchy_predictions = self._predict_using_hierarchy(
                pos_sequence, identified_constructions, pos
            )

            # 4. Combine predictions at this position
            combined = self._combine_position_predictions(
                pos_predictions, construction_predictions, hierarchy_predictions
            )

            position_predictions[pos] = combined

        # Global next POS prediction
        next_pos_predictions = self._predict_next_pos(
            {'prev_tags': pos_sequence[-3:] if len(pos_sequence) >= 3 else pos_sequence}
        )

        return {
            'next_pos': next_pos_predictions,
            'position_predictions': position_predictions
        }

    def _get_context(self, pos_sequence, position):
        """
        Get context around a position for prediction.

        Args:
            pos_sequence: List of POS tags
            position: Position in the sequence

        Returns:
            dict: Context information
        """
        # Get previous and next tags for context window
        prev_tags = []
        if position > 0:
            window_start = max(0, position - 3)
            prev_tags = pos_sequence[window_start:position]

        next_tags = []
        if position < len(pos_sequence) - 1:
            window_end = min(len(pos_sequence), position + 4)
            next_tags = pos_sequence[position+1:window_end]

        return {
            'position': position,
            'prev_tags': prev_tags,
            'next_tags': next_tags,
            'current_tag': pos_sequence[position] if position < len(pos_sequence) else None
        }

    def _predict_next_pos(self, context):
        """
        Predict the next POS tag based on context.

        Args:
            context: Context information

        Returns:
            dict: Prediction probabilities for POS tags
        """
        predictions = {}

        # Get previous tags for prediction
        prev_tags = context.get('prev_tags', [])

        if not prev_tags:
            return {}

        # Use n-gram style prediction
        for n in range(min(3, len(prev_tags)), 0, -1):
            ngram = tuple(prev_tags[-n:])

            if ngram in self.prediction_models['pos_level']:
                # Get predictions from this n-gram
                next_probs = self.prediction_models['pos_level'][ngram]

                # Weight by n-gram length (longer context gets higher weight)
                weight = n / 3.0

                for tag, prob in next_probs.items():
                    if tag in predictions:
                        predictions[tag] += weight * prob
                    else:
                        predictions[tag] = weight * prob

        # Normalize
        total = sum(predictions.values()) or 1.0
        return {tag: prob/total for tag, prob in predictions.items()}

    def _predict_from_constructions(self, constructions, position):
        """
        Generate predictions based on identified constructions.

        Args:
            constructions: Dictionary of identified constructions
            position: Current position in the sequence

        Returns:
            dict: Prediction probabilities for POS tags
        """
        predictions = {}

        if 'all' not in constructions:
            return predictions

        # Check if this position is within any construction
        for const in constructions['all']:
            if const['start'] <= position < const['end']:
                # We're inside a construction
                const_id = const['id']

                if const_id in self.construction_registry:
                    const_pattern = self.construction_registry[const_id]['pos_sequence']
                    relative_pos = position - const['start']

                    # If we have more positions in this construction, predict them
                    if relative_pos + 1 < len(const_pattern):
                        next_pos_in_construction = const_pattern[relative_pos + 1]
                        predictions[next_pos_in_construction] = predictions.get(
                            next_pos_in_construction, 0) + 1.0

                    # Add predictions based on functional equivalences
                    self._add_equivalent_construction_predictions(
                        const_id, relative_pos, predictions)

        # Normalize
        total = sum(predictions.values()) or 1.0
        return {tag: prob/total for tag, prob in predictions.items()}

    def _add_equivalent_construction_predictions(self, const_id, relative_pos, predictions):
        """
        Add predictions based on functionally equivalent constructions.

        Args:
            const_id: Current construction ID
            relative_pos: Relative position within the construction
            predictions: Dictionary to add predictions to
        """
        # Skip if we don't have construction registry (should be passed from main module)
        if not hasattr(self, 'construction_registry') or not hasattr(self, 'construction_categories'):
            return

        # Skip if this construction has no categories
        if const_id not in self.construction_categories:
            return

        # Get categories this construction belongs to
        categories = self.construction_categories[const_id]

        for category in categories:
            # Get all equivalent constructions in this category
            if category not in self.functional_equivalences:
                continue

            equivalent_constructions = self.functional_equivalences[category]

            for equiv_id in equivalent_constructions:
                if equiv_id == const_id:
                    continue  # Skip self

                if equiv_id in self.construction_registry:
                    # Use the equivalent construction to make predictions
                    equiv_pattern = self.construction_registry[equiv_id]['pos_sequence']

                    # If the equivalent has a position that corresponds
                    if relative_pos + 1 < len(equiv_pattern):
                        next_pos = equiv_pattern[relative_pos + 1]
                        # Add with slightly lower weight
                        predictions[next_pos] = predictions.get(next_pos, 0) + 0.8

    def _predict_using_hierarchy(self, pos_sequence, constructions, position):
        """
        Generate predictions using hierarchical construction relationships.

        Args:
            pos_sequence: List of POS tags
            constructions: Dictionary of identified constructions
            position: Current position in the sequence

        Returns:
            dict: Prediction probabilities for POS tags
        """
        predictions = {}

        if 'composite' not in constructions:
            return predictions

        # Check if we're at a component boundary in any composite construction
        for comp_const in constructions['composite']:
            comp_id = comp_const['id']

            if comp_id not in self.construction_registry:
                continue

            comp_info = self.construction_registry[comp_id]
            components = comp_info.get('components', [])

            if not components or len(components) < 2:
                continue

            # Get component constructions
            component_spans = []
            for comp_const_id in components:
                for const in constructions.get('all', []):
                    if const['id'] == comp_const_id:
                        component_spans.append((const['start'], const['end'], comp_const_id))

            # Sort spans by position
            component_spans.sort()

            # Check if we're at a boundary between components
            for i in range(len(component_spans) - 1):
                current = component_spans[i]
                next_comp = component_spans[i + 1]

                # If we're at the end of the current component
                if current[1] - 1 == position:
                    # Predict the first POS of the next component
                    next_comp_id = next_comp[2]

                    if next_comp_id in self.construction_registry:
                        next_comp_pattern = self.construction_registry[next_comp_id]['pos_sequence']

                        if next_comp_pattern:
                            first_pos = next_comp_pattern[0]
                            predictions[first_pos] = predictions.get(first_pos, 0) + 1.0

        # Normalize
        total = sum(predictions.values()) or 1.0
        return {tag: prob/total for tag, prob in predictions.items()}

    def _combine_position_predictions(self, pos_predictions, construction_predictions,
                                     hierarchy_predictions, weights=None):
        """
        Combine predictions from different levels for a position.

        Args:
            pos_predictions: POS-level predictions
            construction_predictions: Construction-level predictions
            hierarchy_predictions: Hierarchy-based predictions
            weights: Optional weights for combining predictions

        Returns:
            dict: Combined prediction probabilities
        """
        if weights is None:
            # Default weights
            weights = {
                'pos': 0.3,
                'construction': 0.5,
                'hierarchy': 0.2
            }

        combined = {}

        # Combine predictions with weights
        for tag, prob in pos_predictions.items():
            combined[tag] = weights['pos'] * prob

        for tag, prob in construction_predictions.items():
            if tag in combined:
                combined[tag] += weights['construction'] * prob
            else:
                combined[tag] = weights['construction'] * prob

        for tag, prob in hierarchy_predictions.items():
            if tag in combined:
                combined[tag] += weights['hierarchy'] * prob
            else:
                combined[tag] = weights['hierarchy'] * prob

        # Normalize
        total = sum(combined.values()) or 1.0
        return {tag: prob/total for tag, prob in combined.items()}

    def calculate_prediction_errors(self, predictions, pos_sequence, constructions):
        """
        Calculate prediction errors at different levels.

        Args:
            predictions: Dictionary with predictions
            pos_sequence: List of POS tags
            constructions: Dictionary of identified constructions

        Returns:
            dict: Dictionary with prediction errors
        """
        errors = {
            'pos_level': 0.0,
            'position_errors': {},
            'total_error': 0.0
        }

        # Calculate position-specific errors
        position_predictions = predictions.get('position_predictions', {})

        for pos in range(len(pos_sequence) - 1):  # Exclude last position
            # Get prediction for this position
            pos_preds = position_predictions.get(pos, {})

            # Actual next POS
            actual_next = pos_sequence[pos + 1]

            # Prediction error (1 - predicted probability)
            pred_prob = pos_preds.get(actual_next, 0.0)
            error = 1.0 - pred_prob

            errors['position_errors'][pos] = error

        # Calculate overall POS-level error
        errors['pos_level'] = sum(errors['position_errors'].values()) / max(1, len(errors['position_errors']))
        errors['total_error'] = errors['pos_level']  # For now, total is same as POS level

        return errors

    def update_models(self, prediction_errors, pos_sequence, constructions, attention_weights=None):
        """
        Update prediction models based on prediction errors.

        Args:
            prediction_errors: Dictionary with prediction errors
            pos_sequence: List of POS tags
            constructions: Dictionary of identified constructions
            attention_weights: Dictionary of attention weights (optional)
        """
        # Skip if sequence is too short
        if len(pos_sequence) <= 1:
            return

        # Update POS-level predictions
        self._update_pos_level_model(prediction_errors, pos_sequence, attention_weights)

        # Update construction-level predictions
        self._update_construction_level_model(prediction_errors, pos_sequence, constructions,
                                            attention_weights)

    def _update_pos_level_model(self, prediction_errors, pos_sequence, attention_weights=None):
        """
        Update POS-level prediction model.

        Args:
            prediction_errors: Dictionary with prediction errors
            pos_sequence: List of POS tags
            attention_weights: Dictionary of attention weights (optional)
        """
        position_errors = prediction_errors.get('position_errors', {})

        for pos in range(len(pos_sequence) - 1):
            # Get context for this position
            prev_tags = pos_sequence[max(0, pos-2):pos+1]
            actual_next = pos_sequence[pos + 1]

            # Get error for this position
            error = position_errors.get(pos, 0.0)

            # Attention-weighted learning rate
            effective_lr = self.learning_rate
            if attention_weights and 'integrated' in attention_weights:
                pos_attention = attention_weights['integrated'].get(pos, 0.5)
                # Higher attention means larger updates
                effective_lr *= (0.5 + pos_attention)

            # Update n-gram predictions
            for n in range(min(3, len(prev_tags)), 0, -1):
                ngram = tuple(prev_tags[-n:])

                # Initialize if needed
                if ngram not in self.prediction_models['pos_level']:
                    self.prediction_models['pos_level'][ngram] = {}

                # Update prediction for actual next tag
                current_prob = self.prediction_models['pos_level'][ngram].get(actual_next, 0.1)

                # Increase probability based on error
                # Higher error means larger adjustment
                adjusted_prob = current_prob + effective_lr * error

                # Ensure it stays within [0, 1]
                adjusted_prob = max(0.0, min(1.0, adjusted_prob))

                self.prediction_models['pos_level'][ngram][actual_next] = adjusted_prob

                # Decrease other probabilities to maintain sum close to 1
                total = sum(self.prediction_models['pos_level'][ngram].values())
                if total > 1.0:
                    scale = 1.0 / total
                    for tag in self.prediction_models['pos_level'][ngram]:
                        self.prediction_models['pos_level'][ngram][tag] *= scale

    def _update_construction_level_model(self, prediction_errors, pos_sequence,
                                        constructions, attention_weights=None):
        """
        Update construction-level prediction model.

        Args:
            prediction_errors: Dictionary with prediction errors
            pos_sequence: List of POS tags
            constructions: Dictionary of identified constructions
            attention_weights: Dictionary of attention weights (optional)
        """
        if 'all' not in constructions or len(constructions['all']) <= 1:
            return

        position_errors = prediction_errors.get('position_errors', {})

        # Update predictions for construction transitions
        sorted_constructions = sorted(constructions['all'], key=lambda x: x['start'])

        for i in range(len(sorted_constructions) - 1):
            current = sorted_constructions[i]
            next_const = sorted_constructions[i + 1]

            # Only consider adjacent constructions
            if current['end'] != next_const['start']:
                continue

            current_id = current['id']
            next_id = next_const['id']

            # Get the position at the boundary
            boundary_pos = current['end'] - 1

            # Get error at this position
            error = position_errors.get(boundary_pos, 0.0)

            # Attention-weighted learning rate
            effective_lr = self.learning_rate
            if attention_weights and 'construction' in attention_weights:
                current_attention = attention_weights['construction'].get(current_id, 0.5)
                effective_lr *= (0.5 + current_attention)

            # Update construction transition prediction
            if current_id not in self.prediction_models['construction_level']:
                self.prediction_models['construction_level'][current_id] = {}

            current_prob = self.prediction_models['construction_level'][current_id].get(next_id, 0.1)

            # Adjust probability based on error
            adjusted_prob = current_prob + effective_lr * error
            adjusted_prob = max(0.0, min(1.0, adjusted_prob))

            self.prediction_models['construction_level'][current_id][next_id] = adjusted_prob

            # Normalize to maintain sum close to 1
            total = sum(self.prediction_models['construction_level'][current_id].values())
            if total > 1.0:
                scale = 1.0 / total
                for const_id in self.prediction_models['construction_level'][current_id]:
                    self.prediction_models['construction_level'][current_id][const_id] *= scale

    def get_prediction_for_position(self, pos, predictions):
        """
        Get the prediction for a specific position.

        Args:
            pos: Position in the sequence
            predictions: Dictionary with predictions

        Returns:
            dict: Prediction probabilities for POS tags at this position
        """
        position_predictions = predictions.get('position_predictions', {})
        return position_predictions.get(pos, {})

    def get_total_prediction_error(self, prediction_errors):
        """
        Get the total prediction error.

        Args:
            prediction_errors: Dictionary with prediction errors

        Returns:
            float: Total prediction error
        """
        return prediction_errors.get('total_error', 0.0)