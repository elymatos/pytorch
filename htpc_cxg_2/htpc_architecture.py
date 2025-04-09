"""
Hierarchical Temporal Predictive Coding (HTPC) Architecture for Construction Grammar

This module defines the overall architecture for a system that uses HTPC
to recognize and generalize constructions in a construction grammar framework.
"""

import numpy as np
from collections import defaultdict, Counter


class HTPCArchitecture:
    """
    Overall architecture for the HTPC Construction Grammar system.
    This class defines the layers, connections, and information flow.
    """

    def __init__(self, num_hierarchical_levels=3):
        """
        Initialize the HTPC Architecture.

        Args:
            num_hierarchical_levels: Number of hierarchical levels in the system
        """
        self.num_levels = num_hierarchical_levels
        self.levels = []

        # Initialize the hierarchical levels
        for i in range(num_hierarchical_levels):
            level_params = {
                'level_idx': i,
                'temporal_window': 2 ** i,  # Temporal window grows with level
                'precision': 1.0 / (i + 1),  # Precision decreases with level
                'learning_rate': 0.1 / (i + 1)  # Learning rate decreases with level
            }

            if i == 0:
                # Lowest level deals with individual POS tags
                self.levels.append(POSLevel(**level_params))
            elif i == num_hierarchical_levels - 1:
                # Highest level deals with abstract categories and functions
                self.levels.append(CategoryLevel(**level_params))
            else:
                # Middle levels deal with constructions of varying complexity
                self.levels.append(ConstructionLevel(**level_params))

    def process_sequence(self, pos_sequence):
        """
        Process a POS sequence through the hierarchical levels.

        Args:
            pos_sequence: List of POS tags

        Returns:
            dict: Processing results for each level
        """
        # Initialize result collection
        results = {
            'level_outputs': {},
            'predictions': {},
            'prediction_errors': {},
            'constructions': {},
            'generalizations': {}
        }

        # Bottom-up pass: Process inputs and generate predictions
        bottom_up_outputs = self._bottom_up_pass(pos_sequence)
        results['level_outputs'] = bottom_up_outputs

        # Top-down pass: Generate expectations and calculate errors
        top_down_predictions, prediction_errors = self._top_down_pass(bottom_up_outputs)
        results['predictions'] = top_down_predictions
        results['prediction_errors'] = prediction_errors

        # Update pass: Update models based on prediction errors
        self._update_pass(prediction_errors)

        # Collect constructions and generalizations
        for i, level in enumerate(self.levels):
            results['constructions'][i] = level.get_constructions()
            if hasattr(level, 'get_generalizations'):
                results['generalizations'][i] = level.get_generalizations()

        return results

    def _bottom_up_pass(self, pos_sequence):
        """
        Perform bottom-up processing through the hierarchy.

        Args:
            pos_sequence: Input POS sequence

        Returns:
            dict: Outputs for each level
        """
        outputs = {}
        current_input = pos_sequence

        print("Starting bottom-up pass with sequence:", pos_sequence)

        for i, level in enumerate(self.levels):
            # Each level processes input and generates output
            print(f"\nProcessing level {i}...")
            level_output = level.process_input(current_input)
            outputs[i] = level_output

            # Log what was found
            if i == 0:  # POS level
                patterns = level_output.get('patterns', [])
                print(f"  Level {i}: Found {len(patterns)} patterns")
                for p in patterns[:3]:  # Show first 3
                    print(
                        f"    Pattern: {p.get('pattern', 'unknown')}, Position: {p.get('start', '?')}-{p.get('end', '?')}")
            else:
                constructions = level_output.get('construction_instances', [])
                print(f"  Level {i}: Found {len(constructions)} constructions")
                for c in constructions[:3]:  # Show first 3
                    print(
                        f"    Construction ID: {c.get('id', 'unknown')}, Position: {c.get('start', '?')}-{c.get('end', '?')}")

            # Print representation
            repr_items = level_output.get('representation', [])
            print(f"  Level {i} representation: {len(repr_items)} items")

            # Output becomes input for the next level
            if i < len(self.levels) - 1:
                current_input = level_output['representation']
                print(f"  Passing {len(current_input)} items to next level")

        return outputs

    def _top_down_pass(self, bottom_up_outputs):
        """
        Perform top-down processing to generate predictions.

        Args:
            bottom_up_outputs: Outputs from the bottom-up pass

        Returns:
            tuple: (predictions, prediction_errors) for each level
        """
        predictions = {}
        prediction_errors = {}

        # Start from the top level
        for i in range(len(self.levels) - 1, -1, -1):
            level = self.levels[i]

            # Generate predictions
            if i == len(self.levels) - 1:
                # Top level generates predictions based only on its own output
                level_predictions = level.generate_predictions(bottom_up_outputs[i])
            else:
                # Lower levels incorporate top-down predictions
                level_predictions = level.generate_predictions(
                    bottom_up_outputs[i],
                    higher_level_predictions=predictions.get(i + 1, {})
                )

            predictions[i] = level_predictions

            # Calculate prediction errors
            prediction_errors[i] = level.calculate_prediction_error(
                bottom_up_outputs[i]['representation'],
                level_predictions
            )

        return predictions, prediction_errors

    def _update_pass(self, prediction_errors):
        """
        Update models based on prediction errors.

        Args:
            prediction_errors: Prediction errors from each level
        """
        for i, level in enumerate(self.levels):
            level.update_model(prediction_errors[i])

    def extract_generalizations(self):
        """
        Extract generalizations about functional equivalence from the system.

        Returns:
            dict: Generalizations at different levels
        """
        generalizations = {}

        for i, level in enumerate(self.levels):
            if hasattr(level, 'extract_generalizations'):
                generalizations[i] = level.extract_generalizations()

        return generalizations


class POSLevel:
    """
    Lowest level of the hierarchy dealing with individual POS tags.
    """

    def __init__(self, level_idx, temporal_window, precision, learning_rate):
        """
        Initialize the POS level.

        Args:
            level_idx: Index of this level in the hierarchy
            temporal_window: Size of temporal window
            precision: Precision for this level
            learning_rate: Learning rate for this level
        """
        self.level_idx = level_idx
        self.temporal_window = temporal_window
        self.precision = precision
        self.learning_rate = learning_rate

        # Transition probabilities for POS tags
        self.transitions = {}

        # Recognized patterns at this level
        self.patterns = {}

        # Prediction models
        self.prediction_model = {}

        # Counter for pattern IDs
        self.pattern_counter = 0

    def process_input(self, pos_sequence):
        """
        Process input POS sequence.

        Args:
            pos_sequence: List of POS tags

        Returns:
            dict: Processing output
        """
        # Update transition probabilities
        self._update_transitions(pos_sequence)

        # Detect patterns in the sequence
        patterns = self._detect_patterns(pos_sequence)

        # Create representation for this level
        representation = self._create_representation(pos_sequence, patterns)

        return {
            'representation': representation,
            'patterns': patterns
        }

    def _update_transitions(self, pos_sequence):
        """
        Update transition probabilities between POS tags.

        Args:
            pos_sequence: List of POS tags
        """
        # Simple n-gram counting
        for i in range(len(pos_sequence) - 1):
            current = pos_sequence[i]
            next_pos = pos_sequence[i + 1]

            if current not in self.transitions:
                self.transitions[current] = {}

            if next_pos not in self.transitions[current]:
                self.transitions[current][next_pos] = 0

            self.transitions[current][next_pos] += 1

    def _detect_patterns(self, pos_sequence):
        """
        Detect patterns in the POS sequence.

        Args:
            pos_sequence: List of POS tags

        Returns:
            list: Detected patterns
        """
        # For the POS level, patterns are simply n-grams
        patterns = []

        for n in range(1, min(4, len(pos_sequence) + 1)):  # 1 to 3-grams
            for i in range(len(pos_sequence) - n + 1):
                pattern = tuple(pos_sequence[i:i + n])
                pattern_id = f"pattern_{self.pattern_counter}"
                self.pattern_counter += 1

                # Track pattern frequency
                if pattern not in self.patterns:
                    self.patterns[pattern] = {
                        'count': 0,
                        'positions': [],
                        'pattern': pattern  # Store the actual pattern
                    }

                self.patterns[pattern]['count'] += 1
                self.patterns[pattern]['positions'].append(i)

                patterns.append({
                    'id': pattern_id,
                    'pattern': pattern,
                    'start': i,
                    'end': i + n
                })

        return patterns

    def _create_representation(self, pos_sequence, patterns):
        """
        Create a representation for this level.

        Args:
            pos_sequence: List of POS tags
            patterns: Detected patterns

        Returns:
            list: Representation for this level
        """
        # For the POS level, the representation is based on the most
        # significant patterns detected
        representation = []

        # Sort patterns by length (longer patterns are more significant)
        sorted_patterns = sorted(patterns, key=lambda x: len(x['pattern']), reverse=True)

        # Track positions that have been covered
        covered_positions = set()

        # Add patterns to representation
        for pattern_info in sorted_patterns:
            pattern = pattern_info['pattern']
            start = pattern_info['start']
            end = pattern_info['end']

            # Skip if this position is already covered
            if any(pos in covered_positions for pos in range(start, end)):
                continue

            # Add to representation
            representation.append({
                'type': 'pattern',
                'value': pattern,
                'start': start,
                'end': end
            })

            # Mark positions as covered
            covered_positions.update(range(start, end))

        # Add any uncovered POS tags
        for i, pos in enumerate(pos_sequence):
            if i not in covered_positions:
                representation.append({
                    'type': 'pos',
                    'value': pos,
                    'start': i,
                    'end': i + 1
                })
                covered_positions.add(i)

        # Sort by position
        representation.sort(key=lambda x: x['start'])

        return representation

    def generate_predictions(self, level_output, higher_level_predictions=None):
        """
        Generate predictions for the next elements.

        Args:
            level_output: Output from this level's processing
            higher_level_predictions: Predictions from higher level (optional)

        Returns:
            dict: Predictions for next elements
        """
        predictions = {}

        # Get representation
        representation = level_output['representation']

        # Base predictions on transitions
        for i, item in enumerate(representation):
            if item['type'] == 'pos':
                pos = item['value']

                if pos in self.transitions:
                    next_probs = {}
                    total = sum(self.transitions[pos].values())

                    for next_pos, count in self.transitions[pos].items():
                        next_probs[next_pos] = count / total

                    predictions[i] = {
                        'type': 'pos_transition',
                        'probabilities': next_probs
                    }

            elif item['type'] == 'pattern':
                pattern = item['value']

                if len(pattern) > 1:
                    # Predict continuation of the pattern
                    predictions[i] = {
                        'type': 'pattern_continuation',
                        'pattern': pattern,
                        'next': pattern[-1] if len(pattern) > 1 else None
                    }

        # Incorporate higher-level predictions if available
        if higher_level_predictions:
            # Implementation depends on the format of higher-level predictions
            pass

        return predictions

    def calculate_prediction_error(self, actual, predictions):
        """
        Calculate prediction error between actual and predicted values.

        Args:
            actual: Actual values (representation)
            predictions: Predicted values

        Returns:
            dict: Prediction errors
        """
        errors = {}

        for i, item in enumerate(actual):
            if i in predictions:
                prediction = predictions[i]

                if prediction['type'] == 'pos_transition' and i + 1 < len(actual):
                    next_item = actual[i + 1]

                    if next_item['type'] == 'pos':
                        next_pos = next_item['value']

                        # Error is 1 - probability of correct next POS
                        probs = prediction['probabilities']
                        error = 1.0 - probs.get(next_pos, 0.0)

                        errors[i] = error

                elif prediction['type'] == 'pattern_continuation':
                    # Error calculation for pattern continuation
                    pass

        # Average error
        if errors:
            errors['average'] = sum(errors.values()) / len(errors)
        else:
            errors['average'] = 0.0

        return errors

    def update_model(self, prediction_errors):
        """
        Update the model based on prediction errors.

        Args:
            prediction_errors: Dictionary of prediction errors
        """
        # Simple implementation - just uses the internal state that's
        # already updated during processing
        pass

    def get_constructions(self):
        """
        Get constructions recognized at this level.

        Returns:
            dict: Recognized constructions
        """
        # For POS level, constructions are frequent n-grams
        constructions = {}

        for pattern, info in self.patterns.items():
            if info['count'] > 1:  # Only include patterns seen multiple times
                # Create a consistent ID for the pattern
                pattern_id = f"pos_pattern_{'_'.join(str(p) for p in pattern)}"

                constructions[pattern_id] = {
                    'count': info['count'],
                    'positions': info['positions'],
                    'pattern': pattern
                }

        return constructions


class ConstructionLevel:
    """
    Middle level of the hierarchy dealing with constructions of varying complexity.
    """

    def __init__(self, level_idx, temporal_window, precision, learning_rate):
        """
        Initialize the Construction level.

        Args:
            level_idx: Index of this level in the hierarchy
            temporal_window: Size of temporal window
            precision: Precision for this level
            learning_rate: Learning rate for this level
        """
        self.level_idx = level_idx
        self.temporal_window = temporal_window
        self.precision = precision
        self.learning_rate = learning_rate

        # Constructions recognized at this level
        self.constructions = {}

        # Construction co-occurrence statistics
        self.construction_transitions = {}

        # Functional similarities between constructions
        self.functional_similarities = {}

        # Prediction models for constructions
        self.prediction_model = {}

    def process_input(self, input_representation):
        """
        Process input from the level below.

        Args:
            input_representation: Representation from lower level

        Returns:
            dict: Processing output
        """
        # Identify constructions in the input
        construction_instances = self._identify_constructions(input_representation)

        # Update construction statistics
        self._update_construction_stats(construction_instances)

        # Create representation for this level
        representation = self._create_representation(input_representation, construction_instances)

        return {
            'representation': representation,
            'construction_instances': construction_instances
        }

    def _identify_constructions(self, input_representation):
        """
        Identify constructions in the input representation.

        Args:
            input_representation: Representation from lower level

        Returns:
            list: Identified construction instances
        """
        construction_instances = []

        # Group items that form known constructions
        i = 0
        while i < len(input_representation):
            # Try to match a construction starting at this position
            matched = False

            for const_id, const_info in self.constructions.items():
                pattern = const_info['pattern']
                pattern_len = len(pattern)

                # Check if we can match this pattern
                if i + pattern_len <= len(input_representation):
                    sub_representation = input_representation[i:i + pattern_len]
                    if self._match_pattern(sub_representation, pattern):
                        construction_instances.append({
                            'id': const_id,
                            'start': i,
                            'end': i + pattern_len,
                            'items': sub_representation
                        })
                        i += pattern_len
                        matched = True
                        break

            if not matched:
                # No construction matched, move to next position
                i += 1

        # Look for new potential constructions
        new_constructions = self._discover_new_constructions(input_representation, construction_instances)
        construction_instances.extend(new_constructions)

        return construction_instances

    def _match_pattern(self, items, pattern):
        """
        Check if a sequence of items matches a construction pattern.

        Args:
            items: Sequence of items from input representation
            pattern: Construction pattern to match

        Returns:
            bool: True if pattern matches
        """
        if len(items) != len(pattern):
            return False

        for i, pattern_item in enumerate(pattern):
            item = items[i]

            # Extract item value based on item type
            item_value = None
            if isinstance(item, dict):
                if 'value' in item:
                    item_value = item['value']
                elif 'id' in item:
                    item_value = item['id']
                else:
                    item_value = str(item)
            else:
                item_value = str(item)

            # Extract pattern item value
            pattern_value = pattern_item

            # Compare values
            if item_value != pattern_value:
                return False

        return True

    def _discover_new_constructions(self, input_representation, existing_instances):
        """
        Discover potential new constructions.

        Args:
            input_representation: Input representation
            existing_instances: Already identified construction instances

        Returns:
            list: Newly discovered construction instances
        """
        new_constructions = []

        # Find sequences that appear multiple times
        # Track positions that have already been covered by existing constructions
        covered_positions = set()
        for instance in existing_instances:
            for pos in range(instance['start'], instance['end']):
                covered_positions.add(pos)

        # Look for repeating patterns of length 2-4
        for pattern_length in range(2, min(5, len(input_representation))):
            # Find all patterns of this length
            pattern_occurrences = {}

            for i in range(len(input_representation) - pattern_length + 1):
                # Skip if positions are already covered
                if any(pos in covered_positions for pos in range(i, i + pattern_length)):
                    continue

                # Extract the pattern
                pattern_items = []
                for j in range(pattern_length):
                    item = input_representation[i + j]
                    # Extract value based on item type
                    if isinstance(item, dict):
                        if 'value' in item:
                            pattern_items.append(item['value'])
                        elif 'id' in item:
                            pattern_items.append(item['id'])
                        else:
                            pattern_items.append(str(item))
                    else:
                        pattern_items.append(str(item))

                # Convert to tuple for hashing
                pattern = tuple(pattern_items)

                # Record this occurrence
                if pattern not in pattern_occurrences:
                    pattern_occurrences[pattern] = []
                pattern_occurrences[pattern].append(i)

            # Create new constructions for patterns that occur multiple times
            for pattern, positions in pattern_occurrences.items():
                if len(positions) >= 2:  # Pattern must occur at least twice
                    # Create a new construction ID
                    const_id = f"const_{len(self.constructions)}"

                    # Register the construction
                    self.constructions[const_id] = {
                        'pattern': pattern,
                        'count': len(positions),
                        'instances': []
                    }

                    # Create instances for each occurrence
                    for start_pos in positions:
                        # Get the actual items for this instance
                        items = input_representation[start_pos:start_pos + pattern_length]

                        new_constructions.append({
                            'id': const_id,
                            'start': start_pos,
                            'end': start_pos + pattern_length,
                            'items': items
                        })

                        # Mark these positions as covered
                        for pos in range(start_pos, start_pos + pattern_length):
                            covered_positions.add(pos)

        return new_constructions

    def _update_construction_stats(self, construction_instances):
        """
        Update construction statistics based on identified instances.

        Args:
            construction_instances: Identified construction instances
        """
        # Update frequency counts
        for instance in construction_instances:
            const_id = instance['id']

            if const_id not in self.constructions:
                self.constructions[const_id] = {
                    'pattern': [item['value'] for item in instance['items']],
                    'count': 0,
                    'instances': []
                }

            self.constructions[const_id]['count'] += 1
            self.constructions[const_id]['instances'].append(instance)

        # Update transition statistics
        sorted_instances = sorted(construction_instances, key=lambda x: x['start'])

        for i in range(len(sorted_instances) - 1):
            current = sorted_instances[i]['id']
            next_const = sorted_instances[i + 1]['id']

            if current not in self.construction_transitions:
                self.construction_transitions[current] = {}

            if next_const not in self.construction_transitions[current]:
                self.construction_transitions[current][next_const] = 0

            self.construction_transitions[current][next_const] += 1

        # Update functional similarity metrics
        self._update_functional_similarities(sorted_instances)

    def _update_functional_similarities(self, construction_instances):
        """
        Update functional similarity metrics between constructions.

        Args:
            construction_instances: Identified construction instances
        """
        # Track which constructions appear in similar contexts
        for i, instance in enumerate(construction_instances):
            const_id = instance['id']

            # Get context (constructions before and after)
            context = []
            for j, other in enumerate(construction_instances):
                if j != i:
                    if other['end'] == instance['start']:
                        context.append(('before', other['id']))
                    elif other['start'] == instance['end']:
                        context.append(('after', other['id']))

            # Update context statistics for this construction
            if const_id not in self.functional_similarities:
                self.functional_similarities[const_id] = {
                    'contexts': [],
                    'similar_constructions': {}
                }

            self.functional_similarities[const_id]['contexts'].append(context)

            # Check for constructions with similar contexts
            for other_id in self.functional_similarities:
                if other_id != const_id:
                    other_contexts = self.functional_similarities[other_id]['contexts']

                    # Calculate context similarity
                    similarity = self._calculate_context_similarity(context, other_contexts)

                    if similarity > 0:
                        if other_id not in self.functional_similarities[const_id]['similar_constructions']:
                            self.functional_similarities[const_id]['similar_constructions'][other_id] = 0

                        self.functional_similarities[const_id]['similar_constructions'][other_id] += similarity

    def _calculate_context_similarity(self, context, other_contexts):
        """
        Calculate similarity between contexts.

        Args:
            context: Context for current construction
            other_contexts: List of contexts for another construction

        Returns:
            float: Similarity score
        """
        # Simple implementation - count matching context elements
        similarity = 0

        for other_context in other_contexts:
            matches = sum(1 for item in context if item in other_context)
            if matches > 0:
                similarity += matches / max(len(context), len(other_context))

        return similarity if other_contexts else 0

    def _create_representation(self, input_representation, construction_instances):
        """
        Create a representation for this level.

        Args:
            input_representation: Input representation from lower level
            construction_instances: Identified construction instances

        Returns:
            list: Representation for this level
        """
        # Create a representation based on identified constructions
        representation = []

        # Sort construction instances by position
        sorted_instances = sorted(construction_instances, key=lambda x: x['start'])

        # Track positions that have been covered
        covered_positions = set()

        # Add constructions to representation
        for instance in sorted_instances:
            start = instance['start']
            end = instance['end']

            # Skip if positions already covered
            if any(pos in covered_positions for pos in range(start, end)):
                continue

            # Add to representation
            representation.append({
                'type': 'construction',
                'id': instance['id'],
                'start': start,
                'end': end
            })

            # Mark positions as covered
            covered_positions.update(range(start, end))

        # Add any uncovered input items
        for i, item in enumerate(input_representation):
            if i not in covered_positions:
                representation.append({
                    'type': 'item',
                    'value': item['value'],
                    'start': i,
                    'end': i + 1
                })
                covered_positions.add(i)

        # Sort by position
        representation.sort(key=lambda x: x['start'])

        return representation

    def generate_predictions(self, level_output, higher_level_predictions=None):
        """
        Generate predictions for next elements.

        Args:
            level_output: Output from this level's processing
            higher_level_predictions: Predictions from higher level (optional)

        Returns:
            dict: Predictions for next elements
        """
        predictions = {}

        # Get representation
        representation = level_output['representation']

        # Generate predictions based on construction transitions
        for i, item in enumerate(representation):
            if item['type'] == 'construction':
                const_id = item['id']

                if const_id in self.construction_transitions:
                    next_probs = {}
                    total = sum(self.construction_transitions[const_id].values())

                    for next_id, count in self.construction_transitions[const_id].items():
                        next_probs[next_id] = count / total

                    predictions[i] = {
                        'type': 'construction_transition',
                        'probabilities': next_probs
                    }

        # Incorporate predictions from functional similarities
        for i, item in enumerate(representation):
            if item['type'] == 'construction':
                const_id = item['id']

                if const_id in self.functional_similarities:
                    similar_constructions = self.functional_similarities[const_id]['similar_constructions']

                    # Get predictions from similar constructions
                    similar_predictions = {}

                    for similar_id, similarity in similar_constructions.items():
                        if similar_id in self.construction_transitions:
                            for next_id, count in self.construction_transitions[similar_id].items():
                                if next_id not in similar_predictions:
                                    similar_predictions[next_id] = 0

                                # Weight by similarity
                                similar_predictions[next_id] += count * similarity

                    # Add to predictions if not already present
                    if i not in predictions and similar_predictions:
                        total = sum(similar_predictions.values())
                        if total > 0:
                            predictions[i] = {
                                'type': 'similarity_based',
                                'probabilities': {k: v / total for k, v in similar_predictions.items()}
                            }

        # Incorporate higher-level predictions if available
        if higher_level_predictions:
            # Implementation depends on the format of higher-level predictions
            pass

        return predictions

    def calculate_prediction_error(self, actual, predictions):
        """
        Calculate prediction error between actual and predicted values.

        Args:
            actual: Actual values (representation)
            predictions: Predicted values

        Returns:
            dict: Prediction errors
        """
        errors = {}

        for i, item in enumerate(actual):
            if i in predictions:
                prediction = predictions[i]

                if i + 1 < len(actual):
                    next_item = actual[i + 1]

                    if next_item['type'] == 'construction':
                        next_id = next_item['id']

                        # Error depends on prediction type
                        if prediction['type'] in ['construction_transition', 'similarity_based']:
                            probs = prediction['probabilities']
                            error = 1.0 - probs.get(next_id, 0.0)
                            errors[i] = error

        # Average error
        if errors:
            errors['average'] = sum(errors.values()) / len(errors)
        else:
            errors['average'] = 0.0

        return errors

    def update_model(self, prediction_errors):
        """
        Update the model based on prediction errors.

        Args:
            prediction_errors: Dictionary of prediction errors
        """
        # Implementation would update internal models based on errors
        pass

    def get_constructions(self):
        """
        Get constructions recognized at this level.

        Returns:
            dict: Recognized constructions
        """
        return self.constructions

    def extract_generalizations(self):
        """
        Extract generalizations about functional equivalence.

        Returns:
            dict: Generalizations about functional equivalence
        """
        generalizations = {}

        # Group constructions by functional similarity
        for const_id, similarity_info in self.functional_similarities.items():
            similar_constructions = similarity_info['similar_constructions']

            # Only consider strong similarities
            strong_similarities = {other_id: sim for other_id, sim in similar_constructions.items()
                                   if sim >= 0.5}  # Threshold for similarity

            if strong_similarities:
                generalizations[const_id] = {
                    'functionally_similar': strong_similarities
                }

        return generalizations


class CategoryLevel:
    """
    Highest level of the hierarchy dealing with abstract categories and functions.
    """

    def __init__(self, level_idx, temporal_window, precision, learning_rate):
        """
        Initialize the Category level.

        Args:
            level_idx: Index of this level in the hierarchy
            temporal_window: Size of temporal window
            precision: Precision for this level
            learning_rate: Learning rate for this level
        """
        self.level_idx = level_idx
        self.temporal_window = temporal_window
        self.precision = precision
        self.learning_rate = learning_rate

        # Functional categories
        self.categories = {}

        # Construction-to-category mappings
        self.construction_categories = {}

        # Category transitions
        self.category_transitions = {}

        # Abstract templates with category slots
        self.templates = {}

    def process_input(self, input_representation):
        """
        Process input from the level below.

        Args:
            input_representation: Representation from lower level

        Returns:
            dict: Processing output
        """
        # Identify categories in the input
        category_instances = self._identify_categories(input_representation)

        # Update category statistics
        self._update_category_stats(category_instances)

        # Identify templates
        template_instances = self._identify_templates(input_representation, category_instances)

        # Create representation for this level
        representation = self._create_representation(input_representation,
                                                     category_instances,
                                                     template_instances)

        return {
            'representation': representation,
            'category_instances': category_instances,
            'template_instances': template_instances
        }

    def _identify_categories(self, input_representation):
        """
        Identify categories in the input representation.

        Args:
            input_representation: Representation from lower level

        Returns:
            list: Identified category instances
        """
        category_instances = []

        for i, item in enumerate(input_representation):
            if item['type'] == 'construction':
                const_id = item['id']

                if const_id in self.construction_categories:
                    categories = self.construction_categories[const_id]

                    for category in categories:
                        category_instances.append({
                            'category': category,
                            'construction_id': const_id,
                            'start': item['start'],
                            'end': item['end']
                        })

        # If no categories are found but we have constructions, create initial categories
        # But don't do recursive call to avoid infinite recursion
        if not category_instances and input_representation:
            # Create initial categories
            construction_items = [item for item in input_representation if item['type'] == 'construction']

            for item in construction_items:
                const_id = item['id']

                # Check if already categorized
                if const_id in self.construction_categories:
                    continue

                # Assign to a default category
                category = f"category_{len(self.categories)}"

                if category not in self.categories:
                    self.categories[category] = {
                        'count': 0,
                        'constructions': set(),
                        'instances': []
                    }

                self.categories[category]['constructions'].add(const_id)

                if const_id not in self.construction_categories:
                    self.construction_categories[const_id] = []

                self.construction_categories[const_id].append(category)

                # Create a category instance
                category_instances.append({
                    'category': category,
                    'construction_id': const_id,
                    'start': item['start'],
                    'end': item['end']
                })

        return category_instances

    def _create_initial_categories(self, input_representation):
        """
        Create initial categories based on input patterns.

        Args:
            input_representation: Input representation

        Returns:
            list: Category instances created
        """
        category_instances = []

        # Simple implementation to create initial categories
        for i, item in enumerate(input_representation):
            if item['type'] == 'construction':
                const_id = item['id']

                # Check if already categorized
                if const_id in self.construction_categories:
                    continue

                # Assign to a default category
                category = f"category_{len(self.categories)}"

                if category not in self.categories:
                    self.categories[category] = {
                        'count': 0,
                        'constructions': set(),
                        'instances': []
                    }

                self.categories[category]['constructions'].add(const_id)

                if const_id not in self.construction_categories:
                    self.construction_categories[const_id] = []

                self.construction_categories[const_id].append(category)

                # Create a category instance
                category_instances.append({
                    'category': category,
                    'construction_id': const_id,
                    'start': item['start'],
                    'end': item['end']
                })

        return category_instances

    def _update_category_stats(self, category_instances):
        """
        Update category statistics.

        Args:
            category_instances: Identified category instances
        """
        # Update category frequencies
        for instance in category_instances:
            category = instance['category']
            const_id = instance['construction_id']

            if category not in self.categories:
                self.categories[category] = {
                    'count': 0,
                    'constructions': set(),
                    'instances': []
                }

            self.categories[category]['count'] += 1
            self.categories[category]['constructions'].add(const_id)
            self.categories[category]['instances'].append(instance)

        # Update category transitions
        sorted_instances = sorted(category_instances, key=lambda x: x['start'])

        for i in range(len(sorted_instances) - 1):
            current = sorted_instances[i]['category']
            next_cat = sorted_instances[i + 1]['category']

            if current not in self.category_transitions:
                self.category_transitions[current] = {}

            if next_cat not in self.category_transitions[current]:
                self.category_transitions[current][next_cat] = 0

            self.category_transitions[current][next_cat] += 1

    def _identify_templates(self, input_representation, category_instances):
        """
        Identify template patterns based on category sequences.

        Args:
            input_representation: Input representation
            category_instances: Identified category instances

        Returns:
            list: Identified template instances
        """
        template_instances = []

        # If no category instances, return empty list
        if not category_instances:
            return template_instances

        # Sort category instances by position
        sorted_instances = sorted(category_instances, key=lambda x: x['start'])

        # Extract the sequence of categories
        category_sequence = [instance['category'] for instance in sorted_instances]

        # Check if this sequence matches any known template
        for template_id, template_info in self.templates.items():
            template_pattern = template_info['pattern']

            # Simple substring matching
            for i in range(len(category_sequence) - len(template_pattern) + 1):
                if category_sequence[i:i + len(template_pattern)] == template_pattern:
                    # Found a match
                    start_pos = sorted_instances[i]['start']
                    end_pos = sorted_instances[i + len(template_pattern) - 1]['end']

                    template_instances.append({
                        'template_id': template_id,
                        'start': start_pos,
                        'end': end_pos,
                        'category_indices': list(range(i, i + len(template_pattern)))
                    })

        # Look for potential new templates
        self._discover_new_templates(category_sequence)

        return template_instances

    def _discover_new_templates(self, category_sequence):
        """
        Discover potential new templates from category sequences.

        Args:
            category_sequence: Sequence of categories
        """
        # For simplicity, consider sequences of 2-4 categories as potential templates
        for length in range(2, min(5, len(category_sequence) + 1)):
            for i in range(len(category_sequence) - length + 1):
                candidate = tuple(category_sequence[i:i + length])

                # Check if we've seen this pattern before
                template_exists = False
                for template_id, template_info in self.templates.items():
                    if template_info['pattern'] == candidate:
                        template_exists = True
                        template_info['count'] += 1
                        break

                # If not, create a new template
                if not template_exists:
                    template_id = f"template_{len(self.templates)}"
                    self.templates[template_id] = {
                        'pattern': candidate,
                        'count': 1
                    }

    def _create_representation(self, input_representation, category_instances, template_instances):
        """
        Create a representation for this level.

        Args:
            input_representation: Input representation from lower level
            category_instances: Identified category instances
            template_instances: Identified template instances

        Returns:
            list: Representation for this level
        """
        representation = []

        # Add templates to representation
        for template in template_instances:
            representation.append({
                'type': 'template',
                'id': template['template_id'],
                'start': template['start'],
                'end': template['end']
            })

        # Add categories not part of templates
        template_covered = set()
        for template in template_instances:
            template_covered.update(template['category_indices'])

        sorted_categories = sorted(category_instances, key=lambda x: x['start'])
        for i, category in enumerate(sorted_categories):
            if i not in template_covered:
                representation.append({
                    'type': 'category',
                    'category': category['category'],
                    'start': category['start'],
                    'end': category['end']
                })

        # Sort by position
        representation.sort(key=lambda x: x['start'])

        return representation

    def generate_predictions(self, level_output, higher_level_predictions=None):
        """
        Generate predictions for next elements.

        Args:
            level_output: Output from this level's processing
            higher_level_predictions: Predictions from higher level (optional)

        Returns:
            dict: Predictions for next elements
        """
        predictions = {}

        # Get representation
        representation = level_output['representation']

        # Generate predictions based on template and category transitions
        for i, item in enumerate(representation):
            if item['type'] == 'template':
                template_id = item['id']

                # Template-based predictions would be implemented here
                # This is a placeholder
                predictions[i] = {
                    'type': 'template_based',
                    'template_id': template_id,
                    'probabilities': {}  # Would contain actual predictions
                }

            elif item['type'] == 'category':
                category = item['category']

                if category in self.category_transitions:
                    next_probs = {}
                    total = sum(self.category_transitions[category].values())

                    for next_cat, count in self.category_transitions[category].items():
                        next_probs[next_cat] = count / total

                    predictions[i] = {
                        'type': 'category_transition',
                        'probabilities': next_probs
                    }

        return predictions

    def calculate_prediction_error(self, actual, predictions):
        """
        Calculate prediction error between actual and predicted values.

        Args:
            actual: Actual values (representation)
            predictions: Predicted values

        Returns:
            dict: Prediction errors
        """
        errors = {}

        for i, item in enumerate(actual):
            if i in predictions:
                prediction = predictions[i]

                if i + 1 < len(actual):
                    next_item = actual[i + 1]

                    if next_item['type'] == 'category':
                        next_category = next_item['category']

                        if prediction['type'] == 'category_transition':
                            probs = prediction['probabilities']
                            error = 1.0 - probs.get(next_category, 0.0)
                            errors[i] = error

        # Average error
        if errors:
            errors['average'] = sum(errors.values()) / len(errors)
        else:
            errors['average'] = 0.0

        return errors

    def update_model(self, prediction_errors):
        """
        Update the model based on prediction errors.

        Args:
            prediction_errors: Dictionary of prediction errors
        """
        # Implementation would update internal models based on errors
        pass

    def get_constructions(self):
        """
        Get constructions recognized at this level.

        Returns:
            dict: Recognized constructions (templates in this case)
        """
        return self.templates

    def get_generalizations(self):
        """
        Get generalizations about functional categories.

        Returns:
            dict: Functional categories and their members
        """
        return self.categories

    def extract_generalizations(self):
        """
        Extract generalizations about functional equivalence.

        Returns:
            dict: Generalizations about functional equivalence
        """
        generalizations = {}

        # Group constructions by category
        for category, category_info in self.categories.items():
            if len(category_info['constructions']) > 1:
                generalizations[category] = {
                    'constructions': list(category_info['constructions']),
                    'count': category_info['count']
                }

        return generalizations