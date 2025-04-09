"""
Hierarchical Temporal Predictive Coding (HTPC) Learning Module

This module implements the learning mechanisms for HTPC,
including the inference of functional equivalence between constructions.
"""

import numpy as np
from collections import defaultdict, Counter


class HTPCLearningModule:
    """
    Learning module for the HTPC system.
    This implements the mechanisms for learning from prediction errors
    and inferring generalizations.
    """

    def __init__(self, architecture):
        """
        Initialize the learning module.

        Args:
            architecture: HTPCArchitecture instance
        """
        self.architecture = architecture

        # Learning parameters
        self.error_history = []
        self.generalization_threshold = 0.7
        self.min_observations = 5

        # Tracking co-occurrence patterns
        self.context_observations = defaultdict(list)
        self.substitution_patterns = defaultdict(Counter)

        # Tracking generalizations
        self.inferred_equivalences = {}
        self.equivalence_confidence = {}

    def update_from_errors(self, prediction_errors):
        """
        Update learning based on prediction errors.

        Args:
            prediction_errors: Prediction errors from processing

        Returns:
            dict: Learning updates
        """
        # Store error history
        self.error_history.append(prediction_errors)

        # Apply updates to each level
        updates = {}

        for level, errors in prediction_errors.items():
            level_updates = self._update_level(level, errors)
            updates[level] = level_updates

        return updates

    def _update_level(self, level_idx, errors):
        """
        Update a specific level based on its prediction errors.

        Args:
            level_idx: Index of the level
            errors: Prediction errors for this level

        Returns:
            dict: Learning updates for this level
        """
        level = self.architecture.levels[level_idx]

        # Simple learning rate adjustment based on error trends
        if len(self.error_history) > 10:
            recent_errors = [history[level_idx].get('average', 0)
                             for history in self.error_history[-10:]]

            error_trend = recent_errors[-1] - sum(recent_errors[:-1]) / (len(recent_errors) - 1)

            if error_trend < 0:
                # Errors decreasing, reduce learning rate
                level.learning_rate *= 0.99
            else:
                # Errors increasing or stable, increase learning rate
                level.learning_rate *= 1.01
                level.learning_rate = min(level.learning_rate, 0.1)  # Cap learning rate

        return {
            'learning_rate': level.learning_rate
        }

    def observe_context_patterns(self, results):
        """
        Observe and record patterns in the data for future generalization.

        Args:
            results: Processing results from the architecture
        """
        # Extract constructions from the middle level (Construction level)
        constructions = results['constructions'].get(1, {})

        # Analyze contexts for constructions
        for const_id, const_info in constructions.items():
            instances = const_info.get('instances', [])

            for instance in instances:
                # Get context (what comes before and after)
                context = self._extract_context(instance, results)

                # Store this observation
                self.context_observations[const_id].append(context)

    def _extract_context(self, construction_instance, results):
        """
        Extract context information for a construction instance.

        Args:
            construction_instance: Instance of a construction
            results: Processing results

        Returns:
            dict: Context information
        """
        # This implementation would extract information about what precedes
        # and follows a construction, positions where it appears, etc.

        # Placeholder implementation
        return {
            'preceding': [],  # Would contain preceding constructions
            'following': [],  # Would contain following constructions
            'position': construction_instance.get('start', 0)
        }

    def observe_substitution_patterns(self, sequence_results):
        """
        Observe substitution patterns across different sequences.

        Args:
            sequence_results: Results from processing a sequence
        """
        # Extract patterns from the higher level (Category level)
        categories = sequence_results['generalizations'].get(2, {})

        # For each category, observe which constructions can substitute for each other
        for category, category_info in categories.items():
            constructions = category_info.get('constructions', [])

            if len(constructions) > 1:
                # For each pair of constructions in this category
                for i, const1 in enumerate(constructions):
                    for const2 in constructions[i + 1:]:
                        # Record this substitution pattern
                        self.substitution_patterns[const1][const2] += 1
                        self.substitution_patterns[const2][const1] += 1

    def analyze_similarity_patterns(self):
        """
        Analyze patterns of similarity between constructions.

        Returns:
            dict: Similarity matrices between constructions
        """
        # Placeholder for a more sophisticated similarity analysis
        similarity_matrices = {
            'context_similarity': self._calculate_context_similarity(),
            'substitution_similarity': self._calculate_substitution_similarity()
        }

        return similarity_matrices

    def _calculate_context_similarity(self):
        """
        Calculate similarity between constructions based on their contexts.

        Returns:
            dict: Context similarity matrix
        """
        # Placeholder implementation
        similarity_matrix = {}

        # For each pair of constructions
        for const1, contexts1 in self.context_observations.items():
            similarity_matrix[const1] = {}

            for const2, contexts2 in self.context_observations.items():
                if const1 != const2 and contexts1 and contexts2:
                    # Calculate Jaccard similarity of contexts
                    similarity = self._jaccard_similarity(contexts1, contexts2)
                    similarity_matrix[const1][const2] = similarity

        return similarity_matrix

    def _jaccard_similarity(self, contexts1, contexts2):
        """
        Calculate Jaccard similarity between context sets.

        Args:
            contexts1: First set of contexts
            contexts2: Second set of contexts

        Returns:
            float: Jaccard similarity
        """
        # Extract preceding and following elements as sets
        preceding1 = set()
        following1 = set()
        for context in contexts1:
            preceding1.update(context.get('preceding', []))
            following1.update(context.get('following', []))

        preceding2 = set()
        following2 = set()
        for context in contexts2:
            preceding2.update(context.get('preceding', []))
            following2.update(context.get('following', []))

        # Calculate Jaccard similarities
        if preceding1 or preceding2:
            preceding_similarity = len(preceding1.intersection(preceding2)) / len(preceding1.union(preceding2))
        else:
            preceding_similarity = 0

        if following1 or following2:
            following_similarity = len(following1.intersection(following2)) / len(following1.union(following2))
        else:
            following_similarity = 0

        # Overall similarity is average of preceding and following similarities
        return (preceding_similarity + following_similarity) / 2

    def _calculate_substitution_similarity(self):
        """
        Calculate similarity between constructions based on substitution patterns.

        Returns:
            dict: Substitution similarity matrix
        """
        similarity_matrix = {}

        # For each pair of constructions
        all_constructions = set(self.substitution_patterns.keys())

        for const1 in all_constructions:
            similarity_matrix[const1] = {}

            for const2 in all_constructions:
                if const1 != const2:
                    # Calculate similarity based on shared substitution partners
                    partners1 = set(self.substitution_patterns[const1].keys())
                    partners2 = set(self.substitution_patterns[const2].keys())

                    # Also consider if they substitute for each other
                    mutual_substitution = self.substitution_patterns[const1].get(const2, 0) + \
                                          self.substitution_patterns[const2].get(const1, 0)

                    if partners1 or partners2:
                        partner_similarity = len(partners1.intersection(partners2)) / len(partners1.union(partners2))
                    else:
                        partner_similarity = 0

                    # Weight mutual substitution more heavily
                    if mutual_substitution > 0:
                        similarity = 0.7 * (mutual_substitution / (mutual_substitution + 5)) + 0.3 * partner_similarity
                    else:
                        similarity = partner_similarity

                    similarity_matrix[const1][const2] = similarity

        return similarity_matrix

    def infer_functional_equivalence(self):
        """
        Infer functional equivalence between constructions.

        Returns:
            dict: Inferred functional equivalence classes
        """
        # Calculate similarity matrices
        similarity_matrices = self.analyze_similarity_patterns()

        # Combine different similarity measures
        combined_similarity = self._combine_similarity_matrices(similarity_matrices)

        # Cluster constructions into equivalence classes
        equivalence_classes = self._cluster_by_similarity(combined_similarity)

        # Store these for future use
        self.inferred_equivalences = equivalence_classes

        return equivalence_classes

    def _combine_similarity_matrices(self, similarity_matrices):
        """
        Combine multiple similarity matrices into one.

        Args:
            similarity_matrices: Dictionary of similarity matrices

        Returns:
            dict: Combined similarity matrix
        """
        # Simple weighted average of different similarity measures
        combined = {}

        # Get all constructions
        all_constructions = set()
        for matrix in similarity_matrices.values():
            all_constructions.update(matrix.keys())

        # Combine similarities
        for const1 in all_constructions:
            combined[const1] = {}

            for const2 in all_constructions:
                if const1 != const2:
                    # Calculate weighted average
                    similarities = []
                    weights = []

                    for matrix_type, matrix in similarity_matrices.items():
                        if const1 in matrix and const2 in matrix.get(const1, {}):
                            if matrix_type == 'context_similarity':
                                weight = 0.5
                            elif matrix_type == 'substitution_similarity':
                                weight = 0.8
                            else:
                                weight = 0.3

                            similarities.append(matrix[const1][const2])
                            weights.append(weight)

                    if similarities:
                        weighted_sum = sum(s * w for s, w in zip(similarities, weights))
                        total_weight = sum(weights)
                        combined[const1][const2] = weighted_sum / total_weight
                    else:
                        combined[const1][const2] = 0

        return combined

    def _cluster_by_similarity(self, similarity_matrix):
        """
        Cluster constructions into equivalence classes based on similarity.

        Args:
            similarity_matrix: Similarity matrix between constructions

        Returns:
            dict: Equivalence classes
        """
        # Simple threshold-based clustering
        equivalence_classes = {}
        assigned = set()

        # Sort construction pairs by similarity (descending)
        pairs = []
        for const1, similarities in similarity_matrix.items():
            for const2, sim in similarities.items():
                if sim >= self.generalization_threshold:
                    pairs.append((const1, const2, sim))

        pairs.sort(key=lambda x: x[2], reverse=True)

        # Create clusters
        class_idx = 0

        for const1, const2, similarity in pairs:
            # Skip if both are already assigned to the same class
            const1_class = None
            const2_class = None

            for cls, members in equivalence_classes.items():
                if const1 in members:
                    const1_class = cls
                if const2 in members:
                    const2_class = cls

            if const1_class is not None and const2_class is not None:
                if const1_class == const2_class:
                    continue
                else:
                    # Merge classes
                    equivalence_classes[const1_class].update(equivalence_classes[const2_class])
                    del equivalence_classes[const2_class]
            elif const1_class is not None:
                # Add const2 to const1's class
                equivalence_classes[const1_class].add(const2)
                assigned.add(const2)
            elif const2_class is not None:
                # Add const1 to const2's class
                equivalence_classes[const2_class].add(const1)
                assigned.add(const1)
            else:
                # Create a new class with both
                class_name = f"equivalence_{class_idx}"
                equivalence_classes[class_name] = {const1, const2}
                assigned.add(const1)
                assigned.add(const2)
                class_idx += 1

        # Add confidence scores
        equivalence_confidence = {}

        for cls, members in equivalence_classes.items():
            # Calculate average similarity within the class
            total_sim = 0
            count = 0

            for const1 in members:
                for const2 in members:
                    if const1 != const2 and const1 in similarity_matrix and const2 in similarity_matrix[const1]:
                        total_sim += similarity_matrix[const1][const2]
                        count += 1

            if count > 0:
                equivalence_confidence[cls] = total_sim / count
            else:
                equivalence_confidence[cls] = 0

        self.equivalence_confidence = equivalence_confidence

        return equivalence_classes

    def get_high_confidence_equivalences(self, min_confidence=0.8):
        """
        Get high-confidence functional equivalence classes.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            dict: High-confidence equivalence classes
        """
        high_confidence = {}

        for cls, confidence in self.equivalence_confidence.items():
            if confidence >= min_confidence:
                high_confidence[cls] = self.inferred_equivalences[cls]

        return high_confidence

    def apply_generalizations(self):
        """
        Apply the inferred generalizations to the architecture.

        Returns:
            bool: True if generalizations were applied
        """
        # Get high-confidence equivalences
        equivalences = self.get_high_confidence_equivalences()

        if not equivalences:
            return False

        # For each equivalence class, create a functional category
        category_level = self.architecture.levels[-1]  # Top level

        for cls, members in equivalences.items():
            # Convert to category name
            category_name = f"category_{len(category_level.categories)}"

            # Add to category level
            category_level.categories[category_name] = {
                'count': 0,
                'constructions': set(members),
                'instances': []
            }

            # Update construction-to-category mappings
            for const_id in members:
                if const_id not in category_level.construction_categories:
                    category_level.construction_categories[const_id] = []

                category_level.construction_categories[const_id].append(category_name)

        return True