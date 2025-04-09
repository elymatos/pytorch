"""
Main module for the Hierarchical Temporal Predictive Coding (HTPC)
Construction Grammar system.

This module provides the main interface for using the system.
"""

from collections import defaultdict, Counter

from htpc_architecture import HTPCArchitecture
from htpc_learning_module import HTPCLearningModule


class HTPCSystem:
    """
    Main system class for HTPC Construction Grammar.
    """

    def __init__(self, num_hierarchical_levels=3):
        """
        Initialize the HTPC system.

        Args:
            num_hierarchical_levels: Number of hierarchical levels
        """
        self.architecture = HTPCArchitecture(num_hierarchical_levels)
        self.learning_module = HTPCLearningModule(self.architecture)

        # Processing history
        self.processing_history = []

        # Generalization history
        self.generalizations = {
            'inferred': {},
            'confidence': {}
        }

        # System parameters
        self.generalization_interval = 10  # Process this many sequences before generalizing
        self.sequences_processed = 0

    def process_sequence(self, pos_sequence):
        """
        Process a POS sequence through the system.

        Args:
            pos_sequence: List of POS tags

        Returns:
            dict: Processing results
        """
        # Process through the architecture
        results = self.architecture.process_sequence(pos_sequence)

        # Update learning module
        self.learning_module.update_from_errors(results['prediction_errors'])

        # Observe patterns for future generalization
        self.learning_module.observe_context_patterns(results)
        self.learning_module.observe_substitution_patterns(results)

        # Update processing history
        self.processing_history.append({
            'sequence': pos_sequence,
            'results': results
        })

        self.sequences_processed += 1

        # Periodically attempt to infer generalizations
        if self.sequences_processed % self.generalization_interval == 0:
            generalizations = self.learning_module.infer_functional_equivalence()
            confidence = self.learning_module.equivalence_confidence

            self.generalizations['inferred'] = generalizations
            self.generalizations['confidence'] = confidence

            # Apply high-confidence generalizations
            self.learning_module.apply_generalizations()

        return results

    def get_constructions(self, level=None):
        """
        Get constructions recognized by the system.

        Args:
            level: Optional level index to get constructions from

        Returns:
            dict: Recognized constructions
        """
        if level is not None:
            if 0 <= level < len(self.architecture.levels):
                return self.architecture.levels[level].get_constructions()
            else:
                return {}
        else:
            constructions = {}
            for i, level in enumerate(self.architecture.levels):
                constructions[i] = level.get_constructions()
            return constructions

    def get_inferred_equivalences(self, min_confidence=0.0):
        """
        Get inferred functional equivalences.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            dict: Inferred equivalence classes
        """
        if min_confidence > 0:
            return self.learning_module.get_high_confidence_equivalences(min_confidence)
        else:
            return self.generalizations['inferred']

    def get_categories(self):
        """
        Get functional categories recognized by the system.

        Returns:
            dict: Functional categories
        """
        # Categories are maintained at the top level
        top_level = self.architecture.levels[-1]

        if hasattr(top_level, 'categories'):
            return top_level.categories
        else:
            return {}

    def predict_next_pos(self, partial_sequence, k=3):
        """
        Predict the next POS tags for a partial sequence.

        Args:
            partial_sequence: Partial POS sequence
            k: Number of predictions to return

        Returns:
            list: Top k predicted next POS tags with probabilities
        """
        # Process the partial sequence
        results = self.architecture.process_sequence(partial_sequence)

        # Get predictions from the lowest level
        predictions = results['predictions'].get(0, {})

        # Aggregate all next-position predictions
        next_pos_probs = {}

        # Check for predictions in the results
        if predictions:
            for pos, pred in predictions.items():
                if isinstance(pred, dict) and 'type' in pred and pred['type'] == 'pos_transition':
                    for pos_tag, prob in pred.get('probabilities', {}).items():
                        if pos_tag not in next_pos_probs:
                            next_pos_probs[pos_tag] = 0
                        next_pos_probs[pos_tag] += prob

        # If no predictions from model, use transition statistics from POSLevel
        if not next_pos_probs and partial_sequence:
            pos_level = self.architecture.levels[0]
            last_pos = partial_sequence[-1]

            if last_pos in pos_level.transitions:
                transitions = pos_level.transitions[last_pos]
                total = sum(transitions.values())

                if total > 0:
                    for next_pos, count in transitions.items():
                        next_pos_probs[next_pos] = count / total

        # If still no predictions, provide defaults based on global frequencies
        if not next_pos_probs:
            # Determine common POS tags from training data
            common_tags = {
                'NOUN': 0.3,
                'VERB': 0.2,
                'DET': 0.15,
                'ADJ': 0.1,
                'PREP': 0.1,
                'PRON': 0.1,
                'ADV': 0.05
            }
            next_pos_probs = common_tags

        # Normalize
        total = sum(next_pos_probs.values()) or 1.0
        normalized = {tag: prob / total for tag, prob in next_pos_probs.items()}

        # Sort and return top k
        sorted_preds = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        return sorted_preds[:k]

    def reset(self):
        """
        Reset the system.

        Returns:
            bool: True if reset was successful
        """
        # Reinitialize architecture and learning module
        num_levels = len(self.architecture.levels)
        self.architecture = HTPCArchitecture(num_levels)
        self.learning_module = HTPCLearningModule(self.architecture)

        # Reset history
        self.processing_history = []
        self.generalizations = {
            'inferred': {},
            'confidence': {}
        }
        self.sequences_processed = 0

        return True