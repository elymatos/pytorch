"""
Base Module for the Predictive Coding Construction Grammar System.

This module provides core functionality for processing POS sequences,
building transition graphs, and handling basic sequence operations.
"""

import numpy as np
from collections import defaultdict, Counter


class BaseModule:
    def __init__(self):
        """
        Initialize the base module with core data structures for POS sequence processing.
        """
        # Transition probability matrices
        self.forward_transitions = defaultdict(Counter)
        self.backward_transitions = defaultdict(Counter)

        # Statistics for POS tags
        self.pos_frequencies = Counter()
        self.total_pos_count = 0

        # Transition counts
        self.transition_counts = defaultdict(Counter)
        self.total_transitions = 0

    def process_sequence(self, pos_sequence):
        """
        Process a POS sequence and update transition probabilities.

        Args:
            pos_sequence: List of POS tags
        """
        # Update POS frequencies
        self.pos_frequencies.update(pos_sequence)
        self.total_pos_count += len(pos_sequence)

        # Update forward transitions
        for i in range(len(pos_sequence) - 1):
            current_pos = pos_sequence[i]
            next_pos = pos_sequence[i + 1]
            self.forward_transitions[current_pos][next_pos] += 1
            self.transition_counts[current_pos][next_pos] += 1
            self.total_transitions += 1

        # Update backward transitions
        for i in range(1, len(pos_sequence)):
            current_pos = pos_sequence[i]
            prev_pos = pos_sequence[i - 1]
            self.backward_transitions[current_pos][prev_pos] += 1

    def get_forward_probability(self, pos_tag, next_pos_tag):
        """
        Get the probability of a forward transition from pos_tag to next_pos_tag.

        Args:
            pos_tag: Current POS tag
            next_pos_tag: Next POS tag

        Returns:
            float: Probability of the transition
        """
        if pos_tag not in self.forward_transitions:
            return 0.0

        total = sum(self.forward_transitions[pos_tag].values())
        if total == 0:
            return 0.0

        return self.forward_transitions[pos_tag][next_pos_tag] / total

    def get_backward_probability(self, pos_tag, prev_pos_tag):
        """
        Get the probability of a backward transition from pos_tag to prev_pos_tag.

        Args:
            pos_tag: Current POS tag
            prev_pos_tag: Previous POS tag

        Returns:
            float: Probability of the transition
        """
        if pos_tag not in self.backward_transitions:
            return 0.0

        total = sum(self.backward_transitions[pos_tag].values())
        if total == 0:
            return 0.0

        return self.backward_transitions[pos_tag][prev_pos_tag] / total

    def predict_next_pos(self, pos_sequence, k=3):
        """
        Predict the k most likely next POS tags given a sequence.

        Args:
            pos_sequence: List of POS tags
            k: Number of predictions to return

        Returns:
            list: k most likely next POS tags with probabilities
        """
        if not pos_sequence:
            # If sequence is empty, return the most common POS tags
            total = sum(self.pos_frequencies.values())
            if total == 0:
                return []

            probs = {pos: count / total for pos, count in self.pos_frequencies.most_common(k)}
            return sorted(probs.items(), key=lambda x: x[1], reverse=True)

        # Get the last POS tag
        last_pos = pos_sequence[-1]

        # If we haven't seen this POS tag before, return most common next tags
        if last_pos not in self.forward_transitions:
            total = sum(self.pos_frequencies.values())
            if total == 0:
                return []

            probs = {pos: count / total for pos, count in self.pos_frequencies.most_common(k)}
            return sorted(probs.items(), key=lambda x: x[1], reverse=True)

        # Get forward transition probabilities
        transitions = self.forward_transitions[last_pos]
        total = sum(transitions.values())

        if total == 0:
            return []

        probs = {pos: count / total for pos, count in transitions.most_common(k)}
        return sorted(probs.items(), key=lambda x: x[1], reverse=True)

    def calculate_mutual_information(self, pos_a, pos_b):
        """
        Calculate mutual information between two POS tags.

        Args:
            pos_a: First POS tag
            pos_b: Second POS tag

        Returns:
            float: Mutual information
        """
        if self.total_transitions == 0:
            return 0.0

        # Joint probability
        joint_prob = self.transition_counts[pos_a][pos_b] / self.total_transitions

        if joint_prob == 0:
            return 0.0

        # Marginal probabilities
        prob_a = sum(self.transition_counts[pos_a].values()) / self.total_transitions
        prob_b = sum(self.transition_counts[x][pos_b] for x in self.transition_counts) / self.total_transitions

        if prob_a == 0 or prob_b == 0:
            return 0.0

        # Calculate mutual information
        return joint_prob * np.log2(joint_prob / (prob_a * prob_b))

    def get_transition_entropy(self, pos_tag, direction='forward'):
        """
        Calculate the entropy of transitions from a POS tag.

        Args:
            pos_tag: POS tag
            direction: 'forward' or 'backward'

        Returns:
            float: Entropy value
        """
        if direction == 'forward':
            transitions = self.forward_transitions[pos_tag]
        else:
            transitions = self.backward_transitions[pos_tag]

        total = sum(transitions.values())

        if total == 0:
            return 0.0

        probs = [count / total for count in transitions.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)