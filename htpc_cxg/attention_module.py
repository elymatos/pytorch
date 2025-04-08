"""
Attention Module for the Predictive Coding Construction Grammar System.

This module implements attention mechanisms to weight important patterns
and constructions at different levels of the hierarchy.
"""

import numpy as np
from collections import defaultdict

class AttentionModule:
    def __init__(self, construction_registry=None):
        """
        Initialize the attention module.

        Args:
            construction_registry: Dictionary of constructions
        """
        self.construction_registry = construction_registry or {}
        self.pos_attention = {}  # Attention weights for POS tags
        self.construction_attention = {}  # Attention weights for constructions
        self.cross_level_attention = defaultdict(dict)  # Attention between levels

        # Hyperparameters for attention calculation
        self.attention_decay = 0.9  # Decay factor for past attention
        self.novelty_weight = 0.7  # Weight for novel patterns
        self.predefined_boost = 1.5  # Boost for predefined constructions

    def calculate_attention(self, pos_sequence, identified_constructions, transition_counts=None):
        """
        Calculate attention weights at multiple levels.

        Args:
            pos_sequence: List of POS tags
            identified_constructions: Dictionary of identified constructions
            transition_counts: Dictionary of transition counts (optional)

        Returns:
            dict: Dictionary with attention weights at different levels
        """
        # Calculate POS-level attention
        pos_attention = self._calculate_pos_attention(pos_sequence, transition_counts)

        # Calculate construction-level attention
        construction_attention = self._calculate_construction_attention(
            identified_constructions,
            give_priority_to_predefined=True
        )

        # Calculate cross-level attention
        cross_attention = self._calculate_cross_level_attention(
            pos_sequence,
            identified_constructions
        )

        # Integrate attention levels
        integrated_attention = self._integrate_attention_levels(
            pos_attention,
            construction_attention,
            cross_attention,
            pos_sequence,
            identified_constructions
        )

        # Update internal state
        self._update_attention_state(pos_attention, construction_attention)

        return {
            'pos': pos_attention,
            'construction': construction_attention,
            'cross': cross_attention,
            'integrated': integrated_attention
        }

    def _calculate_pos_attention(self, pos_sequence, transition_counts=None):
        """
        Calculate attention weights for POS tags.

        Args:
            pos_sequence: List of POS tags
            transition_counts: Dictionary of transition counts (optional)

        Returns:
            dict: Dictionary mapping POS tags to attention weights
        """
        attention = {}

        # If no sequence or counts, return empty attention
        if not pos_sequence:
            return attention

        # Get unique POS tags
        unique_pos = set(pos_sequence)

        if transition_counts:
            # Calculate attention based on transition entropy
            for pos in unique_pos:
                # Calculate entropy from transition counts
                transitions = transition_counts.get(pos, {})
                total = sum(transitions.values())

                if total > 0:
                    probs = [count/total for count in transitions.values()]
                    entropy = -sum(p * np.log2(p) for p in probs if p > 0)

                    # Higher entropy (more unpredictable) gets higher attention
                    attention[pos] = min(1.0, entropy / 4.0)  # Normalize to [0, 1]
                else:
                    attention[pos] = 0.5  # Default for unseen POS
        else:
            # Simple frequency-based attention
            pos_counts = {pos: pos_sequence.count(pos) for pos in unique_pos}
            total_count = len(pos_sequence)

            # Calculate inverse frequency (rare items get more attention)
            for pos, count in pos_counts.items():
                frequency = count / total_count
                # Inverse frequency, normalized
                attention[pos] = min(1.0, 0.1 + 0.9 * (1.0 - frequency))

        # Incorporate past attention with decay
        for pos in unique_pos:
            past_attention = self.pos_attention.get(pos, 0.5)
            attention[pos] = (1.0 - self.attention_decay) * attention[pos] + \
                             self.attention_decay * past_attention

        return attention

    def _calculate_construction_attention(self, constructions, give_priority_to_predefined=True):
        """
        Calculate attention weights for identified constructions.

        Args:
            constructions: Dictionary with keys 'predefined', 'new', and 'composite'
            give_priority_to_predefined: Whether to boost attention for predefined constructions

        Returns:
            dict: Dictionary mapping construction IDs to attention weights
        """
        attention_weights = {}

        # If no constructions, return empty attention
        if not constructions or 'all' not in constructions:
            return attention_weights

        # Process all constructions
        for const in constructions['all']:
            const_id = const['id']
            const_type = const['type']

            # Base attention weight depends on type
            if const_type == 'predefined' and give_priority_to_predefined:
                base_weight = 0.8 * self.predefined_boost
            elif const_type == 'composite':
                base_weight = 0.7
            else:  # new
                base_weight = 0.5 * self.novelty_weight

            # Adjust by construction length (longer constructions get more attention)
            length_factor = min(1.0, const.get('length', 1) / 5.0)

            # Adjust by construction frequency if available
            freq_factor = 1.0
            if const_id in self.construction_registry:
                freq = self.construction_registry[const_id].get('frequency', 1)
                # Novel constructions get higher attention
                freq_factor = min(1.0, 0.3 + 0.7 * (1.0 / (1.0 + np.log(1 + freq))))

            # Combine factors
            attention_weights[const_id] = base_weight * length_factor * freq_factor

            # Incorporate past attention with decay
            past_attention = self.construction_attention.get(const_id, 0.5)
            attention_weights[const_id] = (1.0 - self.attention_decay) * attention_weights[const_id] + \
                                         self.attention_decay * past_attention

        return attention_weights

    def _calculate_cross_level_attention(self, pos_sequence, constructions):
        """
        Calculate cross-level attention between POS tags and constructions.
        This captures how individual POS tags draw attention to specific constructions.

        Args:
            pos_sequence: List of POS tags
            constructions: Dictionary with identified constructions

        Returns:
            dict: Dictionary mapping (pos, construction_id) pairs to attention weights
        """
        cross_attention = {}

        # If no constructions, return empty cross-attention
        if not constructions or 'all' not in constructions:
            return cross_attention

        # Process each construction
        for const in constructions['all']:
            const_id = const['id']
            start = const['start']
            end = const['end']

            # Extract the POS tags covered by this construction
            if start < len(pos_sequence) and end <= len(pos_sequence):
                const_pos_tags = pos_sequence[start:end]

                # For each POS tag in this construction
                for pos in const_pos_tags:
                    key = (pos, const_id)

                    # Calculate the cross-attention weight
                    # Higher for POS tags that are distinctive for this construction
                    if const_id in self.construction_registry:
                        const_seq = self.construction_registry[const_id].get('pos_sequence', ())

                        # Calculate how distinctive this POS tag is for the construction
                        if const_seq:
                            pos_freq_in_const = const_seq.count(pos) / len(const_seq)
                            distinctiveness = min(1.0, pos_freq_in_const + 0.2)

                            cross_attention[key] = distinctiveness
                        else:
                            cross_attention[key] = 0.5
                    else:
                        cross_attention[key] = 0.5

        return cross_attention

    def _integrate_attention_levels(self, pos_attention, construction_attention,
                                   cross_attention, pos_sequence, constructions):
        """
        Integrate attention from multiple levels.

        Args:
            pos_attention: Dictionary of POS-level attention
            construction_attention: Dictionary of construction-level attention
            cross_attention: Dictionary of cross-level attention
            pos_sequence: List of POS tags
            constructions: Dictionary with identified constructions

        Returns:
            dict: Dictionary with integrated attention weights for positions in sequence
        """
        # Initialize integrated attention for each position
        integrated = {i: 0.5 for i in range(len(pos_sequence))}

        # Add POS-level attention
        for i, pos in enumerate(pos_sequence):
            if pos in pos_attention:
                integrated[i] += 0.3 * pos_attention[pos]

        # Add construction-level attention
        if 'all' in constructions:
            for const in constructions['all']:
                const_id = const['id']
                if const_id in construction_attention:
                    const_attention = construction_attention[const_id]

                    # Add this attention to all positions covered by the construction
                    for pos in range(const['start'], min(const['end'], len(pos_sequence))):
                        integrated[pos] += 0.4 * const_attention

        # Add cross-level attention
        for (pos, const_id), cross_value in cross_attention.items():
            # Find positions with this POS tag within this construction
            for const in constructions.get('all', []):
                if const['id'] == const_id:
                    for i in range(const['start'], min(const['end'], len(pos_sequence))):
                        if i < len(pos_sequence) and pos_sequence[i] == pos:
                            integrated[i] += 0.3 * cross_value

        # Normalize values to [0, 1]
        for pos in integrated:
            integrated[pos] = min(1.0, integrated[pos])

        return integrated

    def _update_attention_state(self, pos_attention, construction_attention):
        """
        Update the internal attention state.

        Args:
            pos_attention: Dictionary of POS-level attention
            construction_attention: Dictionary of construction-level attention
        """
        # Update POS attention
        for pos, value in pos_attention.items():
            self.pos_attention[pos] = value

        # Update construction attention
        for const_id, value in construction_attention.items():
            self.construction_attention[const_id] = value

    def get_attention_for_position(self, position, integrated_attention):
        """
        Get the integrated attention weight for a specific position.

        Args:
            position: Position in the sequence
            integrated_attention: Dictionary of integrated attention

        Returns:
            float: Attention weight for the position
        """
        return integrated_attention.get(position, 0.5)

    def get_attention_for_construction(self, const_id, construction_attention):
        """
        Get the attention weight for a specific construction.

        Args:
            const_id: Construction ID
            construction_attention: Dictionary of construction attention

        Returns:
            float: Attention weight for the construction
        """
        return construction_attention.get(const_id, 0.5)