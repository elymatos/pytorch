"""
Construction Module for the Predictive Coding Construction Grammar System.

This module handles the identification, registration, and management of
grammatical constructions of varying sizes, including their hierarchical relationships.
"""

from collections import defaultdict, Counter
import numpy as np


class ConstructionModule:
    def __init__(self, min_chunk_size=1, predefined_constructions=None):
        """
        Initialize the construction module.

        Args:
            min_chunk_size: Minimum size of a construction (default: 1 for single POS)
            predefined_constructions: List of tuples/lists, each containing a sequence
                                     of POS tags that define a construction
        """
        self.min_chunk_size = min_chunk_size
        self.construction_registry = {}  # Stores all constructions
        self.hierarchical_relations = defaultdict(list)  # parent_id -> [child_ids]
        self.specialization_relations = defaultdict(list)  # general_id -> [specialized_ids]
        self.construction_frequencies = Counter()  # Frequency counts for constructions
        self.construction_transitions = defaultdict(Counter)  # Transitions between constructions

        # Register predefined constructions if provided
        if predefined_constructions:
            self._register_predefined_constructions(predefined_constructions)

    def _register_predefined_constructions(self, constructions):
        """
        Register a list of predefined constructions.

        Args:
            constructions: List of tuples/lists, each containing a sequence
                          of POS tags that define a construction
        """
        for idx, construction in enumerate(constructions):
            # Create a unique ID for each predefined construction
            construction_id = f"pre_{idx}"
            # Store the construction with metadata
            self.construction_registry[construction_id] = {
                'pos_sequence': tuple(construction),
                'predefined': True,
                'frequency': 0,  # Will track occurrences in actual data
                'confidence': 1.0,  # High initial confidence
                'entropy': 0.0,  # Entropy of internal transitions
                'cohesion': 0.0  # Measure of internal cohesion
            }

    def identify_constructions(self, pos_sequence):
        """
        Identify all constructions in a POS sequence.

        Args:
            pos_sequence: List of POS tags

        Returns:
            dict: Dictionary with keys 'predefined', 'new', and 'composite',
                 each containing a list of identified constructions
        """
        # First: identify occurrences of predefined constructions
        matched_predefined = self._match_predefined_constructions(pos_sequence)

        # Second: identify potential new constructions that aren't covered
        new_constructions = self._identify_new_constructions(pos_sequence, matched_predefined)

        # Third: identify higher-level constructions (combinations of existing ones)
        higher_level = self._identify_composite_constructions(matched_predefined, new_constructions)

        # Update statistics and hierarchical relationships
        self._update_construction_statistics(matched_predefined, new_constructions, higher_level)

        # All identified constructions
        all_constructions = matched_predefined + new_constructions + higher_level

        return {
            'predefined': matched_predefined,
            'new': new_constructions,
            'composite': higher_level,
            'all': all_constructions
        }

    def _match_predefined_constructions(self, pos_sequence):
        """
        Match predefined constructions in the given POS sequence.

        Args:
            pos_sequence: List of POS tags

        Returns:
            list: List of dictionaries with construction match information
        """
        matches = []
        seq_len = len(pos_sequence)

        # For each predefined construction
        for const_id, const_info in self.construction_registry.items():
            if not const_info.get('predefined', False):
                continue  # Skip non-predefined constructions

            const_pos = const_info['pos_sequence']
            const_len = len(const_pos)

            # Skip if sequence is shorter than construction
            if seq_len < const_len:
                continue

            # Sliding window search through the sequence
            for i in range(seq_len - const_len + 1):
                if tuple(pos_sequence[i:i + const_len]) == const_pos:
                    matches.append({
                        'id': const_id,
                        'start': i,
                        'end': i + const_len,
                        'type': 'predefined',
                        'length': const_len
                    })

        # Sort matches by position and handle overlaps
        return self._resolve_overlapping_matches(matches)

    def _identify_new_constructions(self, pos_sequence, existing_matches):
        """
        Identify potential new constructions in regions not covered by
        predefined constructions.

        Args:
            pos_sequence: List of POS tags
            existing_matches: List of dictionaries with existing construction matches

        Returns:
            list: List of dictionaries with new construction match information
        """
        # Find uncovered regions
        covered_positions = set()
        for match in existing_matches:
            for pos in range(match['start'], match['end']):
                covered_positions.add(pos)

        # Find continuous uncovered regions
        uncovered_regions = []
        current_region = []

        for i, pos in enumerate(pos_sequence):
            if i not in covered_positions:
                current_region.append((i, pos))
            elif current_region:
                if len(current_region) >= self.min_chunk_size:
                    uncovered_regions.append(current_region)
                current_region = []

        # Add the last region if it exists
        if current_region and len(current_region) >= self.min_chunk_size:
            uncovered_regions.append(current_region)

        # For now, a simple approach: consider each uncovered region as a potential new construction
        new_matches = []
        for idx, region in enumerate(uncovered_regions):
            positions = [pos for pos, _ in region]
            pos_tags = [tag for _, tag in region]

            # Only consider regions that meet minimum size
            if len(pos_tags) < self.min_chunk_size:
                continue

            # Create a new construction ID
            const_id = f"new_{len(self.construction_registry)}"

            # Register this as a new construction
            self.construction_registry[const_id] = {
                'pos_sequence': tuple(pos_tags),
                'predefined': False,
                'frequency': 1,  # First occurrence
                'confidence': 0.5,  # Lower initial confidence
                'entropy': 0.0,
                'cohesion': 0.0
            }

            new_matches.append({
                'id': const_id,
                'start': min(positions),
                'end': max(positions) + 1,
                'type': 'new',
                'length': len(pos_tags)
            })

        return new_matches

    def _identify_composite_constructions(self, predefined_matches, new_matches):
        """
        Identify higher-level constructions that are combinations of
        predefined and newly discovered constructions.

        Args:
            predefined_matches: List of dictionaries with predefined construction matches
            new_matches: List of dictionaries with new construction matches

        Returns:
            list: List of dictionaries with composite construction match information
        """
        # Combine all matches and sort by position
        all_matches = predefined_matches + new_matches
        sorted_matches = sorted(all_matches, key=lambda x: x['start'])

        # Find adjacent constructions
        composite_candidates = []

        for i in range(len(sorted_matches) - 1):
            current = sorted_matches[i]
            next_match = sorted_matches[i + 1]

            # Check if they are adjacent
            if current['end'] == next_match['start']:
                composite_candidates.append((current, next_match))

        # For now, a simple approach: consider each pair of adjacent constructions
        composite_matches = []

        for idx, (const1, const2) in enumerate(composite_candidates):
            # Create a composite ID
            comp_id = f"composite_{len(self.construction_registry)}"

            # Get the construction sequences
            seq1 = self.construction_registry[const1['id']]['pos_sequence']
            seq2 = self.construction_registry[const2['id']]['pos_sequence']

            # Combined sequence
            combined_seq = seq1 + seq2

            # Register this as a new composite construction
            self.construction_registry[comp_id] = {
                'pos_sequence': combined_seq,
                'predefined': False,
                'composite': True,
                'components': [const1['id'], const2['id']],
                'frequency': 1,  # First occurrence
                'confidence': 0.3,  # Lower initial confidence
                'entropy': 0.0,
                'cohesion': 0.0
            }

            # Update hierarchical relations
            self.hierarchical_relations[comp_id].extend([const1['id'], const2['id']])

            composite_matches.append({
                'id': comp_id,
                'start': const1['start'],
                'end': const2['end'],
                'type': 'composite',
                'length': len(combined_seq),
                'components': [const1['id'], const2['id']]
            })

        return composite_matches

    def _resolve_overlapping_matches(self, matches):
        """
        Resolve overlapping construction matches, typically giving
        preference to longer or predefined constructions.

        Args:
            matches: List of dictionaries with construction match information

        Returns:
            list: List of dictionaries with resolved construction match information
        """
        if not matches:
            return []

        # Sort matches by length (descending) and then by type priority
        type_priority = {'predefined': 3, 'composite': 2, 'new': 1}
        sorted_matches = sorted(matches,
                                key=lambda x: (x['length'], type_priority.get(x['type'], 0)),
                                reverse=True)

        # Greedy algorithm to resolve overlaps
        resolved_matches = []
        covered_positions = set()

        for match in sorted_matches:
            # Check if this match overlaps with already covered positions
            match_positions = set(range(match['start'], match['end']))
            if not match_positions.intersection(covered_positions):
                resolved_matches.append(match)
                covered_positions.update(match_positions)

        # Sort by start position for output
        return sorted(resolved_matches, key=lambda x: x['start'])

    def _update_construction_statistics(self, predefined, new, composite):
        """
        Update statistics for all identified constructions.

        Args:
            predefined: List of predefined construction matches
            new: List of new construction matches
            composite: List of composite construction matches
        """
        # Update frequencies
        for match in predefined + new + composite:
            const_id = match['id']
            self.construction_frequencies[const_id] += 1

            if const_id in self.construction_registry:
                self.construction_registry[const_id]['frequency'] += 1

        # Update construction transitions
        all_matches = sorted(predefined + new + composite, key=lambda x: x['start'])

        for i in range(len(all_matches) - 1):
            current = all_matches[i]
            next_match = all_matches[i + 1]

            # If they are adjacent, record the transition
            if current['end'] == next_match['start']:
                self.construction_transitions[current['id']][next_match['id']] += 1

        # Update cohesion and entropy measures for each construction
        for const_id in self.construction_registry:
            # Calculate internal cohesion based on mutual information
            # This would require additional data from the base module
            # For now, we'll set a placeholder value
            self.construction_registry[const_id]['cohesion'] = 0.5

            # Entropy calculation would also need additional data
            self.construction_registry[const_id]['entropy'] = 0.0

    def get_construction_sequence(self, const_id):
        """
        Get the POS sequence for a construction.

        Args:
            const_id: Construction ID

        Returns:
            tuple: Tuple of POS tags
        """
        if const_id not in self.construction_registry:
            return ()

        return self.construction_registry[const_id]['pos_sequence']

    def get_component_constructions(self, const_id):
        """
        Get the component constructions for a composite construction.

        Args:
            const_id: Construction ID

        Returns:
            list: List of component construction IDs
        """
        if const_id not in self.construction_registry:
            return []

        return self.construction_registry[const_id].get('components', [])

    def _identify_specialization_relations(self):
        """
        Identify specialization relationships between constructions.
        A construction is a specialization of another if it contains the other
        as a subsequence.

        Returns:
            dict: Dictionary mapping general construction IDs to specialized construction IDs
        """
        specialization_map = defaultdict(list)

        for const_id, const_info in self.construction_registry.items():
            # Skip composite constructions as they have explicit component relations
            if const_info.get('composite', False):
                continue

            seq1 = const_info['pos_sequence']

            for other_id, other_info in self.construction_registry.items():
                if other_id == const_id:
                    continue

                seq2 = other_info['pos_sequence']

                # Check if seq2 is a subsequence of seq1
                if len(seq2) < len(seq1):
                    # Try to find seq2 as a contiguous subsequence in seq1
                    for i in range(len(seq1) - len(seq2) + 1):
                        if seq1[i:i + len(seq2)] == seq2:
                            specialization_map[other_id].append(const_id)
                            break

        return specialization_map

    def update_specialization_relations(self):
        """
        Update the specialization relations between constructions.
        """
        self.specialization_relations = self._identify_specialization_relations()

    def _annotate_construction_function(self, construction_id):
        """
        Annotate the grammatical function of a construction.

        Args:
            construction_id: Construction ID

        Returns:
            str: Functional annotation (e.g., 'NP', 'VP', etc.)
        """
        if construction_id not in self.construction_registry:
            return "Unknown"

        const_info = self.construction_registry[construction_id]
        seq = const_info['pos_sequence']

        # Simple heuristic-based annotation
        if not seq:
            return "Unknown"

        # Very simple rules for illustration
        if seq[0] == 'DET' and 'NOUN' in seq:
            return "NP"  # Noun Phrase
        elif seq[0] == 'VERB':
            return "VP"  # Verb Phrase
        elif 'PREP' in seq:
            return "PP"  # Prepositional Phrase

        # Default
        return "Unknown"

    def annotate_all_constructions(self):
        """
        Annotate all constructions with functional labels.
        """
        for const_id in self.construction_registry:
            function = self._annotate_construction_function(const_id)
            self.construction_registry[const_id]['function'] = function

    def _build_construction_hierarchy(self, base_constructions):
        """
        Build a hierarchy of constructions starting from base constructions.

        Args:
            base_constructions: List of base construction IDs

        Returns:
            dict: Dictionary with levels of the hierarchy
        """
        # Start with base constructions
        current_level = base_constructions
        all_levels = {'level_0': current_level}

        level = 1
        # Iteratively build higher levels
        while True:
            next_level = []

            # Find compositions of current level constructions
            for i in range(len(current_level)):
                for j in range(len(current_level)):
                    comp_id = self._find_composition(current_level[i], current_level[j])
                    if comp_id and comp_id not in next_level:
                        next_level.append(comp_id)

            if not next_level:  # No new compositions found
                break

            all_levels[f'level_{level}'] = next_level
            current_level = next_level
            level += 1

        return all_levels

    def _find_composition(self, const_id1, const_id2):
        """
        Find a composition of two constructions, if it exists.

        Args:
            const_id1: First construction ID
            const_id2: Second construction ID

        Returns:
            str: ID of the composition, or None if no composition exists
        """
        for const_id, const_info in self.construction_registry.items():
            if const_info.get('composite', False) and \
                    const_info.get('components', []) == [const_id1, const_id2]:
                return const_id

        return None