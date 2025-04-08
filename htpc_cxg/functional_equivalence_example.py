"""
Example demonstrating functional equivalence between constructions.

This script shows how to define functionally equivalent constructions
and template constructions that allow substitutions.
"""

from main_module import MainModule


def main():
    """
    Main function to demonstrate functional equivalence.
    """
    print("Initializing Construction Grammar Predictive Coding System...")

    # Define predefined constructions
    predefined_constructions = [
        ('DET', 'NOUN'),  # construction #0 - simple NP with determiner
        ('NOUN',),  # construction #1 - bare noun NP
        ('DET', 'ADJ', 'NOUN'),  # construction #2 - modified NP
        ('VERB', 'DET', 'NOUN'),  # construction #3 - simple VP with object
        ('VERB', 'NOUN'),  # construction #4 - simple VP with bare noun object
        ('NOUN', 'VERB'),  # construction #5 - simple subject-verb
    ]

    # Initialize the system
    system = MainModule(predefined_constructions=predefined_constructions)

    # Print initial constructions
    print("\nPredefined Constructions:")
    for const_id, const_info in system.construction_registry.items():
        print(f"  {const_id}: {const_info['pos_sequence']}")

    # Define functional equivalences
    print("\nDefining Functional Equivalences:")

    # Define noun phrase category
    system.define_functional_equivalence('NP', ['pre_0', 'pre_1', 'pre_2'])
    print("  Defined 'NP' category with: 'DET NOUN', 'NOUN', 'DET ADJ NOUN'")

    # Define verb phrase category
    system.define_functional_equivalence('VP', ['pre_3', 'pre_4'])
    print("  Defined 'VP' category with: 'VERB DET NOUN', 'VERB NOUN'")

    # Define template construction that allows substitutions
    print("\nDefining Template Construction:")
    template_id = system.define_template_construction(
        ['NP', 'VERB', 'NP'],
        substitution_slots={0: ['NP'], 2: ['NP']}
    )
    print(f"  Template {template_id}: 'NP VERB NP' with NP substitutions allowed")

    # Print construction details with categories
    print("\nConstruction Details with Categories:")
    for const_id in ['pre_0', 'pre_1', 'pre_2']:
        details = system.get_construction_details(const_id)
        print(f"  {const_id}: {details['pos_sequence']}")
        print(f"    Categories: {details.get('categories', [])}")
        print(f"    Equivalents: {details.get('equivalent_constructions', [])}")

    # Test with different sentence patterns
    test_sequences = [
        # NP variants
        ['DET', 'NOUN', 'VERB', 'DET', 'NOUN'],  # The dog chased the cat
        ['NOUN', 'VERB', 'DET', 'NOUN'],  # Dogs chase the cat
        ['DET', 'ADJ', 'NOUN', 'VERB', 'NOUN'],  # The big dog chased cats

        # Mixed NP types
        ['DET', 'NOUN', 'VERB', 'NOUN'],  # The dog chased cats
        ['NOUN', 'VERB', 'DET', 'ADJ', 'NOUN'],  # Dogs chase the small cats
    ]

    print("\nTesting with Different Sentence Patterns:")

    for i, sequence in enumerate(test_sequences):
        print(f"\n--- Sequence {i + 1}: {sequence} ---")

        # Process the sequence
        results = system.process_sequence(sequence)

        # Print identified constructions
        print("Identified Constructions:")
        constructions = results.get('constructions', {})

        # Track if we found a template match
        found_template = False

        for const_type in ['predefined', 'new', 'composite']:
            if const_type in constructions and constructions[const_type]:
                print(f"  {const_type.capitalize()}:")
                for const in constructions[const_type]:
                    const_id = const['id']
                    start = const['start']
                    end = const['end']
                    const_sequence = sequence[start:end]
                    print(f"    {const_id}: {const_sequence} (position {start}-{end})")

                    # Check if this is a template match
                    if const.get('type') == 'template':
                        found_template = True
                        print(f"      ** Template match using substitutions: **")
                        if 'substitutions' in const:
                            for pos, subst in const['substitutions'].items():
                                subst_id = subst['const_id']
                                if subst_id in system.construction_registry:
                                    subst_pattern = system.construction_registry[subst_id]['pos_sequence']
                                    print(f"        Position {pos}: {subst_id} {subst_pattern}")

    # Test prediction with functional equivalence
    print("\n--- Testing Prediction with Functional Equivalence ---")
    partial = ['DET', 'NOUN', 'VERB']
    print(f"Partial sequence: {partial}")

    # Process sequence to build statistics
    system.process_sequence(['DET', 'NOUN', 'VERB', 'DET', 'NOUN'])
    system.process_sequence(['DET', 'NOUN', 'VERB', 'NOUN'])

    # Get predictions
    predictions = system.predict_for_partial_sequence(partial)

    print("Next POS predictions:")
    if 'next_pos_predictions' in predictions:
        for pos, prob in predictions['next_pos_predictions'][:3]:
            print(f"  {pos}: {prob:.2f}")

    # Show how predictions would differ without equivalence
    # (We can't easily demonstrate this without modifying the system,
    # but in a real implementation, predictions would be enriched by
    # considering functionally equivalent constructions)
    print("\nWith functional equivalence, both 'DET' and 'NOUN' are")
    print("predicted as possible continuations because they both can")
    print("start an NP, which is expected after 'DET NOUN VERB'.")


if __name__ == "__main__":
    main()