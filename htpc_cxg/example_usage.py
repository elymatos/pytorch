"""
Example usage of the Predictive Coding Construction Grammar System.

This script demonstrates how to use the system with predefined constructions,
process sequences, and analyze the results.
"""

from main_module import MainModule

def main():
    """
    Main function to demonstrate the system.
    """
    print("Initializing Construction Grammar Predictive Coding System...")

    # Define some predefined constructions
    predefined_constructions = [
        ('DET', 'NOUN'),                   # Simple noun phrase
        ('DET', 'ADJ', 'NOUN'),            # Modified noun phrase
        ('VERB', 'DET', 'NOUN'),           # Simple verb phrase with object
        ('NOUN', 'VERB'),                  # Simple subject-verb
        ('DET', 'NOUN', 'VERB', 'DET', 'NOUN')  # Simple sentence
    ]

    # Initialize the system
    system = MainModule(predefined_constructions=predefined_constructions)

    # Print initial system state
    print("\nPredefined Constructions:")
    for const_id, const_info in system.construction_registry.items():
        print(f"  {const_id}: {const_info['pos_sequence']}")

    # Example POS sequences to process
    sequences = [
        ['DET', 'NOUN', 'VERB', 'DET', 'ADJ', 'NOUN'],
        ['NOUN', 'VERB', 'DET', 'NOUN'],
        ['DET', 'ADJ', 'NOUN', 'VERB', 'NOUN'],
        ['VERB', 'DET', 'NOUN', 'PREP', 'DET', 'NOUN']
    ]

    # Process each sequence
    for i, sequence in enumerate(sequences):
        print(f"\n--- Processing Sequence {i+1}: {sequence} ---")

        # Process the sequence with bidirectional analysis
        results = system.process_sequence(sequence, bidirectional=True)

        # Print identified constructions
        print("\nIdentified Constructions:")
        constructions = None

        # Try various possible locations for constructions in the results
        if 'constructions' in results:
            constructions = results['constructions']
        elif 'combined' in results and 'constructions' in results['combined']:
            constructions = results['combined']['constructions']

        if constructions:
            for const_type in ['predefined', 'new', 'composite']:
                if const_type in constructions and constructions[const_type]:
                    print(f"  {const_type.capitalize()}:")
                    for const in constructions[const_type]:
                        const_id = const['id']
                        start = const['start']
                        end = const['end']
                        const_sequence = sequence[start:end]
                        print(f"    {const_id}: {const_sequence} (position {start}-{end})")
                else:
                    print(f"  {const_type.capitalize()}: None identified")
        else:
            print("  No constructions identified")

        # Print attention highlights
        print("\nAttention Highlights:")
        attention = None

        # Try various possible locations for attention in the results
        if 'attention' in results:
            attention = results['attention']
        elif 'combined' in results and 'attention' in results['combined']:
            attention = results['combined']['attention']

        if attention and 'integrated' in attention:
            integrated = attention['integrated']

            # Find positions with highest attention
            sorted_positions = sorted(
                [(pos, val) for pos, val in integrated.items() if isinstance(pos, int)],
                key=lambda x: x[1],
                reverse=True
            )

            if sorted_positions:
                for pos, value in sorted_positions[:3]:
                    if pos < len(sequence):
                        print(f"  Position {pos}: {sequence[pos]} (attention: {value:.2f})")
            else:
                print("  No position-specific attention values found")
        else:
            print("  No attention data available")

        # Print predictions
        print("\nNext POS Predictions:")
        predictions = None

        # Try various possible locations for predictions in the results
        if 'predictions' in results:
            predictions = results['predictions']
        elif 'combined' in results and 'predictions' in results['combined']:
            predictions = results['combined']['predictions']

        if predictions and 'next_pos' in predictions:
            next_pos = predictions['next_pos']
            if next_pos:
                sorted_preds = sorted(next_pos.items(), key=lambda x: x[1], reverse=True)

                for pos, prob in sorted_preds[:3]:
                    print(f"  {pos}: {prob:.2f}")
            else:
                print("  No next POS predictions available")
        else:
            print("  No prediction data available")

        # Print prediction errors
        prediction_error = None
        if 'combined' in results and 'prediction_error' in results['combined']:
            prediction_error = results['combined']['prediction_error']
        else:
            prediction_error = results.get('prediction_error')

        if prediction_error:
            print(f"\nPrediction Error: {prediction_error.get('total_error', 0.0):.4f}")

    # Demonstrate adding a new construction
    print("\n--- Adding a New Construction ---")
    new_construction = ('PREP', 'DET', 'NOUN')
    const_id = system.add_predefined_construction(new_construction)
    print(f"Added construction {const_id}: {new_construction}")

    # Process a sequence with the new construction
    test_sequence = ['DET', 'NOUN', 'VERB', 'PREP', 'DET', 'NOUN']
    print(f"\nProcessing sequence with new construction: {test_sequence}")
    results = system.process_sequence(test_sequence)

    # Print identified constructions
    print("\nIdentified Constructions (should include the new one):")
    if 'combined' in results and 'constructions' in results['combined']:
        constructions = results['combined']['constructions']
        for const_type in ['predefined', 'new', 'composite']:
            if const_type in constructions:
                print(f"  {const_type.capitalize()}:")
                for const in constructions[const_type]:
                    const_id = const['id']
                    start = const['start']
                    end = const['end']
                    const_sequence = test_sequence[start:end]
                    print(f"    {const_id}: {const_sequence} (position {start}-{end})")

    # Demonstrate getting construction details
    print("\n--- Construction Details ---")
    # Get a composite construction if available
    composite_id = None
    for result in system.processing_history:
        if 'combined' in result['results'] and 'constructions' in result['results']['combined']:
            constructions = result['results']['combined']['constructions']
            if 'composite' in constructions and constructions['composite']:
                composite_id = constructions['composite'][0]['id']
                break

    if composite_id:
        details = system.get_construction_details(composite_id)
        print(f"Details for composite construction {composite_id}:")
        print(f"  POS sequence: {details['pos_sequence']}")
        print(f"  Frequency: {details['frequency']}")
        print(f"  Components: {details.get('components', [])}")
        if 'component_details' in details:
            print("  Component details:")
            for comp in details['component_details']:
                print(f"    {comp['id']}: {comp['pos_sequence']}")
        print(f"  Function: {details.get('function', 'Unknown')}")

    # Demonstrate prediction for partial sequence
    print("\n--- Prediction for Partial Sequence ---")
    partial = ['DET', 'ADJ']
    print(f"Partial sequence: {partial}")
    pred_results = system.predict_for_partial_sequence(partial)

    print("Next POS predictions:")
    if 'next_pos_predictions' in pred_results and pred_results['next_pos_predictions']:
        for pos, prob in pred_results['next_pos_predictions'][:3]:
            print(f"  {pos}: {prob:.2f}")
    else:
        # Try direct access to predictions
        predictions = pred_results.get('predictions', {})
        if 'next_pos' in predictions:
            next_pos = predictions['next_pos']
            sorted_preds = sorted(next_pos.items(), key=lambda x: x[1], reverse=True)

            for pos, prob in sorted_preds[:3]:
                print(f"  {pos}: {prob:.2f}")
        else:
            print("  No predictions available.")

    # Demonstrate adding a new construction
    print("\n--- Adding a New Construction ---")
    new_construction = ('O', 'QUE', 'NOUN','ESTÁ','FAZENDO','NOUN')
    const_id = system.add_predefined_construction(new_construction)
    print(f"Added construction {const_id}: {new_construction}")

    # Process a sequence with the new construction
    test_sequence = ['O', 'QUE', 'DET','NOUN','ESTÁ','FAZENDO','ADP','DET','NOUN']
    print(f"\nProcessing sequence with new construction: {test_sequence}")
    results = system.process_sequence(test_sequence)

    # Print identified constructions
    print("\nIdentified Constructions (should include the new one):")
    if 'combined' in results and 'constructions' in results['combined']:
        constructions = results['combined']['constructions']
        for const_type in ['predefined', 'new', 'composite']:
            if const_type in constructions:
                print(f"  {const_type.capitalize()}:")
                for const in constructions[const_type]:
                    const_id = const['id']
                    start = const['start']
                    end = const['end']
                    const_sequence = test_sequence[start:end]
                    print(f"    {const_id}: {const_sequence} (position {start}-{end})")



if __name__ == "__main__":
    main()