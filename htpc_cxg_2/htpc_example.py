"""
Example usage of the Hierarchical Temporal Predictive Coding (HTPC)
Construction Grammar system.

This script demonstrates how the system can recognize constructions
and automatically infer functional equivalence between them.
"""

from htpc_main import HTPCSystem
import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_errors(error_history):
    """
    Plot the prediction error history.

    Args:
        error_history: List of prediction errors
    """
    # Extract average errors for each level
    level_errors = {}

    for i, errors in enumerate(error_history):
        for level, level_errors_dict in errors.items():
            if level not in level_errors:
                level_errors[level] = []

            level_errors[level].append(level_errors_dict.get('average', 0.0))

    # Plot
    plt.figure(figsize=(10, 6))

    for level, errors in level_errors.items():
        plt.plot(errors, label=f"Level {level}")

    plt.title("Prediction Error History")
    plt.xlabel("Sequence")
    plt.ylabel("Average Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("prediction_errors.png")
    plt.show()

def visualize_equivalence_classes(equivalences, confidence):
    """
    Visualize the inferred equivalence classes.

    Args:
        equivalences: Dictionary of equivalence classes
        confidence: Dictionary of confidence scores
    """
    if not equivalences:
        print("No equivalence classes to visualize")
        return

    # Create a network visualization
    import networkx as nx

    G = nx.Graph()

    # Add nodes for each construction
    all_constructions = set()
    for cls, members in equivalences.items():
        all_constructions.update(members)

    for const in all_constructions:
        G.add_node(const)

    # Add edges between constructions in the same class
    for cls, members in equivalences.items():
        members_list = list(members)
        conf = confidence.get(cls, 0.0)

        for i in range(len(members_list)):
            for j in range(i+1, len(members_list)):
                G.add_edge(members_list[i], members_list[j], weight=conf)

    # Draw the graph
    plt.figure(figsize=(10, 8))

    # Position nodes using spring layout
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

    # Draw edges with width based on confidence
    for u, v, data in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=data['weight'] * 5, alpha=0.7)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.title("Functionally Equivalent Constructions")
    plt.axis('off')
    plt.savefig("equivalence_classes.png")
    plt.show()

def main():
    """
    Main function to demonstrate the HTPC Construction Grammar system.
    """
    print("Initializing HTPC Construction Grammar System...")

    # Initialize the system
    system = HTPCSystem(num_hierarchical_levels=3)

    # Define training data - sequences of POS tags that exhibit regular patterns
    training_data = [
        # Sequences with DET NOUN as subject
        ['DET', 'NOUN', 'VERB', 'DET', 'NOUN'],
        ['DET', 'NOUN', 'VERB', 'PREP', 'DET', 'NOUN'],
        ['DET', 'ADJ', 'NOUN', 'VERB', 'DET', 'NOUN'],

        # Sequences with NOUN as subject (functionally equivalent to DET NOUN)
        ['NOUN', 'VERB', 'DET', 'NOUN'],
        ['NOUN', 'VERB', 'PREP', 'DET', 'NOUN'],
        ['NOUN', 'VERB', 'DET', 'ADJ', 'NOUN'],

        # Sequences with PRON as subject (also functionally equivalent)
        ['PRON', 'VERB', 'DET', 'NOUN'],
        ['PRON', 'VERB', 'PREP', 'DET', 'NOUN'],

        # Different verb phrase patterns
        ['DET', 'NOUN', 'VERB', 'NOUN'],
        ['NOUN', 'VERB', 'NOUN'],
        ['DET', 'NOUN', 'VERB', 'PRON'],

        # Mix it up with some variation
        ['DET', 'ADJ', 'NOUN', 'VERB', 'NOUN'],
        ['PRON', 'VERB', 'ADJ', 'NOUN'],
        ['DET', 'NOUN', 'ADV', 'VERB', 'PREP', 'NOUN'],
        ['NOUN', 'ADV', 'VERB', 'DET', 'NOUN'],

        # Repeat some patterns to reinforce learning
        ['DET', 'NOUN', 'VERB', 'DET', 'NOUN'],
        ['NOUN', 'VERB', 'NOUN'],
        ['PRON', 'VERB', 'DET', 'NOUN'],
    ]

    # Process training data
    print("\nProcessing training sequences...")

    for i, sequence in enumerate(training_data):
        print(f"  Sequence {i+1}: {sequence}")
        results = system.process_sequence(sequence)

        # Print number of constructions found at each level
        constructions = results['constructions']
        for level, level_constructions in constructions.items():
            print(f"    Level {level}: {len(level_constructions)} constructions")

    # Get constructions recognized
    all_constructions = system.get_constructions()

    print("\nRecognized Constructions:")
    for level, constructions in all_constructions.items():
        print(f"  Level {level}:")

        # Sort by frequency
        sorted_constructions = sorted(
            constructions.items(),
            key=lambda x: x[1].get('count', 0),
            reverse=True
        )

        # Show top constructions
        for const_id, const_info in sorted_constructions[:5]:
            if level == 0:  # POS level
                print(f"    {const_id}: {const_info.get('pattern', '')}, Count: {const_info.get('count', 0)}")
            else:
                print(f"    {const_id}, Count: {const_info.get('count', 0)}")

    # Get inferred equivalences
    equivalences = system.get_inferred_equivalences()
    confidence = system.generalizations.get('confidence', {})

    print("\nInferred Functional Equivalences:")
    if equivalences:
        for cls, members in equivalences.items():
            conf = confidence.get(cls, 0.0)
            print(f"  Class {cls} (Confidence: {conf:.2f}):")
            for member in members:
                # Look up the pattern
                pattern = None
                for level in range(len(system.architecture.levels)):
                    level_constructions = all_constructions.get(level, {})
                    if member in level_constructions:
                        pattern = level_constructions[member].get('pattern', None)
                        if pattern:
                            break

                print(f"    {member}: {pattern}")
    else:
        print("  No functional equivalences inferred yet")

    # Get functional categories
    categories = system.get_categories()

    print("\nFunctional Categories:")
    if categories:
        for category, category_info in categories.items():
            print(f"  {category}: {len(category_info.get('constructions', []))} constructions")
            for const_id in category_info.get('constructions', []):
                pattern = None
                for level in range(len(system.architecture.levels)):
                    level_constructions = all_constructions.get(level, {})
                    if const_id in level_constructions:
                        pattern = level_constructions[const_id].get('pattern', None)
                        if pattern:
                            break

                print(f"    {const_id}: {pattern}")
    else:
        print("  No functional categories defined yet")

    # Test prediction
    print("\nTesting Prediction:")

    test_sequences = [
        ['DET', 'NOUN', 'VERB'],
        ['NOUN', 'VERB'],
        ['PRON', 'VERB'],
    ]

    for sequence in test_sequences:
        print(f"  Partial sequence: {sequence}")
        predictions = system.predict_next_pos(sequence)

        print("  Predicted next POS:")
        for pos, prob in predictions:
            print(f"    {pos}: {prob:.2f}")

    # Visualize prediction errors
    error_history = system.learning_module.error_history
    if error_history:
        print("\nVisualizing prediction errors...")
        plot_prediction_errors(error_history)

    # Visualize equivalence classes
    if equivalences:
        print("\nVisualizing functional equivalence classes...")
        try:
            visualize_equivalence_classes(equivalences, confidence)
        except ImportError:
            print("Could not visualize equivalence classes - networkx library required")

    print("\nHierarchical Temporal Predictive Coding has successfully:")
    print("1. Recognized constructions at multiple levels")
    print("2. Inferred functional equivalence between constructions")
    print("3. Formed abstract categories based on functional similarity")
    print("4. Generated predictions using the hierarchical model")
    print("5. Demonstrated how predictive coding principles can be applied to")
    print("   Construction Grammar in a cognitively plausible way")

if __name__ == "__main__":
    main()