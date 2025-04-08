"""
Comparison Script for the Predictive Coding Construction Grammar System.

This script compares the performance of different configurations
of the system (bidirectional vs. unidirectional, with/without attention).
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from main_module import MainModule

def generate_test_sequences(num_sequences=20, min_length=5, max_length=15):
    """
    Generate test sequences for evaluation.

    Args:
        num_sequences: Number of sequences to generate
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        list: List of test sequences
    """
    # POS tags to use
    pos_tags = ['DET', 'NOUN', 'VERB', 'ADJ', 'PREP', 'ADV', 'CONJ', 'PRON']

    # Predefined patterns to include
    patterns = [
        ['DET', 'NOUN'],
        ['DET', 'ADJ', 'NOUN'],
        ['VERB', 'DET', 'NOUN'],
        ['NOUN', 'VERB'],
        ['PREP', 'DET', 'NOUN']
    ]

    sequences = []

    for _ in range(num_sequences):
        # Determine sequence length
        length = np.random.randint(min_length, max_length + 1)

        # Start with an empty sequence
        sequence = []

        # Fill the sequence
        while len(sequence) < length:
            # Decide whether to add a pattern or a random tag
            if len(sequence) < length - 1 and np.random.random() < 0.7:
                # Add a pattern
                pattern = patterns[np.random.randint(0, len(patterns))]

                # Only add if it fits
                if len(sequence) + len(pattern) <= length:
                    sequence.extend(pattern)
                else:
                    # Add a random tag
                    sequence.append(pos_tags[np.random.randint(0, len(pos_tags))])
            else:
                # Add a random tag
                sequence.append(pos_tags[np.random.randint(0, len(pos_tags))])

        sequences.append(sequence)

    return sequences

def evaluate_system(system, test_sequences):
    """
    Evaluate the system's performance on test sequences.

    Args:
        system: MainModule instance
        test_sequences: List of test sequences

    Returns:
        dict: Evaluation metrics
    """
    # Reset the system
    system.reset()

    total_constructions = 0
    total_error = 0.0
    processing_times = []

    # Process each sequence
    for sequence in test_sequences:
        # Measure processing time
        start_time = time.time()
        results = system.process_sequence(sequence)
        end_time = time.time()

        processing_times.append(end_time - start_time)

        # Count identified constructions
        if 'combined' in results and 'constructions' in results['combined']:
            constructions = results['combined']['constructions']
            if 'all' in constructions:
                total_constructions += len(constructions['all'])

        # Add prediction error
        if 'combined' in results and 'prediction_error' in results['combined']:
            total_error += results['combined']['prediction_error'].get('total_error', 0.0)

    # Calculate metrics
    avg_constructions = total_constructions / len(test_sequences)
    avg_error = total_error / len(test_sequences)
    avg_time = sum(processing_times) / len(processing_times)

    return {
        'avg_constructions': avg_constructions,
        'avg_error': avg_error,
        'avg_time': avg_time,
        'processing_times': processing_times
    }

def compare_configurations():
    """
    Compare different configurations of the system.

    Returns:
        dict: Comparison results
    """
    # Predefined constructions
    predefined_constructions = [
        ('DET', 'NOUN'),
        ('DET', 'ADJ', 'NOUN'),
        ('VERB', 'DET', 'NOUN'),
        ('NOUN', 'VERB'),
        ('DET', 'NOUN', 'VERB', 'DET', 'NOUN'),
        ('PREP', 'DET', 'NOUN')
    ]

    # Generate test sequences
    print("Generating test sequences...")
    test_sequences = generate_test_sequences(num_sequences=50)

    # Define configurations to test
    configurations = {
        'Unidirectional': {'bidirectional': False},
        'Bidirectional': {'bidirectional': True}
    }

    results = {}

    for config_name, config_params in configurations.items():
        print(f"\nEvaluating configuration: {config_name}")

        # Initialize the system
        system = MainModule(predefined_constructions=predefined_constructions)

        # Evaluate
        results[config_name] = evaluate_system(system, test_sequences)

        # Print results
        print(f"  Average constructions identified: {results[config_name]['avg_constructions']:.2f}")
        print(f"  Average prediction error: {results[config_name]['avg_error']:.4f}")
        print(f"  Average processing time: {results[config_name]['avg_time']:.4f} seconds")

    return {
        'configurations': configurations,
        'results': results,
        'test_sequences': test_sequences
    }

def plot_comparison_results(comparison_results):
    """
    Plot comparison results.

    Args:
        comparison_results: Results from compare_configurations
    """
    results = comparison_results['results']
    config_names = list(results.keys())

    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot average constructions
    avg_constructions = [results[config]['avg_constructions'] for config in config_names]
    axes[0].bar(config_names, avg_constructions)
    axes[0].set_title('Average Constructions Identified')
    axes[0].set_ylabel('Number of Constructions')

    # Plot average prediction error
    avg_errors = [results[config]['avg_error'] for config in config_names]
    axes[1].bar(config_names, avg_errors)
    axes[1].set_title('Average Prediction Error')
    axes[1].set_ylabel('Error')

    # Plot average processing time
    avg_times = [results[config]['avg_time'] for config in config_names]
    axes[2].bar(config_names, avg_times)
    axes[2].set_title('Average Processing Time')
    axes[2].set_ylabel('Time (seconds)')

    plt.tight_layout()
    plt.savefig('comparison_results.png')
    plt.show()

def main():
    """
    Main function to run comparisons.
    """
    print("Comparing system configurations...")
    comparison_results = compare_configurations()

    print("\nPlotting results...")
    plot_comparison_results(comparison_results)

    print("\nComparison complete. Results saved to 'comparison_results.png'")

if __name__ == "__main__":
    main()