# Predictive Coding Construction Grammar System

This repository implements a Predictive Coding framework for Construction Grammar, allowing for the identification, learning, and prediction of grammatical constructions in language processing.

## Overview

The system processes part-of-speech (POS) sequences using predictive coding principles, identifying constructions of varying sizes and establishing hierarchical relationships between them. It features:

- Bidirectional processing for improved accuracy
- Multi-level attention mechanisms
- Support for predefined constructions
- Hierarchical construction relationships
- Automated discovery of new constructions

## System Architecture

The implementation follows a modular design with the following components:

### Base Module (`base_module.py`)
- Core graph functionality for POS sequence processing
- Transition probability calculations
- Basic sequence operations and statistics

### Construction Module (`construction_module.py`)
- Identification and management of constructions
- Registration of predefined constructions
- Discovery of new and composite constructions
- Hierarchical and specialization relationships

### Attention Module (`attention_module.py`)
- Multi-level attention mechanisms
- POS-level, construction-level, and cross-level attention
- Attention-weighted learning

### Bidirectional Module (`bidirectional_module.py`)
- Processing sequences in both directions
- Combining evidence from forward and backward passes
- Adaptive direction weighting

### Predictive Coding Module (`predictive_coding_module.py`)
- Multi-level prediction generation
- Prediction error calculation
- Model updating based on errors

### Main Module (`main_module.py`)
- Integration of all components through multiple inheritance
- Primary interface for using the system

## Usage

### Basic Usage

```python
from main_module import MainModule

# Define predefined constructions (if any)
predefined_constructions = [
    ('DET', 'NOUN'),
    ('DET', 'ADJ', 'NOUN'),
    ('VERB', 'DET', 'NOUN')
]

# Initialize the system
system = MainModule(predefined_constructions=predefined_constructions)

# Process a POS sequence
pos_sequence = ['DET', 'ADJ', 'NOUN', 'VERB', 'DET', 'NOUN']
results = system.process_sequence(pos_sequence, bidirectional=True)

# Access results
constructions = results['combined']['constructions']
attention = results['combined']['attention']
predictions = results['combined']['predictions']
errors = results['combined']['prediction_error']
```

### Adding New Constructions

```python
# Add a new construction
new_construction = ('PREP', 'DET', 'NOUN')
const_id = system.add_predefined_construction(new_construction)
```

### Getting Construction Details

```python
# Get details for a specific construction
const_id = 'pre_0'  # ID of the construction
details = system.get_construction_details(const_id)
```

### Predicting from Partial Sequences

```python
# Make predictions from a partial sequence
partial = ['DET', 'ADJ']
predictions = system.predict_for_partial_sequence(partial)
```

## Construction Grammar Principles

This implementation aligns with Construction Grammar principles by:

1. **Treating constructions as core units**: Constructions are form-meaning pairings of varying sizes
2. **Supporting single-POS constructions**: Even individual POS tags can be treated as constructions
3. **Modeling part-whole relationships**: Constructions can be composed of other constructions
4. **Bidirectional processing**: Constructions create expectations in both directions
5. **Functional annotation**: Constructions can be annotated with their grammatical functions

## Extending the System

The modular design makes it easy to extend the system:

- Add semantic information to constructions by modifying the `construction_registry`
- Implement additional attention mechanisms in the `AttentionModule`
- Add new prediction strategies in the `PredictiveCodingModule`
- Integrate with external ontologies or resources

## Example Scripts

- `example_usage.py`: Demonstrates basic usage of the system
- `comparison_script.py`: Compares different configurations (bidirectional vs. unidirectional)

## Requirements

- Python 3.7+
- NumPy
- Matplotlib (for visualization in the comparison script)

## Future Enhancements

- Integration with semantic information from ontologies
- Support for construction-based parsing
- Statistical optimization of attention mechanisms
- Dynamic adaptation of learning rates
- Visualization tools for constructions and their relationships

## License

[MIT License](LICENSE)