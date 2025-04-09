# Hierarchical Temporal Predictive Coding for Construction Grammar

This project implements a Hierarchical Temporal Predictive Coding (HTPC) framework for Construction Grammar, capable of recognizing, learning, and generalizing grammatical constructions from sequential data.

## Overview

The system processes part-of-speech (POS) sequences using hierarchical predictive coding principles, identifying constructions at multiple levels of abstraction and automatically inferring functional equivalence between constructions.

Key features:
- Multiple hierarchical levels with increasing temporal scope
- Automatic discovery of constructions and patterns
- Inference of functional equivalence between constructions
- Prediction at multiple levels of abstraction
- Bidirectional processing with top-down and bottom-up information flow
- Self-organization of abstract grammatical categories

## Architecture

The implementation follows a hierarchical architecture with the following components:

### HTPC Architecture (`htpc_architecture.py`)
- Defines the hierarchical structure with multiple processing levels
- Coordinates bidirectional information flow between levels
- Integrates bottom-up processing, top-down predictions, and prediction errors

### Level-specific Components
- **POSLevel**: Lowest level dealing with individual POS tags and simple n-grams
- **ConstructionLevel**: Middle level handling constructions of varying complexity
- **CategoryLevel**: Highest level handling abstract categories and template constructions

### Learning Module (`htpc_learning_module.py`)
- Implements mechanisms for learning from prediction errors
- Analyzes construction co-occurrence and substitution patterns
- Infers functional equivalence between constructions
- Discovers and forms abstract grammatical categories

### Main System (`htpc_main.py`)
- Provides the main interface for using the system
- Coordinates processing, learning, and generalization
- Manages system state and history

## Construction Grammar and Predictive Coding

This implementation combines Construction Grammar principles with predictive coding in a novel way:

1. **Constructions as form-meaning pairings**: The system treats constructions as fundamental units that can be nested and combined
2. **Functional equivalence**: The system automatically infers when different constructions serve the same grammatical function
3. **Abstraction through prediction**: Categories emerge naturally from the system's attempt to minimize prediction error
4. **Hierarchical processing**: Information flows both bottom-up and top-down, allowing for rich contextual processing
5. **Error-driven learning**: The system learns by continuously predicting upcoming elements and updating based on errors

## Usage

### Basic Usage

```python
from htpc_main import HTPCSystem

# Initialize the system
system = HTPCSystem(num_hierarchical_levels=3)

# Process sequences
pos_sequence = ['DET', 'NOUN', 'VERB', 'DET', 'NOUN']
results = system.process_sequence(pos_sequence)

# Get recognized constructions
constructions = system.get_constructions()

# Get inferred functional equivalences
equivalences = system.get_inferred_equivalences()

# Make predictions for a partial sequence
partial = ['DET', 'NOUN']
predictions = system.predict_next_pos(partial, k=3)
```

### Example Script

The `htpc_example.py` script demonstrates the system's capabilities:
- Processing training sequences
- Identifying constructions at multiple levels
- Inferring functional equivalence between constructions
- Making predictions based on the learned model
- Visualizing prediction errors and equivalence classes

## Key Theoretical Aspects

### Hierarchical Processing

The system implements a hierarchy of processing levels:
1. **Level 0 (POS)**: Processes individual POS tags and simple n-grams
2. **Level 1 (Construction)**: Identifies and manages constructions of varying complexity
3. **Level 2 (Category)**: Abstracts over constructions to form functional categories and templates

Each level maintains its own representation, predictions, and error calculations, with information flowing bidirectionally between adjacent levels.

### Functional Equivalence Inference

The system infers functional equivalence through:
1. **Context similarity**: Constructions that appear in similar contexts are likely functionally equivalent
2. **Substitution patterns**: Constructions that can substitute for each other are considered for equivalence
3. **Prediction similarity**: Constructions that lead to similar predictions are grouped together
4. **Error-based feedback**: Equivalence hypotheses that lead to lower prediction error are reinforced

### Emergent Categories

Rather than pre-defining grammatical categories, the system allows them to emerge through:
1. **Statistical regularities**: Patterns of co-occurrence and substitution
2. **Functional similarity**: Common behavior in the larger syntactic context
3. **Hierarchical feedback**: Top-down influence from templates and higher-level patterns
4. **Error minimization**: Categories that minimize overall prediction error are favored

## Extending the System

The modular design makes it easy to extend the system:
- Add more hierarchical levels for deeper abstraction
- Incorporate semantic information to model meaning alongside form
- Integrate with external linguistic resources
- Add more sophisticated statistical methods for pattern recognition
- Implement additional visualization and analysis tools

## Requirements

- Python 3.7+
- NumPy
- Matplotlib (for visualization)
- NetworkX (for visualizing equivalence classes)

## Future Directions

- Integration with semantic information
- Extension to handle lexical items alongside POS tags
- Support for cross-linguistic variation in constructions
- Implementation of incremental processing for real-time applications
- Addition of more sophisticated attention mechanisms
- Integration with larger language models

## License

[MIT License](LICENSE)